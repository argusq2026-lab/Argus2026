package com.argus.edge

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Base64
import kotlin.math.abs

/**
 * Same input, same model, different silicon: the NPU must agree with the CPU
 * reference within stated tolerances.
 *
 * `tests/data/yolox_parity.json` (packaged as an androidTest asset straight
 * from the repo's tests/data — one file, both platforms) freezes what
 * onnxruntime's CPU EP computed on the procedural pattern input. This test
 * rebuilds that exact input on the device, runs the *staged real model* on the
 * Hexagon with CPU fallback disabled, and compares raw quantized outputs
 * anchor by anchor.
 *
 * Skips (rather than fails) when no model is staged — the model is 9 MB of
 * gitignored artifact, not a test asset. Staging is two adb commands; see
 * android/README.md. A skip is loud in the test report, not silent.
 */
@RunWith(AndroidJUnit4::class)
class NpuParityTest {

    private val fixture: JSONObject by lazy {
        InstrumentationRegistry.getInstrumentation().context.assets
            .open("yolox_parity.json").use { JSONObject(String(it.readBytes())) }
    }

    private fun stagedModel(): Pair<File, File>? {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val models = File(ctx.filesDir, "models")
        val onnx = File(models, "yolox.onnx")
        val sidecar = File(models, "yolox.json")
        return if (onnx.isFile && sidecar.isFile) onnx to sidecar else null
    }

    @Test
    fun theNpuReproducesTheCpuReferenceOnTheFrozenInput() {
        val staged = stagedModel()
        assumeTrue(
            "SKIP: no model staged at files/models/yolox.onnx — see android/README.md",
            staged != null,
        )
        val (onnx, sidecarFile) = staged!!
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext

        // The staged model must be the exact artifact the fixture was generated
        // from — comparing outputs of two different models proves nothing.
        val sidecar = ModelSidecar.load(sidecarFile)
        assertEquals(
            "staged model is not the fixture's model — regenerate one or the other",
            fixture.getString("model_sha256"), sidecar.modelSha256,
        )

        val detector = QnnDetector(
            onnx, sidecarFile, File(ctx.applicationInfo.nativeLibraryDir, "libQnnHtp.so"),
        )
        try {
            val (boxesQ, scoresQ, classQ) = detector.runRaw(patternInput(detector.inputSize))

            val tolerances = fixture.getJSONObject("tolerances")
            val scoreTol = tolerances.getInt("score_q_lsb")
            val boxTol = tolerances.getInt("boxes_q_lsb")
            val top = fixture.getJSONArray("top50_anchors")

            var worstScore = 0
            var worstBox = 0
            for (i in 0 until top.length()) {
                val a = top.getJSONObject(i)
                val idx = a.getInt("index")
                val scoreDelta = abs((scoresQ[idx].toInt() and 0xFF) - a.getInt("score_q"))
                worstScore = maxOf(worstScore, scoreDelta)
                val eBoxes = a.getJSONArray("boxes_q")
                for (k in 0 until 4) {
                    val boxDelta = abs((boxesQ[idx * 4 + k].toInt() and 0xFF) - eBoxes.getInt(k))
                    worstBox = maxOf(worstBox, boxDelta)
                }
            }
            println("NPU-vs-CPU parity: worst score delta $worstScore LSB, worst box delta $worstBox LSB")
            assertTrue(
                "NPU score deviates $worstScore LSB from CPU reference (tolerance $scoreTol)",
                worstScore <= scoreTol,
            )
            assertTrue(
                "NPU box coordinate deviates $worstBox LSB (tolerance $boxTol)",
                worstBox <= boxTol,
            )

            // And the decode chain agrees end to end on the crafted cases,
            // on-device, using the sidecar exactly as production does.
            val cases = fixture.getJSONArray("decode_cases")
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                val raw = case.getJSONObject("raw_b64")
                fun b64(key: String) = Base64.getDecoder().decode(raw.getString(key))
                val got = decodeDetections(
                    b64("boxes"), b64("scores"), b64("class_idx"),
                    sidecar, sidecar.scoreThreshold,
                    LetterboxInfo(1f, 0, 0), 640, 640,
                )
                assertEquals(
                    "${case.getString("name")}: decode disagrees on device",
                    case.getJSONArray("expected").length(), got.size,
                )
            }
        } finally {
            detector.close()
        }
    }
}
