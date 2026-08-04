package com.argus.edge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.FloatBuffer

/**
 * Can the BlazePose landmark model run on this phone's NPU?
 *
 * The feasibility question behind real `keypoints_xy`/`keypoints_conf` in the
 * protocol observations. The staged model is a **float32** export (13 MB,
 * `scripts/` docs) rather than w8a8 — deliberately: pose quantization is where
 * the placeholder-calibration damage documented in docs/VALIDATION.md §1
 * happened (a detector left with two effective confidence values), and this
 * HTP runs fp16 natively (`htp-supports-fp16:true`), so float-in, fp16-on-NPU
 * sidesteps calibration entirely until real footage exists to calibrate on.
 *
 * Same rules as everything else on this branch: CPU fallback disabled, so a
 * pass means the Hexagon genuinely took the graph; skip (loudly) when the
 * model is not staged.
 */
@RunWith(AndroidJUnit4::class)
class PoseFeasibilityTest {

    @Test
    fun theLandmarkModelRunsOnTheNpuInFp16() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = File(ctx.filesDir, "models/pose_landmark_fp32.onnx")
        assumeTrue("SKIP: stage pose_landmark_fp32.onnx first (see android/README.md)", model.isFile)

        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
        options.addQnn(
            mapOf(
                "backend_path" to File(ctx.applicationInfo.nativeLibraryDir, "libQnnHtp.so").absolutePath,
                "htp_performance_mode" to "burst",
                "enable_htp_fp16_precision" to "1",
            )
        )

        env.createSession(model.absolutePath, options).use { session ->
            assertEquals(setOf("image"), session.inputNames.toSet())
            assertEquals(setOf("scores", "landmarks"), session.outputNames.toSet())

            val input = FloatBuffer.allocate(1 * 3 * 256 * 256).apply {
                repeat(capacity()) { put(0.5f) }; rewind()
            }
            OnnxTensor.createTensor(env, input, longArrayOf(1, 3, 256, 256)).use { tensor ->
                val feed = mapOf("image" to tensor)
                repeat(5) { session.run(feed).close() }  // warm up
                val runs = 50
                val started = System.nanoTime()
                var landmarksShape: LongArray? = null
                repeat(runs) {
                    session.run(feed).use { result ->
                        landmarksShape = (result.get("landmarks").get() as OnnxTensor).info.shape
                    }
                }
                val usPerRun = (System.nanoTime() - started) / 1000.0 / runs
                println(
                    "POSE FEASIBILITY: landmark model on NPU fp16, " +
                        "%.0f us/inference, landmarks shape ${landmarksShape?.toList()}".format(usPerRun)
                )
                assertTrue(landmarksShape.contentEquals(longArrayOf(1, 25, 4)))
            }
        }
    }
}
