package com.argus.edge

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Base64

/**
 * The Kotlin decode against the Python reference, on the host JVM.
 *
 * `tests/data/yolox_parity.json` carries crafted quantized tensors and what the
 * (deleted, history-pinned) PC reference decode says about them. No model and
 * no device is involved here — any disagreement is arithmetic in
 * [decodeDetections], not quantization noise, which is exactly what makes this
 * runnable in CI on every commit.
 *
 * The identity letterbox (scale 1, no offset) keeps expectations in canvas
 * space, matching the fixture's `bbox_canvas_xyxy`.
 */
class DecodeTest {

    private val fixture: JSONObject by lazy {
        val dir = System.getProperty("argus.fixtures")
            ?: error("argus.fixtures system property not set; see app/build.gradle.kts")
        JSONObject(File(dir, "yolox_parity.json").readText())
    }

    private val sidecar: ModelSidecar by lazy {
        val q = fixture.getJSONObject("quant")
        ModelSidecar(
            modelFile = "fixture", modelSha256 = fixture.getString("model_sha256"),
            inputName = "image", inputShape = listOf(1, 3, 640, 640),
            boxes = q.getJSONObject("boxes").let { QuantParams(it.getDouble("scale"), it.getInt("zero_point")) },
            scores = q.getJSONObject("scores").let { QuantParams(it.getDouble("scale"), it.getInt("zero_point")) },
            personClassIndex = 0, scoreThreshold = 0.35, nmsIouThreshold = 0.45,
            maxDetections = 64, letterboxPadValue = 114,
        )
    }

    private val identityLetterbox = LetterboxInfo(scale = 1f, left = 0, top = 0)

    private fun b64(obj: JSONObject, key: String): ByteArray =
        Base64.getDecoder().decode(obj.getString(key))

    @Test
    fun `every crafted decode case matches the python reference exactly`() {
        val cases = fixture.getJSONArray("decode_cases")
        assertTrue("fixture has no decode cases", cases.length() >= 4)

        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val name = case.getString("name")
            val raw = case.getJSONObject("raw_b64")

            val got = decodeDetections(
                boxesQ = b64(raw, "boxes"), scoresQ = b64(raw, "scores"),
                classQ = b64(raw, "class_idx"), sidecar = sidecar,
                scoreThreshold = sidecar.scoreThreshold,
                letterbox = identityLetterbox, srcWidth = 640, srcHeight = 640,
            )
            val expected = case.getJSONArray("expected")
            assertEquals("$name: detection count", expected.length(), got.size)

            for (j in 0 until expected.length()) {
                val e = expected.getJSONObject(j)
                val box = e.getJSONArray("bbox_canvas_xyxy")
                val g = got[j]
                assertEquals("$name[$j].score", e.getDouble("score"), g.score.toDouble(), 1e-4)
                for ((k, v) in listOf(g.x0, g.y0, g.x1, g.y1).withIndex()) {
                    assertEquals("$name[$j].box[$k]", box.getDouble(k), v.toDouble(), 1e-2)
                }
            }
        }
    }

    @Test
    fun `the pattern generator matches the fixture's declaration`() {
        assertEquals(
            "(x*7 + y*13 + c*31) % 256, NCHW uint8 (1,3,640,640)",
            fixture.getString("input_pattern"),
        )
        val tensor = patternInput(640)
        assertEquals(3 * 640 * 640, tensor.size)
        // Spot-check the formula at known coordinates.
        val plane = 640 * 640
        assertEquals(0, tensor[0].toInt() and 0xFF)                       // x=0,y=0,c=0
        assertEquals((7 * 5 + 13 * 3) % 256, tensor[3 * 640 + 5].toInt() and 0xFF)
        assertEquals(31, tensor[plane].toInt() and 0xFF)                  // c=1 origin
    }

    @Test
    fun `decoding the pattern raw outputs reproduces the frozen top anchors`() {
        val raw = fixture.getJSONObject("raw_outputs_b64")
        val scoresQ = b64(raw, "scores")
        val classQ = b64(raw, "class_idx")
        val top = fixture.getJSONArray("top50_anchors")
        for (i in 0 until top.length()) {
            val a = top.getJSONObject(i)
            val idx = a.getInt("index")
            assertEquals("anchor $idx score_q", a.getInt("score_q"), scoresQ[idx].toInt() and 0xFF)
            assertEquals("anchor $idx class", a.getInt("class"), classQ[idx].toInt() and 0xFF)
        }
    }

    @Test
    fun `letterbox geometry round trips`() {
        val info = letterboxGeometry(1920, 1080, 640)
        // 1920 -> 640 => scale 1/3; 1080 * 1/3 = 360, centred: top = (640-360)/2 = 140
        assertEquals(640f / 1920f, info.scale, 1e-6f)
        assertEquals(0, info.left)
        assertEquals(140, info.top)
        val mapped = undoLetterbox(0f, 140f, 640f, 500f, info)
        assertEquals(0.0, mapped[0].toDouble(), 1e-3)
        assertEquals(0.0, mapped[1].toDouble(), 1e-3)
        assertEquals(1920.0, mapped[2].toDouble(), 1e-2)
        assertEquals(1080.0, mapped[3].toDouble(), 1e-2)
    }
}
