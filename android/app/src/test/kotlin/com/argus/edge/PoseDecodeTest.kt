package com.argus.edge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The pose decode arithmetic, on the host JVM.
 *
 * Pins the three things a wrong port would get silently wrong: the
 * BlazePose→COCO remap indices (a transposed index reads an eye as a
 * shoulder — the exact failure `d3bd15e:src/argus/vision/keypoints.py` exists
 * to prevent), the ROI back-projection, and the visibility activation
 * (`sigmoid(v * 256)`, because the network's last op divides the whole
 * landmark tensor by 256 and visibility is a logit — see Pose.kt).
 */
class PoseDecodeTest {

    private fun landmarks(fill: (Int) -> FloatArray): FloatArray {
        val out = FloatArray(25 * 4)
        for (i in 0 until 25) fill(i).copyInto(out, i * 4)
        return out
    }

    @Test
    fun `the remap table matches the historical reference`() {
        // Spot checks against d3bd15e:src/argus/vision/keypoints.py.
        assertEquals(0, COCO_FROM_BLAZEPOSE[0])    // nose -> nose
        assertEquals(2, COCO_FROM_BLAZEPOSE[1])    // left_eye (not inner/outer)
        assertEquals(11, COCO_FROM_BLAZEPOSE[5])   // left_shoulder
        assertEquals(12, COCO_FROM_BLAZEPOSE[6])   // right_shoulder
        assertEquals(16, COCO_FROM_BLAZEPOSE[10])  // right_wrist
        assertEquals(23, COCO_FROM_BLAZEPOSE[11])  // left_hip
        assertEquals(24, COCO_FROM_BLAZEPOSE[12])  // right_hip
        // Knees and ankles have no source in the 25-point export.
        for (coco in 13..16) assertEquals(-1, COCO_FROM_BLAZEPOSE[coco])
    }

    @Test
    fun `unmapped joints stay at zero confidence and origin`() {
        val result = decodePose(
            landmarks { floatArrayOf(0.5f, 0.5f, 0f, 10f / 256f) },
            PoseRoi(0f, 0f, 256f), poseScore = 0.9f,
        )
        for (coco in 13..16) {
            assertEquals(0f, result.keypointsConf[coco])
            assertEquals(0f, result.keypointsXy[coco * 2])
            assertEquals(0f, result.keypointsXy[coco * 2 + 1])
        }
        // Mapped joints did land.
        assertTrue(result.keypointsConf[5] > 0.9f)
    }

    @Test
    fun `roi back projection maps normalized landmarks to frame pixels`() {
        // left_shoulder is BlazePose 11; place it at (0.25, 0.75) of the ROI.
        val lm = landmarks { i ->
            if (i == 11) floatArrayOf(0.25f, 0.75f, 0f, 0.02f)
            else floatArrayOf(0f, 0f, 0f, -1f)
        }
        val roi = PoseRoi(x0 = 100f, y0 = 200f, side = 400f)
        val result = decodePose(lm, roi, 1f)
        assertEquals(100f + 0.25f * 400f, result.keypointsXy[5 * 2], 1e-3f)
        assertEquals(200f + 0.75f * 400f, result.keypointsXy[5 * 2 + 1], 1e-3f)
    }

    @Test
    fun `visibility is a logit divided by 256`() {
        // Raw logit 0 -> wire value 0 -> confidence exactly 0.5.
        val neutral = decodePose(
            landmarks { floatArrayOf(0f, 0f, 0f, 0f) }, PoseRoi(0f, 0f, 1f), 1f,
        )
        assertEquals(0.5f, neutral.keypointsConf[0], 1e-6f)

        // Raw logit +8 (wire 8/256) -> confidently visible; -8 -> confidently not.
        val visible = decodePose(
            landmarks { floatArrayOf(0f, 0f, 0f, 8f / 256f) }, PoseRoi(0f, 0f, 1f), 1f,
        )
        val hidden = decodePose(
            landmarks { floatArrayOf(0f, 0f, 0f, -8f / 256f) }, PoseRoi(0f, 0f, 1f), 1f,
        )
        assertTrue(visible.keypointsConf[0] > 0.99f)
        assertTrue(hidden.keypointsConf[0] < 0.01f)
    }

    @Test
    fun `square roi is scaled and shifted upward off the box centre`() {
        val det = Detection(100f, 100f, 200f, 400f, 0.9f)   // 100 x 300, tall
        val roi = squareRoiFor(det, scale = 2.0f, yOffset = -0.10f)
        assertEquals("side = longest edge x scale", 600f, roi.side, 1e-3f)
        assertEquals(150f - 300f, roi.x0, 1e-3f)
        // Centre shifts up by 10% of box height: 250 - 30 = 220.
        assertEquals(220f - 300f, roi.y0, 1e-3f)
    }

    @Test
    fun `the default roi is the measured one, not a tight box`() {
        // A 1.25x box scores 0.000 on the landmark network across every framing
        // tried; the usable plateau is 2.0x-3.0x with a slight upward shift.
        // Pinned so the default cannot drift back without this failing.
        assertEquals(2.2f, POSE_ROI_SCALE, 1e-6f)
        assertEquals(-0.10f, POSE_ROI_Y_OFFSET, 1e-6f)
        val roi = squareRoiFor(Detection(0f, 0f, 100f, 200f, 0.9f))
        assertEquals(440f, roi.side, 1e-3f)
    }

    @Test
    fun `an observation with pose carries real normalized keypoints`() {
        val pose = decodePose(
            landmarks { floatArrayOf(0.5f, 0.5f, 0f, 8f / 256f) },
            PoseRoi(0f, 0f, 1000f), 0.95f,
        )
        val obs = Observation.fromDetection(
            ts = 1.0, det = Detection(0f, 0f, 1000f, 1000f, 0.9f), pose = pose,
            frameWidth = 1000, frameHeight = 1000,
        )
        assertEquals(0.5, obs.keypointsXy[5][0], 1e-6)   // left_shoulder x
        assertTrue(obs.keypointsConf[5] > 0.99)
        assertEquals(0.0, obs.keypointsConf[13], 0.0)     // left_knee unmapped
    }
}
