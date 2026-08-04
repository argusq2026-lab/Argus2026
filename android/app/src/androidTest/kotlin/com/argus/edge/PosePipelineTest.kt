package com.argus.edge

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The whole edge path on device, minus the camera: bitmap → ROI crop → NPU
 * landmarks → COCO-17 → a PROTOCOL.md observation.
 *
 * The camera cannot be driven from a test, but everything downstream of it can,
 * and that is where the wiring bugs live — a transposed crop, a ROI mapped back
 * with the wrong scale, keypoints normalized against the wrong frame. Feeding a
 * synthetic bitmap with a hand-made Detection exercises all of it
 * deterministically.
 *
 * What this does *not* claim: that the landmarks are anatomically correct. The
 * synthetic figure is a stick drawing, not a person, so the assertions are
 * about wiring and invariants (shape, range, normalization, unmapped joints),
 * not accuracy. Accuracy needs real footage — `docs/VALIDATION.md` §1.
 */
@RunWith(AndroidJUnit4::class)
class PosePipelineTest {

    private fun poseModel(): File? {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        return File(ctx.filesDir, "models/pose_landmark_fp32.onnx").takeIf { it.isFile }
    }

    /** A crude upright figure — enough structure for the network to respond to. */
    private fun syntheticFrame(width: Int, height: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.rgb(30, 30, 40))
        val body = Paint().apply { color = Color.rgb(220, 190, 160); isAntiAlias = true }
        val cx = width / 2f
        c.drawCircle(cx, height * 0.18f, height * 0.07f, body)          // head
        c.drawRect(cx - width * 0.09f, height * 0.26f, cx + width * 0.09f, height * 0.62f, body)
        val limb = Paint().apply {
            color = Color.rgb(200, 170, 140); strokeWidth = width * 0.05f; isAntiAlias = true
        }
        c.drawLine(cx - width * 0.09f, height * 0.30f, cx - width * 0.22f, height * 0.52f, limb)
        c.drawLine(cx + width * 0.09f, height * 0.30f, cx + width * 0.22f, height * 0.52f, limb)
        c.drawLine(cx - width * 0.05f, height * 0.62f, cx - width * 0.08f, height * 0.92f, limb)
        c.drawLine(cx + width * 0.05f, height * 0.62f, cx + width * 0.08f, height * 0.92f, limb)
        return bmp
    }

    @Test
    fun theFullPosePathProducesAProtocolObservation() {
        val model = poseModel()
        assumeTrue("SKIP: stage pose_landmark_fp32.onnx (see android/README.md)", model != null)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val backend = File(ctx.applicationInfo.nativeLibraryDir, "libQnnHtp.so")

        val width = 720
        val height = 1280
        val frame = syntheticFrame(width, height)
        // Where a detector would have put the person.
        val det = Detection(width * 0.22f, height * 0.10f, width * 0.78f, height * 0.95f, 0.91f)

        val estimator = PoseEstimator(model!!, backend)
        try {
            // minScore 0f: the synthetic figure is not a person and the model's
            // own confidence is not what is under test here.
            val pose = estimator.estimate(frame, det, minScore = 0f)
            assertTrue("pose estimator returned null with minScore=0", pose != null)
            pose!!

            assertEquals(34, pose.keypointsXy.size)
            assertEquals(17, pose.keypointsConf.size)

            // Every confidence is a probability — proves the sigmoid ran.
            for (k in 0 until 17) {
                assertTrue(
                    "keypoint $k confidence ${pose.keypointsConf[k]} outside [0,1]",
                    pose.keypointsConf[k] in 0f..1f,
                )
            }
            // Unmapped COCO joints must stay exactly zero, never guessed.
            for (k in 13..16) assertEquals(0f, pose.keypointsConf[k])

            // Mapped joints land inside the ROI, which is what proves the
            // back-projection scale and offset are right.
            val roi = squareRoiFor(det)
            for (k in 0 until 13) {
                val x = pose.keypointsXy[k * 2]
                val y = pose.keypointsXy[k * 2 + 1]
                assertTrue(
                    "keypoint $k at ($x, $y) fell outside its ROI",
                    x >= roi.x0 - 1f && x <= roi.x0 + roi.side + 1f &&
                        y >= roi.y0 - 1f && y <= roi.y0 + roi.side + 1f,
                )
            }

            // And the observation the laptop would receive is well-formed.
            val obs = Observation.fromDetection(
                ts = 1730649600.0, det = det, pose = pose,
                frameWidth = width, frameHeight = height,
            )
            assertEquals(NUM_KEYPOINTS, obs.keypointsXy.size)
            assertTrue(obs.bboxNorm.all { it in 0.0..1.0 })
            assertTrue(obs.keypointsXy.all { p -> p.all { it in 0.0..1.0 } })
            assertTrue(obs.keypointsConf.all { it in 0.0..1.0 })
            // Encodes without throwing, which is the server's acceptance contract.
            val json = encodeObservation(obs)
            assertTrue(json.contains("\"keypoints_conf\""))
            println("POSE PIPELINE: observation ok, ${json.length} bytes, score ${pose.poseScore}")
        } finally {
            estimator.close()
            frame.recycle()
        }
    }
}
