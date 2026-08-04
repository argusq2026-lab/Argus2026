package com.argus.edge

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The pose path against an actual person, not a stick figure.
 *
 * `PosePipelineTest` proves the wiring but says nothing about whether the model
 * responds, because its synthetic figure is not a person. This uses the
 * landmark model's own sample input — a real, correctly framed person — pasted
 * into a larger frame at a known location, so the whole app path runs on it:
 * ROI from a detection box, crop, NPU, decode, COCO remap.
 *
 * It is the test that distinguishes "the skeleton is not drawing" from "the
 * skeleton is drawing in the wrong place" — the two failures that look
 * identical on a phone screen and have completely different causes.
 */
@RunWith(AndroidJUnit4::class)
class PoseRealPersonTest {

    private fun personBitmap(): Bitmap =
        InstrumentationRegistry.getInstrumentation().context.assets
            .open("pose_person_256.png").use { BitmapFactory.decodeStream(it) }

    @Test
    fun aRealPersonProducesAConfidentPoseThroughTheWholePath() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val model = File(ctx.filesDir, "models/pose_landmark_fp32.onnx")
        assumeTrue("SKIP: stage pose_landmark_fp32.onnx", model.isFile)

        // Put the person into a 720x1280 "camera frame" at a known place, the
        // way a detection box would sit inside a real frame.
        val frameW = 720
        val frameH = 1280
        val personLeft = 160
        val personTop = 400
        val personSize = 400
        val frame = Bitmap.createBitmap(frameW, frameH, Bitmap.Config.ARGB_8888)
        Canvas(frame).apply {
            drawColor(Color.rgb(20, 20, 25))
            drawBitmap(
                personBitmap(), null,
                Rect(personLeft, personTop, personLeft + personSize, personTop + personSize),
                null,
            )
        }

        // A detection box on the person, as YOLO would emit.
        val det = Detection(
            personLeft.toFloat(), personTop.toFloat(),
            (personLeft + personSize).toFloat(), (personTop + personSize).toFloat(),
            0.92f,
        )

        val estimator = PoseEstimator(
            model, File(ctx.applicationInfo.nativeLibraryDir, "libQnnHtp.so"),
        )
        try {
            val pose = estimator.estimate(frame, det, minScore = 0f)
            assertTrue("estimate returned null on a real person", pose != null)
            pose!!

            val visible = (0 until 17).count { pose.keypointsConf[it] >= 0.3f }
            val roi = squareRoiFor(det)
            // Where did the mapped joints actually land, relative to the person?
            val inside = (0 until 13).count { k ->
                val x = pose.keypointsXy[k * 2]
                val y = pose.keypointsXy[k * 2 + 1]
                x >= personLeft && x <= personLeft + personSize &&
                    y >= personTop && y <= personTop + personSize
            }
            println(
                "REAL PERSON POSE: score=%.3f visible=%d/13 onPerson=%d/13 roi=(%.0f,%.0f,%.0f)"
                    .format(pose.poseScore, visible, inside, roi.x0, roi.y0, roi.side)
            )

            // The model must actually respond to a person: most mapped joints
            // confident, and landing on the person rather than elsewhere in the
            // ROI. Both are needed -- confident-but-misplaced is the silent
            // failure this test exists to catch.
            // The network's own pose score is the assertion that matters, and
            // its absence here was a real gap: with a wrong ROI this test still
            // passed at 25/25 visible while poseScore was 0.003. Per-joint
            // visibility stays high on garbage landmarks; the pose score does
            // not. Check the thing that actually discriminates.
            assertTrue(
                "pose score %.3f -- the network rejects this framing; check the ROI"
                    .format(pose.poseScore),
                pose.poseScore >= 0.5f,
            )
            assertTrue("only $visible/13 keypoints were confident on a real person", visible >= 10)
            assertTrue("only $inside/13 keypoints landed on the person", inside >= 10)
        } finally {
            estimator.close()
            frame.recycle()
        }
    }
}
