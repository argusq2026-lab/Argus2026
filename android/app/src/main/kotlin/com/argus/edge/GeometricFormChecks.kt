package com.argus.edge

import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Geometric form-fault checks: plain joint-angle arithmetic against a fixed
 * threshold, no logistic regression, no training data, no artifact to stage.
 *
 * `FormClassifier` covers the one fault per exercise upstream's dataset
 * actually labelled (`lean_back_error` for bicep, `knee_over_toe` for lunge).
 * Their own README documents a second fault for each that they never
 * collected ML labels for at all, checking it with a joint-angle threshold
 * instead: "loose upper arm" for bicep (elbow-shoulder angle from vertical),
 * "knee angle" for lunge (hip-knee-ankle angle at the bottom of a rep). Both
 * thresholds below are copied verbatim from their stated cutoffs -- neither
 * has been fit or evaluated by anyone, including upstream, against held-out
 * data; see docs/VALIDATION.md 1c/1d.
 *
 * Kept separate from `FormClassifier` because these need none of its
 * machinery -- no artifact, no scaler, no softmax, no confidence threshold --
 * just an angle and a range.
 */
object GeometricFormChecks {

    /** Angle in degrees at vertex `b`, between rays `b->a` and `b->c`. Range [0, 180]. */
    private fun angleAtVertex(
        ax: Double, ay: Double,
        bx: Double, by: Double,
        cx: Double, cy: Double,
    ): Double {
        val ux = ax - bx
        val uy = ay - by
        val vx = cx - bx
        val vy = cy - by
        val uLen = sqrt(ux * ux + uy * uy)
        val vLen = sqrt(vx * vx + vy * vy)
        if (uLen < 1e-9 || vLen < 1e-9) return Double.NaN
        val cos = ((ux * vx + uy * vy) / (uLen * vLen)).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    /**
     * Bicep's "loose upper arm": upstream's angle between the elbow, the
     * shoulder, and the shoulder's projection on the ground -- how far the
     * upper arm has swung from vertical, which should stay still during a
     * curl. Their stated cutoff is 40 degrees.
     *
     * Checked per side independently (a curl can load either or both arms)
     * and flagged if either exceeds the cutoff. A side missing shoulder or
     * elbow visibility is skipped rather than treated as a pass.
     */
    fun looseUpperArm(xy: FloatArray, conf: FloatArray): Boolean {
        fun sideAngle(shoulder: Int, elbow: Int): Double? {
            if (conf[shoulder] < FormClassifier.MIN_LANDMARK_CONFIDENCE ||
                conf[elbow] < FormClassifier.MIN_LANDMARK_CONFIDENCE
            ) return null
            val sx = xy[shoulder * 2].toDouble()
            val sy = xy[shoulder * 2 + 1].toDouble()
            val ex = xy[elbow * 2].toDouble()
            val ey = xy[elbow * 2 + 1].toDouble()
            // "Ground projection" of the shoulder: straight down in image
            // space (y grows downward, the same convention bbox_normalize
            // and the depth_gate use).
            return angleAtVertex(ex, ey, sx, sy, sx, sy + 1.0)
        }
        val left = sideAngle(LEFT_SHOULDER, LEFT_ELBOW)
        val right = sideAngle(RIGHT_SHOULDER, RIGHT_ELBOW)
        return listOfNotNull(left, right).any { it > LOOSE_UPPER_ARM_DEG }
    }

    /**
     * The hip-knee-ankle angle of whichever visible knee is more bent (the
     * working leg in a lunge), in degrees. Null if neither leg has hip, knee
     * and ankle all visible. Shared by [kneeAngleOutOfRange] and
     * [RepCounter]'s lunge rep counting -- both need the same number, one as
     * a fault check, the other as a cycle to count.
     */
    fun kneeAngle(xy: FloatArray, conf: FloatArray): Double? {
        fun sideAngle(hip: Int, knee: Int, ankle: Int): Double? {
            if (conf[hip] < FormClassifier.MIN_LANDMARK_CONFIDENCE ||
                conf[knee] < FormClassifier.MIN_LANDMARK_CONFIDENCE ||
                conf[ankle] < FormClassifier.MIN_LANDMARK_CONFIDENCE
            ) return null
            return angleAtVertex(
                xy[hip * 2].toDouble(), xy[hip * 2 + 1].toDouble(),
                xy[knee * 2].toDouble(), xy[knee * 2 + 1].toDouble(),
                xy[ankle * 2].toDouble(), xy[ankle * 2 + 1].toDouble(),
            )
        }
        val left = sideAngle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        val right = sideAngle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        return listOfNotNull(left, right).minOrNull()
    }

    /**
     * Lunge's knee angle: upstream's stated correct range at the bottom of a
     * lunge is 60-135 degrees at the hip-knee-ankle vertex.
     *
     * This is not gated to the bottom of a rep by itself -- a standing knee
     * reads near 180 degrees, well outside the range, and would misfire as a
     * fault. Callers must only invoke this once `FormClassifier`'s own
     * `depth_gate` has accepted the pose (see `MainActivity.classifyForm`),
     * the same evidence bar `knee_over_toe` is held to.
     */
    fun kneeAngleOutOfRange(xy: FloatArray, conf: FloatArray): Boolean {
        val working = kneeAngle(xy, conf) ?: return false
        return working < KNEE_ANGLE_MIN_DEG || working > KNEE_ANGLE_MAX_DEG
    }

    /**
     * The elbow-flex angle (shoulder-elbow-wrist vertex at the elbow) of
     * whichever visible arm is more contracted, in degrees. Null if neither
     * arm has shoulder, elbow and wrist all visible.
     *
     * This is a different angle from [looseUpperArm]'s -- that one measures
     * how far the upper arm has swung from vertical (a form fault); this one
     * measures how bent the elbow itself is (a curl's cycle), which is what
     * [RepCounter] needs to count reps regardless of form quality.
     */
    fun elbowFlexAngle(xy: FloatArray, conf: FloatArray): Double? {
        fun sideAngle(shoulder: Int, elbow: Int, wrist: Int): Double? {
            if (conf[shoulder] < FormClassifier.MIN_LANDMARK_CONFIDENCE ||
                conf[elbow] < FormClassifier.MIN_LANDMARK_CONFIDENCE ||
                conf[wrist] < FormClassifier.MIN_LANDMARK_CONFIDENCE
            ) return null
            return angleAtVertex(
                xy[shoulder * 2].toDouble(), xy[shoulder * 2 + 1].toDouble(),
                xy[elbow * 2].toDouble(), xy[elbow * 2 + 1].toDouble(),
                xy[wrist * 2].toDouble(), xy[wrist * 2 + 1].toDouble(),
            )
        }
        val left = sideAngle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        val right = sideAngle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        return listOfNotNull(left, right).minOrNull()
    }

    //: Upstream's stated cutoffs verbatim -- see the class doc for provenance.
    private const val LOOSE_UPPER_ARM_DEG = 40.0
    private const val KNEE_ANGLE_MIN_DEG = 60.0
    private const val KNEE_ANGLE_MAX_DEG = 135.0

    // COCO-17 indices, matching FormClassifier.COCO_NAMES's order.
    private const val LEFT_SHOULDER = 5
    private const val RIGHT_SHOULDER = 6
    private const val LEFT_ELBOW = 7
    private const val RIGHT_ELBOW = 8
    private const val LEFT_WRIST = 9
    private const val RIGHT_WRIST = 10
    private const val LEFT_HIP = 11
    private const val RIGHT_HIP = 12
    private const val LEFT_KNEE = 13
    private const val RIGHT_KNEE = 14
    private const val LEFT_ANKLE = 15
    private const val RIGHT_ANKLE = 16
}
