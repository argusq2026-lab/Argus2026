package com.argus.edge

import org.json.JSONObject
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

/**
 * `GeometricFormChecks` is plain trigonometry, not a reimplementation of a
 * fitted model -- there is no Python side to pin it against the way
 * `FormClassifierTest` pins the logistic regressions. What is worth testing
 * instead: the angle arithmetic itself, that missing landmarks are skipped
 * rather than treated as a pass, and that the two codes it can emit are ones
 * the server's vocabulary actually recognises.
 */
class GeometricFormChecksTest {

    private fun property(name: String): File =
        File(System.getProperty(name) ?: error("$name system property not set"))

    private fun coco(name: String) = FormClassifier.COCO_NAMES.indexOf(name)

    private fun emptyPose(): Pair<FloatArray, FloatArray> =
        FloatArray(NUM_KEYPOINTS * 2) to FloatArray(NUM_KEYPOINTS)

    private fun setPoint(xy: FloatArray, landmark: String, x: Double, y: Double) {
        val i = coco(landmark)
        xy[i * 2] = x.toFloat()
        xy[i * 2 + 1] = y.toFloat()
    }

    private fun setVisible(conf: FloatArray, landmark: String) {
        conf[coco(landmark)] = 0.99f
    }

    /** Places an elbow at exactly `degrees` from vertical, shoulder at the origin. */
    private fun armAtAngle(shoulder: String, elbow: String, degrees: Double): Pair<FloatArray, FloatArray> {
        val (xy, conf) = emptyPose()
        val rad = Math.toRadians(degrees)
        setPoint(xy, shoulder, 0.0, 0.0)
        setPoint(xy, elbow, sin(rad), cos(rad))
        setVisible(conf, shoulder)
        setVisible(conf, elbow)
        return xy to conf
    }

    /** Places a knee vertex with hip straight up and ankle at exactly `degrees` from it. */
    private fun kneeAtAngle(hip: String, knee: String, ankle: String, degrees: Double): Pair<FloatArray, FloatArray> {
        val (xy, conf) = emptyPose()
        val rad = Math.toRadians(degrees)
        setPoint(xy, knee, 0.0, 0.0)
        setPoint(xy, hip, 0.0, -1.0)
        setPoint(xy, ankle, sin(rad), -cos(rad))
        setVisible(conf, hip)
        setVisible(conf, knee)
        setVisible(conf, ankle)
        return xy to conf
    }

    // -- loose upper arm ------------------------------------------------------

    @Test
    fun `an arm hanging straight down is not flagged`() {
        val (xy, conf) = armAtAngle("left_shoulder", "left_elbow", 0.0)
        assertFalse(GeometricFormChecks.looseUpperArm(xy, conf))
    }

    @Test
    fun `an arm just under the 40 degree cutoff is not flagged`() {
        val (xy, conf) = armAtAngle("left_shoulder", "left_elbow", 39.0)
        assertFalse(GeometricFormChecks.looseUpperArm(xy, conf))
    }

    @Test
    fun `an arm just over the 40 degree cutoff is flagged`() {
        val (xy, conf) = armAtAngle("left_shoulder", "left_elbow", 41.0)
        assertTrue(GeometricFormChecks.looseUpperArm(xy, conf))
    }

    @Test
    fun `either arm swinging out flags the rep`() {
        val (xy, conf) = armAtAngle("right_shoulder", "right_elbow", 90.0)
        assertTrue(GeometricFormChecks.looseUpperArm(xy, conf))
    }

    @Test
    fun `an invisible arm is skipped, not treated as a pass`() {
        val (xy, conf) = emptyPose()
        assertFalse(GeometricFormChecks.looseUpperArm(xy, conf))
    }

    // -- lunge knee angle -------------------------------------------------------

    @Test
    fun `a knee bent to 90 degrees is within range`() {
        val (xy, conf) = kneeAtAngle("left_hip", "left_knee", "left_ankle", 90.0)
        assertFalse(GeometricFormChecks.kneeAngleOutOfRange(xy, conf))
    }

    @Test
    fun `a standing knee near 180 degrees is out of range`() {
        val (xy, conf) = kneeAtAngle("left_hip", "left_knee", "left_ankle", 179.0)
        assertTrue(GeometricFormChecks.kneeAngleOutOfRange(xy, conf))
    }

    @Test
    fun `a knee bent past the 60 degree floor is out of range`() {
        val (xy, conf) = kneeAtAngle("left_hip", "left_knee", "left_ankle", 45.0)
        assertTrue(GeometricFormChecks.kneeAngleOutOfRange(xy, conf))
    }

    @Test
    fun `angles just inside the 60 to 135 range are not flagged`() {
        // Not the exact boundary: sin/cos/acos does not round-trip 60.0 and
        // 135.0 exactly, so asserting the literal edge is flaky on floating
        // point rather than on the logic under test.
        val low = kneeAtAngle("left_hip", "left_knee", "left_ankle", 60.5)
        val high = kneeAtAngle("left_hip", "left_knee", "left_ankle", 134.5)
        assertFalse(GeometricFormChecks.kneeAngleOutOfRange(low.first, low.second))
        assertFalse(GeometricFormChecks.kneeAngleOutOfRange(high.first, high.second))
    }

    @Test
    fun `the more bent visible knee is the one judged`() {
        // Front leg (left, 90 degrees, correct) and back leg (right, 179
        // degrees, nearly straight -- ordinary in a lunge) both visible: the
        // smaller angle is the working knee, so this must not flag.
        val (xy, conf) = kneeAtAngle("left_hip", "left_knee", "left_ankle", 90.0)
        val (rightXy, rightConf) = kneeAtAngle("right_hip", "right_knee", "right_ankle", 179.0)
        for (i in xy.indices) if (rightXy[i] != 0f) xy[i] = rightXy[i]
        for (i in conf.indices) if (rightConf[i] != 0f) conf[i] = rightConf[i]
        assertFalse(GeometricFormChecks.kneeAngleOutOfRange(xy, conf))
    }

    @Test
    fun `neither knee visible is skipped, not treated as a pass`() {
        val (xy, conf) = emptyPose()
        assertFalse(GeometricFormChecks.kneeAngleOutOfRange(xy, conf))
    }

    // -- the codes this class can emit are ones the server recognises -----------

    @Test
    fun `both emitted codes are in the server's vocabulary`() {
        val vocabulary = JSONObject(
            File(property("argus.fixtures"), "protocol_vectors.json").readText()
        ).getJSONArray("form_error_vocab_keys").let { a ->
            List(a.length()) { a.getString(it) }.toSet()
        }
        assertTrue("loose_upper_arm not in [scoring.form_error_vocab]", "loose_upper_arm" in vocabulary)
        assertTrue("knee_angle_out_of_range not in [scoring.form_error_vocab]", "knee_angle_out_of_range" in vocabulary)
    }
}
