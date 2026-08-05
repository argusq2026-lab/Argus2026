package com.argus.edge

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

/**
 * Pins the Kotlin plank classifier to the model actually fitted in Python.
 *
 * `PlankClassifier` reimplements a scikit-learn multinomial logistic
 * regression as plain arithmetic, which carries the same class of risk the
 * decode path does: the reimplementation can be subtly wrong — feature order
 * transposed, standardization skipped, one-vs-rest sigmoids instead of a
 * softmax — and still produce plausible-looking probabilities. Only comparison
 * against the fitted model catches that.
 *
 * `tests/data/plank_vectors.json` is emitted by
 * `scripts/train_plank_model.py` alongside the artifact, from real held-out
 * rows scored by the fitted model. Both files are written by the same run, so
 * they cannot drift apart — and the artifact read here is the one the app
 * ships, not a copy.
 */
class PlankClassifierTest {

    private fun property(name: String): File =
        File(System.getProperty(name) ?: error("$name system property not set"))

    private val artifactJson: String by lazy {
        File(property("argus.assets"), "plank_lr.json").readText()
    }

    private val fixture: JSONObject by lazy {
        JSONObject(File(property("argus.fixtures"), "plank_vectors.json").readText())
    }

    private val classifier: PlankClassifier by lazy { PlankClassifier.fromJson(artifactJson) }

    /** The landmark names the artifact was fitted on, in feature order. */
    private val landmarks: List<String> by lazy {
        JSONObject(artifactJson).getJSONArray("landmarks").let { a ->
            List(a.length()) { a.getString(it) }
        }
    }

    private val featureKinds: List<String> by lazy {
        JSONObject(artifactJson).getJSONArray("feature_kinds").let { a ->
            List(a.length()) { a.getString(it) }
        }
    }

    /**
     * Scatter a fixture's flat (x, y, v) feature vector back into COCO-17
     * arrays and classify.
     *
     * Going through the public entry point rather than calling the arithmetic
     * directly is deliberate: it puts the COCO index mapping under test too,
     * which is where a silent transposition would otherwise hide.
     */
    private fun classify(features: DoubleArray): PlankClassifier.Verdict {
        val stride = featureKinds.size
        val xy = FloatArray(NUM_KEYPOINTS * 2)
        // Fixture vectors carry coordinates only; confidence is a gate, not a
        // feature, so it is supplied here as "fully visible" to exercise the
        // arithmetic rather than the gate. The gate has its own tests below.
        val conf = FloatArray(NUM_KEYPOINTS) { 1.0f }
        landmarks.forEachIndexed { i, name ->
            val coco = PlankClassifier.COCO_NAMES.indexOf(name)
            xy[coco * 2] = features[i * stride + featureKinds.indexOf("x")].toFloat()
            xy[coco * 2 + 1] = features[i * stride + featureKinds.indexOf("y")].toFloat()
        }
        return classifier.classify(xy, conf)
    }

    private fun eachCase(body: (JSONObject, PlankClassifier.Verdict) -> Unit) {
        val cases = fixture.getJSONArray("cases")
        assertTrue("fixture has no cases", cases.length() > 0)
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val json = case.getJSONArray("features")
            body(case, classify(DoubleArray(json.length()) { json.getDouble(it) }))
        }
    }

    @Test
    fun `probabilities reproduce the fitted model`() {
        val tolerance = fixture.getDouble("tolerance")
        eachCase { case, verdict ->
            val expected = case.getJSONArray("probabilities")
            assertEquals("class count", expected.length(), verdict.probabilities.size)
            for (c in 0 until expected.length()) {
                assertEquals(
                    "${case.getString("true_label")} class $c",
                    expected.getDouble(c), verdict.probabilities[c], tolerance,
                )
            }
        }
    }

    @Test
    fun `form reason codes match the fitted model's decision`() {
        eachCase { case, verdict ->
            val expected = case.getJSONArray("form_reason_codes").let { a ->
                List(a.length()) { a.getString(it) }
            }
            assertEquals("codes", expected, verdict.formReasonCodes)
            assertEquals("class", case.getString("predicted_class"), verdict.label)
        }
    }

    @Test
    fun `a correct plank reports no codes at all`() {
        var sawCorrect = false
        eachCase { case, verdict ->
            if (case.getString("predicted_class") != "C") return@eachCase
            sawCorrect = true
            assertTrue(
                "a correct plank must report an empty form_reason_codes, not a 'correct' code",
                verdict.formReasonCodes.isEmpty(),
            )
        }
        assertTrue("fixture covers no correct planks", sawCorrect)
    }

    @Test
    fun `an invisible body is refused rather than guessed at`() {
        // A three-class softmax always returns one of its three classes, however
        // unlike its training data the input is -- on a real device a photograph
        // of one leg scored `hips_piked` at 100%. The probability threshold
        // cannot catch that: it separates ambiguity between the three classes,
        // not input that is not a plank. The gate is on evidence instead.
        val verdict = classifier.classify(
            FloatArray(NUM_KEYPOINTS * 2) { 0.5f },
            FloatArray(NUM_KEYPOINTS) { 0.0f },
        )
        assertEquals(PlankClassifier.UNKNOWN, verdict.label)
        assertFalse("an unseen body cannot be a confident verdict", verdict.confident)
        assertTrue("an unseen body leaked a code", verdict.formReasonCodes.isEmpty())
    }

    @Test
    fun `a partially visible body is refused`() {
        // Exactly one landmark short of the gate: the pose is well-formed and
        // the model would happily score it.
        val conf = FloatArray(NUM_KEYPOINTS)
        landmarks.take(PlankClassifier.MIN_VISIBLE_LANDMARKS - 1).forEach {
            conf[PlankClassifier.COCO_NAMES.indexOf(it)] = 0.99f
        }
        val verdict = classifier.classify(FloatArray(NUM_KEYPOINTS * 2) { 0.5f }, conf)
        assertEquals(PlankClassifier.UNKNOWN, verdict.label)
        assertTrue(verdict.formReasonCodes.isEmpty())
    }

    @Test
    fun `a sufficiently visible body is judged`() {
        val conf = FloatArray(NUM_KEYPOINTS)
        landmarks.take(PlankClassifier.MIN_VISIBLE_LANDMARKS).forEach {
            conf[PlankClassifier.COCO_NAMES.indexOf(it)] = 0.99f
        }
        val verdict = classifier.classify(FloatArray(NUM_KEYPOINTS * 2) { 0.5f }, conf)
        assertFalse("the gate must not reject a visible body", verdict.label == PlankClassifier.UNKNOWN)
        assertTrue("probabilities must be finite", verdict.probabilities.all { it.isFinite() })
        assertEquals("probabilities must sum to 1", 1.0, verdict.probabilities.sum(), 1e-9)
    }

    @Test
    fun `confidence never reaches the feature vector`() {
        // The withdrawn revision fed MediaPipe visibility in as a feature. It
        // saturates near 1.0 with a standard deviation of 0.0013, so a normal
        // YOLO26 confidence standardized to -116 sigma and the softmax stopped
        // depending on the pose at all. Same pose, different confidences, must
        // now be the same verdict.
        val xy = FloatArray(NUM_KEYPOINTS * 2)
        landmarks.forEachIndexed { i, name ->
            val coco = PlankClassifier.COCO_NAMES.indexOf(name)
            xy[coco * 2] = 0.1f + 0.06f * i
            xy[coco * 2 + 1] = 0.4f + 0.01f * i
        }
        val high = classifier.classify(xy, FloatArray(NUM_KEYPOINTS) { 0.99f })
        val low = classifier.classify(xy, FloatArray(NUM_KEYPOINTS) { 0.35f })
        assertEquals("confidence changed the verdict", high.label, low.label)
        for (c in high.probabilities.indices) {
            assertEquals(
                "confidence changed class $c's probability",
                high.probabilities[c], low.probabilities[c], 1e-12,
            )
        }
    }

    @Test
    fun `the landmark box normalization is translation and scale invariant`() {
        val xs = doubleArrayOf(0.1, 0.2, 0.3, 0.4)
        val ys = doubleArrayOf(0.5, 0.5, 0.7, 0.9)
        val movedX = DoubleArray(4) { xs[it] * 2 + 0.3 }
        val movedY = DoubleArray(4) { ys[it] * 2 + 0.3 }

        PlankClassifier.normalizeToLandmarkBox(xs, ys)
        PlankClassifier.normalizeToLandmarkBox(movedX, movedY)

        for (i in xs.indices) {
            assertEquals("x[$i]", xs[i], movedX[i], 1e-12)
            assertEquals("y[$i]", ys[i], movedY[i], 1e-12)
        }
    }

    @Test
    fun `a collapsed landmark span does not divide by zero`() {
        val xs = doubleArrayOf(0.4, 0.4, 0.4)
        val ys = doubleArrayOf(0.4, 0.4, 0.4)
        PlankClassifier.normalizeToLandmarkBox(xs, ys)
        assertTrue("collapsed span produced non-finite x", xs.all { it.isFinite() })
        assertTrue("collapsed span produced non-finite y", ys.all { it.isFinite() })
    }

    @Test
    fun `the artifact ships the encoding and threshold the classifier implements`() {
        assertEquals(fixture.getString("encoding"), classifier.encoding)
        assertEquals(
            fixture.getDouble("probability_threshold"),
            classifier.probabilityThreshold, 1e-12,
        )
    }

    @Test
    fun `every emitted code is one the server's vocabulary contains`() {
        // The server rejects the connection on an unrecognised code, so a
        // classifier that could emit one is a broken station, not a bad guess.
        val vocabulary = JSONObject(
            File(property("argus.fixtures"), "protocol_vectors.json").readText()
        ).getJSONArray("form_error_vocab_keys").let { a ->
            List(a.length()) { a.getString(it) }.toSet()
        }
        val emitted = JSONObject(artifactJson).getJSONObject("class_to_code")
        for (key in emitted.keys()) {
            if (emitted.isNull(key)) continue
            val code = emitted.getString(key)
            assertTrue(
                "class '$key' maps to '$code', which is not in [scoring.form_error_vocab]",
                code in vocabulary,
            )
        }
    }

    @Test
    fun `a malformed artifact is refused rather than half-loaded`() {
        try {
            PlankClassifier.fromJson("""{"format":"random_forest"}""")
            fail("an unsupported artifact format was accepted")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("unsupported plank artifact format"))
        }
    }

    @Test
    fun `a zero scaler scale is refused`() {
        val artifact = JSONObject(artifactJson)
        artifact.getJSONArray("scaler_scale").put(0, 0.0)
        try {
            PlankClassifier.fromJson(artifact.toString())
            fail("a zero scale was accepted; standardization would divide by zero")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("zero scale"))
        }
    }
}
