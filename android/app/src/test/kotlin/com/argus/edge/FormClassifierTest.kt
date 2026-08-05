package com.argus.edge

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

/**
 * Pins every shipped `FormClassifier` artifact to the model actually fitted
 * in Python, iterating over whatever `*_lr.json` files this build ships
 * (docs/ADDING_AN_EXERCISE.md §3) rather than naming one exercise.
 *
 * `FormClassifier` reimplements a scikit-learn multinomial logistic
 * regression as plain arithmetic, which carries the same class of risk the
 * decode path does: the reimplementation can be subtly wrong — feature order
 * transposed, standardization skipped, one-vs-rest sigmoids instead of a
 * softmax — and still produce plausible-looking probabilities. Only comparison
 * against the fitted model catches that.
 *
 * `tests/data/<exercise>_vectors.json` is emitted by
 * `scripts/train_form_model.py` alongside each artifact, from real held-out
 * rows scored by the fitted model. Both files of a pair are written by the
 * same run, so they cannot drift apart — and the artifact read here is the
 * one the app ships, not a copy.
 */
class FormClassifierTest {

    private fun property(name: String): File =
        File(System.getProperty(name) ?: error("$name system property not set"))

    private class Fixture(
        val exercise: String,
        val artifact: JSONObject,
        val fixture: JSONObject,
        val classifier: FormClassifier,
        val landmarks: List<String>,
        val featureKinds: List<String>,
    )

    /** Every exercise this build ships an artifact + fixture pair for. */
    private val fixtures: List<Fixture> by lazy {
        val assetsDir = property("argus.assets")
        val fixturesDir = property("argus.fixtures")
        val artifacts = (assetsDir.listFiles { f -> f.name.endsWith("_lr.json") } ?: emptyArray())
            .sortedBy { it.name }
        assertTrue("no *_lr.json artifacts found in $assetsDir", artifacts.isNotEmpty())
        artifacts.map { artifactFile ->
            val exercise = artifactFile.name.removeSuffix("_lr.json")
            val artifactJson = artifactFile.readText()
            val artifact = JSONObject(artifactJson)
            val fixtureFile = File(fixturesDir, "${exercise}_vectors.json")
            assertTrue("no fixture for '$exercise' at $fixtureFile", fixtureFile.isFile)
            Fixture(
                exercise = exercise,
                artifact = artifact,
                fixture = JSONObject(fixtureFile.readText()),
                classifier = FormClassifier.fromJson(artifactJson),
                landmarks = artifact.getJSONArray("landmarks").let { a -> List(a.length()) { a.getString(it) } },
                featureKinds = artifact.getJSONArray("feature_kinds").let { a -> List(a.length()) { a.getString(it) } },
            )
        }
    }

    /**
     * Scatter a fixture's flat (x, y) feature vector back into COCO-17
     * arrays and classify.
     *
     * Going through the public entry point rather than calling the arithmetic
     * directly is deliberate: it puts the COCO index mapping under test too,
     * which is where a silent transposition would otherwise hide.
     */
    private fun classify(fx: Fixture, features: DoubleArray): FormClassifier.Verdict {
        val stride = fx.featureKinds.size
        val xy = FloatArray(NUM_KEYPOINTS * 2)
        // Fixture vectors carry coordinates only; confidence is a gate, not a
        // feature, so it is supplied here as "fully visible" to exercise the
        // arithmetic rather than the gate. The gate has its own tests below.
        val conf = FloatArray(NUM_KEYPOINTS) { 1.0f }
        fx.landmarks.forEachIndexed { i, name ->
            val coco = FormClassifier.COCO_NAMES.indexOf(name)
            xy[coco * 2] = features[i * stride + fx.featureKinds.indexOf("x")].toFloat()
            xy[coco * 2 + 1] = features[i * stride + fx.featureKinds.indexOf("y")].toFloat()
        }
        return fx.classifier.classify(xy, conf)
    }

    private fun eachCase(body: (Fixture, JSONObject, FormClassifier.Verdict) -> Unit) {
        for (fx in fixtures) {
            val cases = fx.fixture.getJSONArray("cases")
            assertTrue("${fx.exercise} fixture has no cases", cases.length() > 0)
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                // `raw_features`, not `features`: classify() applies the
                // artifact's own encoding internally, so it must be given
                // pre-encoding coordinates -- the same ones a pose estimator
                // would produce -- not the already-encoded model input. See
                // the docstring on write_fixture in train_form_model.py.
                val json = case.getJSONArray("raw_features")
                body(fx, case, classify(fx, DoubleArray(json.length()) { json.getDouble(it) }))
            }
        }
    }

    /** A pose visible enough to be judged, for every exercise's landmark set.
     *
     * y increases with index, which -- for every shipped landmark ordering,
     * all roughly head-to-toe -- reads as a plausible depth profile rather
     * than a standing pose flattened to one height. That matters for lunge's
     * depth gate: a constant y everywhere is indistinguishable from standing
     * and is refused by design (see the dedicated depth-gate test below).
     */
    private fun visiblePose(fx: Fixture): Pair<FloatArray, FloatArray> {
        val minVisible = fx.artifact.getInt("min_visible_landmarks")
        val xy = FloatArray(NUM_KEYPOINTS * 2) { 0.5f }
        fx.landmarks.forEachIndexed { i, name ->
            val coco = FormClassifier.COCO_NAMES.indexOf(name)
            xy[coco * 2 + 1] = 0.1f + 0.05f * i
        }
        val conf = FloatArray(NUM_KEYPOINTS)
        fx.landmarks.take(minVisible).forEach { conf[FormClassifier.COCO_NAMES.indexOf(it)] = 0.99f }
        return xy to conf
    }

    @Test
    fun `probabilities reproduce the fitted model`() {
        eachCase { fx, case, verdict ->
            val tolerance = fx.fixture.getDouble("tolerance")
            val expected = case.getJSONArray("probabilities")
            assertEquals("${fx.exercise} class count", expected.length(), verdict.probabilities.size)
            for (c in 0 until expected.length()) {
                assertEquals(
                    "${fx.exercise} ${case.getString("true_label")} class $c",
                    expected.getDouble(c), verdict.probabilities[c], tolerance,
                )
            }
        }
    }

    @Test
    fun `form reason codes match the fitted model's decision`() {
        eachCase { fx, case, verdict ->
            val expected = case.getJSONArray("form_reason_codes").let { a ->
                List(a.length()) { a.getString(it) }
            }
            assertEquals("${fx.exercise} codes", expected, verdict.formReasonCodes)
            assertEquals("${fx.exercise} class", case.getString("predicted_class"), verdict.label)
        }
    }

    @Test
    fun `a correct classification reports no codes at all`() {
        val sawCorrect = mutableSetOf<String>()
        eachCase { fx, case, verdict ->
            if (case.getString("predicted_class") != "C") return@eachCase
            sawCorrect += fx.exercise
            assertTrue(
                "${fx.exercise}: a correct classification must report an empty " +
                    "form_reason_codes, not a 'correct' code",
                verdict.formReasonCodes.isEmpty(),
            )
        }
        for (fx in fixtures) {
            assertTrue("${fx.exercise} fixture covers no correct cases", fx.exercise in sawCorrect)
        }
    }

    @Test
    fun `an invisible body is refused rather than guessed at`() {
        // A softmax always returns one of its classes, however unlike its
        // training data the input is -- on a real device a photograph of one
        // leg scored `hips_piked` at 100% for plank. The probability
        // threshold cannot catch that: it separates ambiguity between the
        // classes, not input that is not the exercise. The gate is on
        // evidence instead.
        for (fx in fixtures) {
            val verdict = fx.classifier.classify(
                FloatArray(NUM_KEYPOINTS * 2) { 0.5f },
                FloatArray(NUM_KEYPOINTS) { 0.0f },
            )
            assertEquals(fx.exercise, FormClassifier.UNKNOWN, verdict.label)
            assertFalse("${fx.exercise}: an unseen body cannot be a confident verdict", verdict.confident)
            assertTrue("${fx.exercise}: an unseen body leaked a code", verdict.formReasonCodes.isEmpty())
        }
    }

    @Test
    fun `a partially visible body is refused`() {
        // Exactly one landmark short of the gate: the pose is well-formed and
        // the model would happily score it.
        for (fx in fixtures) {
            val minVisible = fx.artifact.getInt("min_visible_landmarks")
            val conf = FloatArray(NUM_KEYPOINTS)
            fx.landmarks.take(minVisible - 1).forEach {
                conf[FormClassifier.COCO_NAMES.indexOf(it)] = 0.99f
            }
            val verdict = fx.classifier.classify(FloatArray(NUM_KEYPOINTS * 2) { 0.5f }, conf)
            assertEquals(fx.exercise, FormClassifier.UNKNOWN, verdict.label)
            assertTrue(fx.exercise, verdict.formReasonCodes.isEmpty())
        }
    }

    @Test
    fun `a sufficiently visible body is judged`() {
        for (fx in fixtures) {
            val (xy, conf) = visiblePose(fx)
            val verdict = fx.classifier.classify(xy, conf)
            assertFalse("${fx.exercise}: the gate must not reject a visible body", verdict.label == FormClassifier.UNKNOWN)
            assertTrue("${fx.exercise}: probabilities must be finite", verdict.probabilities.all { it.isFinite() })
            assertEquals("${fx.exercise}: probabilities must sum to 1", 1.0, verdict.probabilities.sum(), 1e-9)
        }
    }

    @Test
    fun `a pose outside the depth gate is refused`() {
        // Lunge's knee-over-toe label was only ever collected -- and, per
        // upstream's own detection code, only ever evaluated -- at the bottom
        // of a lunge (docs/VALIDATION.md). A standing pose (every landmark at
        // the same y) is exactly what that gate exists to reject: it is
        // unlike anything the label was ever fit against.
        val gated = fixtures.filter { !it.artifact.isNull("depth_gate") }
        assertTrue("expected at least one exercise with a depth_gate (lunge)", gated.isNotEmpty())
        for (fx in gated) {
            val minVisible = fx.artifact.getInt("min_visible_landmarks")
            val conf = FloatArray(NUM_KEYPOINTS)
            fx.landmarks.take(minVisible).forEach { conf[FormClassifier.COCO_NAMES.indexOf(it)] = 0.99f }
            val verdict = fx.classifier.classify(FloatArray(NUM_KEYPOINTS * 2) { 0.5f }, conf)
            assertEquals(
                "${fx.exercise}: a standing pose must be refused, not classified",
                FormClassifier.UNKNOWN, verdict.label,
            )
        }
    }

    @Test
    fun `confidence never reaches the feature vector`() {
        // The withdrawn plank revision fed MediaPipe visibility in as a
        // feature. It saturates near 1.0 with a standard deviation of 0.0013,
        // so a normal YOLO26 confidence standardized to -116 sigma and the
        // softmax stopped depending on the pose at all. Same pose, different
        // confidences, must now be the same verdict, for every exercise.
        for (fx in fixtures) {
            val (xy, _) = visiblePose(fx)
            val high = fx.classifier.classify(xy, FloatArray(NUM_KEYPOINTS) { 0.99f })
            val low = fx.classifier.classify(xy, FloatArray(NUM_KEYPOINTS) { 0.35f })
            assertEquals("${fx.exercise}: confidence changed the verdict", high.label, low.label)
            for (c in high.probabilities.indices) {
                assertEquals(
                    "${fx.exercise}: confidence changed class $c's probability",
                    high.probabilities[c], low.probabilities[c], 1e-12,
                )
            }
        }
    }

    @Test
    fun `the landmark box normalization is translation and scale invariant`() {
        val xs = doubleArrayOf(0.1, 0.2, 0.3, 0.4)
        val ys = doubleArrayOf(0.5, 0.5, 0.7, 0.9)
        val movedX = DoubleArray(4) { xs[it] * 2 + 0.3 }
        val movedY = DoubleArray(4) { ys[it] * 2 + 0.3 }

        FormClassifier.normalizeToLandmarkBox(xs, ys)
        FormClassifier.normalizeToLandmarkBox(movedX, movedY)

        for (i in xs.indices) {
            assertEquals("x[$i]", xs[i], movedX[i], 1e-12)
            assertEquals("y[$i]", ys[i], movedY[i], 1e-12)
        }
    }

    @Test
    fun `a collapsed landmark span does not divide by zero`() {
        val xs = doubleArrayOf(0.4, 0.4, 0.4)
        val ys = doubleArrayOf(0.4, 0.4, 0.4)
        FormClassifier.normalizeToLandmarkBox(xs, ys)
        assertTrue("collapsed span produced non-finite x", xs.all { it.isFinite() })
        assertTrue("collapsed span produced non-finite y", ys.all { it.isFinite() })
    }

    @Test
    fun `each artifact ships the exercise, encoding and threshold its classifier implements`() {
        for (fx in fixtures) {
            assertEquals(fx.exercise, fx.artifact.getString("exercise"))
            assertEquals(fx.exercise, fx.classifier.exercise)
            assertEquals(fx.exercise, fx.fixture.getString("encoding"), fx.classifier.encoding)
            assertEquals(
                fx.exercise,
                fx.fixture.getDouble("probability_threshold"), fx.classifier.probabilityThreshold, 1e-12,
            )
        }
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
        for (fx in fixtures) {
            val emitted = fx.artifact.getJSONObject("class_to_code")
            for (key in emitted.keys()) {
                if (emitted.isNull(key)) continue
                val code = emitted.getString(key)
                assertTrue(
                    "${fx.exercise}: class '$key' maps to '$code', which is not in [scoring.form_error_vocab]",
                    code in vocabulary,
                )
            }
        }
    }

    @Test
    fun `a malformed artifact is refused rather than half-loaded`() {
        try {
            FormClassifier.fromJson("""{"format":"random_forest"}""")
            fail("an unsupported artifact format was accepted")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("unsupported form artifact format"))
        }
    }

    @Test
    fun `a zero scaler scale is refused`() {
        val artifact = JSONObject(fixtures.first().artifact.toString())
        artifact.getJSONArray("scaler_scale").put(0, 0.0)
        try {
            FormClassifier.fromJson(artifact.toString())
            fail("a zero scale was accepted; standardization would divide by zero")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("zero scale"))
        }
    }
}
