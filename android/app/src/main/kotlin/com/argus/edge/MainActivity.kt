package com.argus.edge

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.ArrayAdapter
import android.widget.AdapterView
import android.widget.SeekBar
import android.widget.TextView
import android.graphics.drawable.GradientDrawable
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import android.content.res.Configuration
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.concurrent.Executors
import android.view.View
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** The exercise a station assumes when none has been configured. */
const val DEFAULT_EXERCISE = "plank"

/**
 * How often an empty station says so.
 *
 * Once a second, not once a frame: "nobody is here" is a liveness fact, not a
 * measurement, and it only has to arrive comfortably inside the server's
 * `ingest.track_ttl_s` (10 s by default) to stop the station being evicted.
 */
const val IDLE_INTERVAL_MS = 1_000L

/**
 * The station screen — what the person setting up a phone on a tripod sees.
 *
 * One phone watches one trainee (`docs/PROTOCOL.md`): this screen exists to
 * make three things legible without an engineer present — is the model loaded
 * and on the NPU, is the camera seeing a person, and is the laptop receiving.
 * Detection boxes render live; the status strip never says "ready" unless the
 * NPU session actually opened; and a rejected connection shows the server's
 * own error text rather than a generic failure.
 *
 * The frame is a local: captured, letterboxed, inferred on, drawn over, and
 * closed. It is never stored on a field, written to disk, or serialised —
 * only the normalized bounding box of the best detection ever reaches the
 * network, as a PROTOCOL.md observation.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DetectionOverlayView
    private lateinit var debugPanel: TextView
    private lateinit var stateText: TextView
    private lateinit var stateDot: View
    private lateinit var toggleButton: Button
    private lateinit var thresholdLabel: TextView
    private lateinit var thresholdRow: LinearLayout
    private var debugVisible = false

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var imageAnalysis: ImageAnalysis? = null
    private val running = AtomicBoolean(false)

    /**
     * Bumped whenever the station starts or stops.
     *
     * A frame takes ~22 ms to go through the NPU, and `running` is read at the
     * top of that. Press Stop mid-flight and the frame finishes, repaints the
     * overlay it computed, and — since no further frames are processed — that
     * repaint is the last thing drawn and stays on screen. Capturing the epoch
     * per frame and discarding results from a superseded one closes the window
     * that a second `running` check would still leave open.
     */
    private val sessionEpoch = AtomicInteger(0)

    private lateinit var deviceId: String
    private lateinit var modelStore: ModelStore
    private var detector: QnnDetector? = null
    private var detectorStatus: String = "no model"
    private var poseEstimator: PoseEstimator? = null
    /** Single-stage prototype; when present it replaces both models above. */
    private var yolo26: Yolo26PoseEstimator? = null
    private var backendLabel: String = "yolox+blazepose"
    private var poseStatus: String = "not staged"
    private var poseMsEma = 0.0
    private var lastVisibleKeypoints = 0
    private var lastPoseScore = 0f
    private val subjectTracker = SubjectTracker()

    /** Last `idle` heartbeat, so an empty station is announced without
     *  spending a frame's worth of socket traffic on saying nothing. */
    @Volatile private var lastIdleSentMs = 0L

    /** The last frame that actually produced something, for the brief hold. */
    private class Snapshot(
        val detections: List<Detection>,
        val subject: Detection?,
        val subjectPose: PoseResult?,
        val otherPoses: List<PoseResult>,
        val width: Int,
        val height: Int,
        val atNanos: Long,
    )
    private var lastGood: Snapshot? = null
    private var framesWithDetection = 0L
    private var framesInferred = 0L

    /**
     * How many people get a pose per frame.
     *
     * Only the subject's pose is ever *reported* -- the protocol carries one
     * trainee. The others are landmarked purely so the operator can see, while
     * placing the phone, who else is in frame and whether the right person is
     * being tracked. Each extra pose costs ~4 ms (crop + NPU), so this is
     * bounded rather than "all detections": at 15 fps the frame budget is
     * 66 ms and detection already spends ~10 ms.
     */
    private val posesPerFrame = 3
    private var client: IngestClient? = null
    private var serverStatus: String = "disconnected"

    /**
     * The exercise -> asset name this build ships a classifier for.
     *
     * Adding an exercise (docs/ADDING_AN_EXERCISE.md) means fitting
     * `scripts/train_form_model.py --exercise <name>` and adding a line here
     * -- nothing else in this class names an exercise.
     */
    private val formClassifierAssets = mapOf(
        DEFAULT_EXERCISE to "${DEFAULT_EXERCISE}_lr.json",
        "bicep" to "bicep_lr.json",
        "lunge" to "lunge_lr.json",
    )

    /**
     * Loaded form classifiers, keyed by exercise. Missing from the map is a
     * working station that reports no `form_reason_codes`, not a broken one:
     * the protocol makes the field optional precisely so a phone without a
     * form model is still a valid station. What must not happen is silently
     * reporting *nothing* while looking like it is classifying, so the status
     * strip names the state either way.
     */
    private var formClassifiers: Map<String, FormClassifier> = emptyMap()
    private var formClassifierStatus: String = "not loaded"

    /**
     * The exercise this station is watching, sent on every observation.
     *
     * Load-bearing on the laptop: it selects the scoring weight profile
     * (`[scoring.exercise_weights]`). A station set to "plank" that failed to
     * report it would have a correct plank scored as a fall, so this is
     * defaulted rather than left blank, and shown on screen.
     */
    private var exercise: String = DEFAULT_EXERCISE

    /** The most recent verdict, for the status strip only — never sent. */
    private var lastVerdict: FormClassifier.Verdict? = null

    private var scoreThreshold = 0.35
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var frameCount = 0L
    private var lastDetections = 0
    private var inferenceMsEma = 0.0
    private var fpsEma = 0.0
    private var lastFrameNanos = 0L

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) bindCamera() else render("camera permission denied") }

    private val pickModelFiles = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        if (uris.isNotEmpty()) {
            val summary = modelStore.importFromUris(uris)
            openYolo26()
            openDetector()
            openPoseEstimator()
            render(summary)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // A station is propped on a tripod for a whole class; a screen that
        // dozes off mid-set silently stops the camera feed with it.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        applyWindowInsets()
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        debugPanel = findViewById(R.id.debugPanel)
        stateText = findViewById(R.id.stateText)
        stateDot = findViewById(R.id.stateDot)
        toggleButton = findViewById(R.id.toggle)
        thresholdLabel = findViewById(R.id.thresholdLabel)
        thresholdRow = findViewById(R.id.thresholdRow)

        deviceId = DeviceIdentity.deviceId(this)
        modelStore = ModelStore(this)
        openYolo26()          // if staged, it supersedes the two-model path
        openDetector()
        openPoseEstimator()
        openFormClassifiers()

        toggleButton.setOnClickListener {
            val now = !running.get()
            // Either backend is sufficient; the single-stage model does not
            // need the two-model path to be staged at all.
            if (now && detector == null && yolo26 == null) {
                render("cannot start: $detectorStatus")
                return@setOnClickListener
            }
            // Retire in-flight frames before clearing, or one of them will
            // repaint over the clear and linger.
            sessionEpoch.incrementAndGet()
            running.set(now)
            toggleButton.text = if (now) "Stop" else "Start"
            toggleButton.setBackgroundResource(
                if (now) R.drawable.bg_action_stop else R.drawable.bg_action_primary
            )
            toggleButton.setTextColor(
                if (now) getColor(R.color.fault) else android.graphics.Color.parseColor("#0E1113")
            )
            if (!now) {
                overlay.update(emptyList(), null, null, emptyList(), 1, 1)
                subjectTracker.reset()
                lastGood = null
                framesInferred = 0; framesWithDetection = 0
                lastDetections = 0
            }
            render()
        }
        findViewById<Button>(R.id.debug).setOnClickListener {
            debugVisible = !debugVisible
            debugPanel.visibility = if (debugVisible) View.VISIBLE else View.GONE
            thresholdRow.visibility = if (debugVisible) View.VISIBLE else View.GONE
            // Model import is a debug-time affordance too: staging happens over
            // adb in practice, and the picker is only reachable when something
            // has gone wrong enough to need it.
            render()
        }
        findViewById<Button>(R.id.debug).setOnLongClickListener {
            pickModelFiles.launch(arrayOf("*/*")); true
        }
        findViewById<Button>(R.id.connect).setOnClickListener { showConnectDialog() }
        findViewById<Button>(R.id.flip).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
            bindCamera()
        }
        findViewById<SeekBar>(R.id.threshold).setOnSeekBarChangeListener(
            object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(bar: SeekBar, progress: Int, fromUser: Boolean) {
                    scoreThreshold = 0.05 + progress / 100.0
                    thresholdLabel.text = "thr %.2f".format(scoreThreshold)
                }
                override fun onStartTrackingTouch(bar: SeekBar) {}
                override fun onStopTrackingTouch(bar: SeekBar) {}
            })

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) bindCamera() else requestCamera.launch(Manifest.permission.CAMERA)
        render()
    }

    /**
     * Keep the controls out from under the system bars.
     *
     * targetSdk 35 means the window is laid out edge to edge by default, so
     * without this the button row sits beneath the navigation bar: visible,
     * and not reliably tappable. It is worse than it looks in one orientation,
     * because the bar moves — bottom in portrait, side in landscape — so a
     * fixed margin fixes one and breaks the other. Insets are asked for
     * instead, and re-applied on rotation because the activity is not recreated.
     *
     * The preview deliberately keeps its full bleed; only the chrome is inset.
     */
    private fun applyWindowInsets() {
        val chip = findViewById<LinearLayout>(R.id.statusChip)
        val panel = findViewById<TextView>(R.id.debugPanel)
        val bottom = findViewById<LinearLayout>(R.id.bottomBar)
        val density = resources.displayMetrics.density
        fun dp(value: Int) = (value * density).toInt()

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            // The floating chrome is positioned by margin, not padding: these
            // views sit over the preview rather than in a column beside it, so
            // padding would grow their backgrounds into the bars instead of
            // moving them clear.
            (chip.layoutParams as android.widget.FrameLayout.LayoutParams).apply {
                leftMargin = bars.left + dp(12)
                topMargin = bars.top + dp(12)
            }.also { chip.layoutParams = it }

            (panel.layoutParams as android.widget.FrameLayout.LayoutParams).apply {
                leftMargin = bars.left + dp(12)
                topMargin = bars.top + dp(64)
            }.also { panel.layoutParams = it }

            bottom.setPadding(
                bars.left + dp(16), bottom.paddingTop,
                bars.right + dp(16), bars.bottom + dp(12),
            )
            insets
        }
    }

    /** Open the active model on the NPU, or record exactly why not. */
    private fun openDetector() {
        detector?.close()
        detector = null
        val entry = modelStore.active()
        detectorStatus = if (entry == null) {
            "no model staged — Model button, or adb push (android/README.md)"
        } else if (!entry.usable) {
            "model '${entry.name}' has no sidecar (.json) — regenerate with scripts/gen_yolox_fixture.py"
        } else {
            try {
                val backend = File(applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                detector = QnnDetector(entry.onnx, entry.sidecar, backend)
                "'${entry.name}' on NPU (${detector!!.inputSize}²), sha " +
                    detector!!.sidecar.modelSha256.take(8)
            } catch (e: NpuUnavailableException) {
                "NPU UNAVAILABLE — ${e.message?.take(160)}"
            }
        }
    }

    /**
     * The single-stage prototype, selected purely by its artifact being staged.
     *
     * A file-presence flag rather than a setting: staging it is already a
     * deliberate act, and it keeps the two-model path untouched underneath so a
     * bad result is one `adb shell rm` away from being reverted.
     */
    private fun openYolo26() {
        yolo26?.close()
        yolo26 = null
        val model = java.io.File(modelStore.modelsDir, "yolo26_pose_fp32.onnx")
        if (!model.isFile) { backendLabel = "yolox+blazepose"; return }
        try {
            val backend = java.io.File(applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            yolo26 = Yolo26PoseEstimator(model, backend)
            backendLabel = "yolo26-pose (single-stage)"
        } catch (e: NpuUnavailableException) {
            backendLabel = "yolo26 FAILED — ${e.message?.take(90)}"
        }
    }

    /** Pose is optional: absent model degrades to zero-conf keypoints, stated. */
    private fun openPoseEstimator() {
        poseEstimator?.close()
        poseEstimator = null
        val model = java.io.File(modelStore.modelsDir, "pose_landmark_fp32.onnx")
        poseStatus = if (!model.isFile) {
            "not staged (pose_landmark_fp32.onnx)"
        } else {
            try {
                val backend = java.io.File(applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                poseEstimator = PoseEstimator(model, backend)
                "on NPU fp16"
            } catch (e: NpuUnavailableException) {
                "NPU UNAVAILABLE — ${e.message?.take(120)}"
            }
        }
    }

    private fun bindCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                // Frames must arrive oriented for the *current* display, not
                // for whatever it was at bind time.
                .setTargetRotation(display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::onFrame) }
                .also { imageAnalysis = it }
            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.Builder().requireLensFacing(lensFacing).build(),
                preview, analysis,
            )
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Rotation, without tearing anything down.
     *
     * The activity survives the configuration change (see the manifest), so all
     * that is needed is to tell CameraX which way the display now faces.
     * `ImageProxy.imageInfo.rotationDegrees` is expressed relative to the
     * analysis use case's target rotation, so leaving it stale is precisely how
     * the overlay ends up drawn against a frame the preview is rendering a
     * quarter-turn away from — the boxes land in the wrong place rather than
     * disappearing, which is harder to spot.
     *
     * The last held overlay is dropped: it was computed in the previous
     * orientation's pixel space and would be mapped through the new one.
     */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalysis?.targetRotation = display.rotation
        // The bars move with the display; ask for the new insets.
        ViewCompat.requestApplyInsets(findViewById(R.id.root))
        lastGood = null
        overlay.update(emptyList(), null, null, emptyList(), 1, 1)
        render()
    }

    private fun onFrame(image: ImageProxy) {
        val epoch = sessionEpoch.get()
        try {
            frameCount += 1
            val nowNanos = System.nanoTime()
            if (lastFrameNanos != 0L) {
                val fps = 1e9 / (nowNanos - lastFrameNanos)
                fpsEma = if (fpsEma == 0.0) fps else 0.9 * fpsEma + 0.1 * fps
            }
            lastFrameNanos = nowNanos

            val det = detector
            val single = yolo26
            if (running.get() && (single != null || det != null)) {
                val upright = rotateUpright(image.toBitmap(), image.imageInfo.rotationDegrees)

                val detections: List<Detection>
                var singlePoses: Map<Detection, PoseResult> = emptyMap()
                val started = System.nanoTime()
                if (single != null) {
                    val (canvas, info) = letterbox(upright, single.inputSize, YOLOX_PAD_VALUE)
                    val tensor = toNchwRgbFloats(canvas)   // float [0,1], not uint8
                    canvas.recycle()
                    val pairs = single.detectAndPose(
                        tensor, info, upright.width, upright.height, scoreThreshold.toFloat(),
                    )
                    detections = pairs.map { it.first }
                    singlePoses = pairs.toMap()
                } else {
                    val (canvas, info) = letterbox(upright, det!!.inputSize, det.sidecar.letterboxPadValue)
                    val tensor = toNchwRgbBytes(canvas)
                    canvas.recycle()
                    detections = det.detect(
                        tensor, info, upright.width, upright.height, scoreThreshold,
                    )
                }
                val ms = (System.nanoTime() - started) / 1e6
                inferenceMsEma = if (inferenceMsEma == 0.0) ms else 0.9 * inferenceMsEma + 0.1 * ms
                lastDetections = detections.size

                // Which of them is this station's trainee -- by size and with
                // hysteresis, not per-frame confidence. See SubjectTracker.
                val subject = subjectTracker.select(detections)

                // Pose the subject first, then the next largest others up to
                // the budget. Only the subject's pose is reported.
                var subjectPose: PoseResult? = null
                val otherPoses = ArrayList<PoseResult>()
                val estimator = poseEstimator
                if (single != null) {
                    // Poses already came back with the detections; no second pass.
                    subjectPose = subject?.let { singlePoses[it] }
                    detections.filter { it !== subject }.forEach { d ->
                        singlePoses[d]?.let { otherPoses.add(it) }
                    }
                    lastVisibleKeypoints = subjectPose?.keypointsConf?.count { it >= 0.3f } ?: 0
                    lastPoseScore = subjectPose?.poseScore ?: 0f
                    poseMsEma = 0.0   // folded into the single inference above
                } else if (estimator != null && detections.isNotEmpty()) {
                    val ordered = buildList {
                        subject?.let { add(it) }
                        addAll(
                            detections.filter { it !== subject }
                                .sortedByDescending { (it.x1 - it.x0) * (it.y1 - it.y0) }
                        )
                    }.take(posesPerFrame)

                    val poseStarted = System.nanoTime()
                    for (det in ordered) {
                        val p = estimator.estimate(upright, det) ?: continue
                        if (det === subject) subjectPose = p else otherPoses.add(p)
                    }
                    val poseMs = (System.nanoTime() - poseStarted) / 1e6
                    poseMsEma = if (poseMsEma == 0.0) poseMs else 0.9 * poseMsEma + 0.1 * poseMs
                    lastVisibleKeypoints = subjectPose?.keypointsConf?.count { it >= 0.3f } ?: 0
                    lastPoseScore = subjectPose?.poseScore ?: 0f
                }

                framesInferred += 1
                if (detections.isNotEmpty()) framesWithDetection += 1

                // Camera motion blurs a frame and the detector drops under
                // threshold, so empty frames are common while the phone is
                // being aimed. Clearing on each one flickers; holding the last
                // result briefly does not, and the overlay fades it so a held
                // skeleton reads as held. Only live frames are ever reported.
                // A stop that landed while this frame was on the NPU makes
                // everything below stale; drop it rather than repaint over the
                // clear the stop already did.
                if (epoch != sessionEpoch.get()) { upright.recycle(); return }

                if (detections.isNotEmpty()) {
                    lastGood = Snapshot(
                        detections, subject, subjectPose, otherPoses,
                        upright.width, upright.height, System.nanoTime(),
                    )
                    overlay.update(
                        detections, subject, subjectPose, otherPoses,
                        upright.width, upright.height, ageMs = 0L,
                    )
                } else {
                    val held = lastGood
                    val ageMs = held?.let { (System.nanoTime() - it.atNanos) / 1_000_000 } ?: Long.MAX_VALUE
                    if (held != null && ageMs < DetectionOverlayView.HOLD_MS) {
                        overlay.update(
                            held.detections, held.subject, held.subjectPose, held.otherPoses,
                            held.width, held.height, ageMs = ageMs,
                        )
                    } else {
                        lastGood = null
                        overlay.update(emptyList(), null, null, emptyList(), 1, 1)
                    }
                }

                if (subject != null) {
                    val codes = classifyForm(subjectPose, upright.width, upright.height)
                    client?.send(
                        Observation.fromDetection(
                            ts = System.currentTimeMillis() / 1000.0,
                            det = subject, pose = subjectPose,
                            frameWidth = upright.width, frameHeight = upright.height,
                        ).copy(
                            exercise = exercise,
                            formOk = if (subjectPose != null) codes.isEmpty() else null,
                            formReasonCodes = codes,
                        )
                    )
                } else {
                    // No subject means no verdict, not the previous one. A
                    // form judgement that outlives the person it was made
                    // about is worse than a blank: it reads as current.
                    lastVerdict = null
                    // Tell the laptop we are *here and watching nobody*, at a
                    // low rate. Sending nothing at all is what made an empty
                    // rack indistinguishable from a dead phone: the server
                    // evicted this station at track_ttl_s, refused its next
                    // message, and it reconnected -- a flap that begins the
                    // moment a station is set up before its trainee arrives.
                    val nowMs = System.currentTimeMillis()
                    if (nowMs - lastIdleSentMs >= IDLE_INTERVAL_MS) {
                        lastIdleSentMs = nowMs
                        client?.sendIdle(nowMs / 1000.0)
                    }
                }
                upright.recycle()
            }
        } finally {
            image.close()
        }
        if (frameCount % 10L == 0L) runOnUiThread { render() }
    }

    /**
     * Load every exercise's form classifier from assets.
     *
     * Unlike the pose models these are not staged over adb: each is a few
     * dozen floats per class, small enough to ship in the APK, and each must
     * match the server's `[scoring.form_error_vocab]` at the version it was
     * built against. A missing or malformed artifact is caught and named
     * rather than thrown -- a station that cannot classify one exercise's
     * form is still worth running for the others, since fall, stillness and
     * occlusion are all scored from the keypoints alone regardless.
     */
    private fun openFormClassifiers() {
        val loaded = mutableMapOf<String, FormClassifier>()
        val failures = mutableListOf<String>()
        for ((ex, asset) in formClassifierAssets) {
            try {
                val json = assets.open(asset).bufferedReader().use { it.readText() }
                loaded[ex] = FormClassifier.fromJson(json)
            } catch (t: Throwable) {
                failures += "$ex: ${t.message}"
            }
        }
        formClassifiers = loaded
        formClassifierStatus = if (failures.isEmpty()) {
            "loaded ${loaded.keys.sorted()}"
        } else {
            "loaded ${loaded.keys.sorted()}; unavailable: ${failures.joinToString("; ")}"
        }
    }

    /**
     * The form verdict for one frame, or no codes when it cannot be trusted.
     *
     * Returns empty for an exercise with no shipped classifier. Reporting one
     * exercise's verdict for another would be worse than reporting nothing:
     * the code would still be in the server's vocabulary, so nothing
     * downstream could tell it was nonsense.
     */
    private fun classifyForm(pose: PoseResult?, frameWidth: Int, frameHeight: Int): List<String> {
        val classifier = formClassifiers[exercise] ?: return emptyList()
        val p = pose ?: run { lastVerdict = null; return emptyList() }
        return try {
            // `PoseResult` is in frame pixels; the model was fitted on
            // normalized coordinates. Under the shipped `bbox` encoding the
            // frame size cancels out of the landmark-box normalization, so
            // this conversion is currently a no-op mathematically -- but it is
            // done anyway, because that cancellation is a property of one
            // encoding and not of the interface. A `raw`-encoded artifact
            // would be silently wrong without it.
            val xy = FloatArray(p.keypointsXy.size)
            for (k in 0 until NUM_KEYPOINTS) {
                xy[k * 2] = p.keypointsXy[k * 2] / frameWidth
                xy[k * 2 + 1] = p.keypointsXy[k * 2 + 1] / frameHeight
            }
            val verdict = classifier.classify(xy, p.keypointsConf)
            lastVerdict = verdict
            verdict.formReasonCodes
        } catch (t: Throwable) {
            // A malformed pose is a bug worth seeing, not a reason to drop the
            // whole observation: the keypoint-derived features still stand.
            formClassifierStatus = "$exercise classify failed: ${t.message}"
            lastVerdict = null
            emptyList()
        }
    }

    /** A small caption above a dialog field. Hints vanish the moment someone
     *  types, which is exactly when they stop being able to check what they
     *  entered against what was asked for. */
    private fun fieldLabel(resId: Int) = TextView(this).apply {
        setText(resId)
        textSize = 12f
        alpha = 0.7f
        setPadding(0, 24, 0, 0)
    }

    private fun showConnectDialog() {
        val prefs = getSharedPreferences("argus_edge_server", MODE_PRIVATE)
        val urlInput = EditText(this).apply {
            hint = "ws://192.168.1.20:8765"
            setText(prefs.getString("url", ""))
            inputType = InputType.TYPE_TEXT_VARIATION_URI
        }
        // One identity field, not two. The dialog used to ask for a trainee id
        // *and* a name shown to the instructor — a systems question and a human
        // one about the same person. Whoever is setting up a rack has no reason
        // to know a triage key exists. So: ask who the trainee is, and use the
        // device's own id underneath, which is already stable and unique across
        // a floor.
        val nameInput = EditText(this).apply {
            hint = getString(R.string.trainee_name_hint)
            setText(prefs.getString("display_name", ""))
        }
        // Remembered from the session that was picked, and sent in the hello
        // so the server can refuse a mismatch. Empty when the address was
        // typed by hand: a phone that never heard a beacon cannot know the
        // name, and must still be able to connect.
        var chosenSession = prefs.getString("session", "") ?: ""
        val sessionLabel = TextView(this).apply {
            text = if (chosenSession.isEmpty()) getString(R.string.session_none)
                   else getString(R.string.session_joining, chosenSession)
        }

        fun choose(server: Discovery.Server) {
            urlInput.setText(server.wsUrl)
            chosenSession = server.sessionName ?: ""
            sessionLabel.text = when {
                chosenSession.isEmpty() -> getString(R.string.session_none)
                server.needsApproval -> getString(R.string.session_joining_manual, chosenSession)
                else -> getString(R.string.session_joining, chosenSession)
            }
        }

        // Discovery fills the address in; it never connects on its own. A
        // beacon is an unauthenticated datagram from whoever is on the Wi-Fi,
        // so it gets to make a suggestion, not a decision — a human still
        // reads it and presses Connect.
        val findButton = Button(this).apply {
            text = getString(R.string.find_server)
            setOnClickListener {
                isEnabled = false
                text = getString(R.string.find_server_searching)
                Executors.newSingleThreadExecutor().execute {
                    val servers = Discovery.listen(this@MainActivity)
                    runOnUiThread {
                        isEnabled = true
                        when {
                            servers.isEmpty() ->
                                text = getString(R.string.find_server_none)
                            servers.size == 1 -> {
                                choose(servers[0])
                                text = getString(R.string.find_server_found)
                            }
                            else -> {
                                // More than one instructor on the floor: name
                                // them and let a human pick. Choosing for them
                                // would put a trainee on a console nobody
                                // meant to be watching.
                                val labels = servers.map { it.label }.toTypedArray()
                                AlertDialog.Builder(this@MainActivity)
                                    .setTitle(R.string.find_server_choose)
                                    .setItems(labels) { _, which -> choose(servers[which]) }
                                    .show()
                                text = getString(R.string.find_server)
                            }
                        }
                    }
                }
            }
        }
        // The exercise is not cosmetic: it selects the laptop's scoring weight
        // profile. Setting it to something with no profile is allowed and
        // scores on the default weights -- but leaving it blank for a plank
        // means a correct plank is scored as a fall, so it defaults filled in.
        // A picker, not free text. This string selects the *server's* scoring
        // profile, so a typo does not fail loudly — it silently scores a plank
        // on standing-HIIT weights, which is the exact misfire the profiles
        // exist to prevent. The options are the exercises this phone can
        // actually classify, plus whatever was configured before.
        val exercises = (formClassifierAssets.keys.toList() +
            listOfNotNull(prefs.getString("exercise", null))).distinct()
        var chosenExercise = prefs.getString("exercise", DEFAULT_EXERCISE) ?: DEFAULT_EXERCISE
        val exerciseSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity, android.R.layout.simple_spinner_dropdown_item, exercises
            )
            setSelection(exercises.indexOf(chosenExercise).coerceAtLeast(0))
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?, view: View?, position: Int, id: Long,
                ) { chosenExercise = exercises[position] }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(findButton)
            addView(sessionLabel)
            addView(fieldLabel(R.string.field_server))
            addView(urlInput)
            addView(fieldLabel(R.string.field_trainee))
            addView(nameInput)
            addView(fieldLabel(R.string.field_exercise))
            addView(exerciseSpinner)
        }
        AlertDialog.Builder(this)
            .setTitle("Ingest server (docs/PROTOCOL.md)")
            .setView(column)
            .setPositiveButton("Connect") { _, _ ->
                val url = urlInput.text.toString().trim()
                // The device id *is* the trainee id: stable, unique across a
                // floor, and not something a human should have to invent.
                val trainee = deviceId
                val display = nameInput.text.toString().trim()
                exercise = chosenExercise
                prefs.edit()
                    .putString("url", url)
                    .putString("display_name", display)
                    .putString("session", chosenSession)
                    .putString("exercise", exercise)
                    .apply()
                client?.disconnect()
                client = IngestClient(
                    url,
                    stationId = deviceId,
                    traineeId = trainee,
                    exercisePlan = exercise,
                    displayName = display,
                    sessionName = chosenSession,
                ) { s ->
                    serverStatus = s
                    runOnUiThread { render() }
                }.also { it.connect() }
            }
            .setNegativeButton("Disconnect") { _, _ ->
                client?.disconnect(); client = null
                serverStatus = "disconnected"
                render()
            }
            .show()
    }

    /**
     * Two audiences, two surfaces.
     *
     * The chip answers the three questions someone placing a phone has — is it
     * running, is the laptop receiving, is it seeing anybody — in one line and
     * one colour. The panel behind Debug answers everything else, and is the
     * same dump this screen used to show unconditionally. Showing both at once
     * was the mistake: it made the engineering detail the loudest thing on a
     * screen whose primary user does not need any of it.
     */
    private fun render(extra: String? = null) {
        val fault = detector == null && yolo26 == null ||
            detectorStatus.startsWith("NPU UNAVAILABLE") ||
            backendLabel.startsWith("yolo26 FAILED") ||
            serverStatus.startsWith("REJECTED")

        val (colour, label) = when {
            fault -> R.color.fault to when {
                detector == null && yolo26 == null -> "No model staged"
                serverStatus.startsWith("REJECTED") -> "Server refused this station"
                else -> "NPU unavailable"
            }
            !running.get() -> R.color.on_surface_dim to "Ready — press Start"
            lastDetections > 0 -> R.color.live to buildString {
                append(if (lastDetections == 1) "Tracking" else "Tracking $lastDetections people")
                append(if (serverStatus.startsWith("streaming")) " · sending" else " · not connected")
                // The form verdict belongs on the one line the operator reads.
                // A station whose whole job is judging an exercise should say
                // what it judged -- and, just as importantly, say when it
                // cannot see enough of the body to judge at all, which is
                // otherwise indistinguishable from "correct".
                lastVerdict?.let { v ->
                    append(" · ")
                    append(
                        when {
                            v.label == FormClassifier.UNKNOWN -> "body not fully in frame"
                            !v.confident -> "form unclear"
                            v.formReasonCodes.isEmpty() -> "form ok"
                            else -> v.formReasonCodes.first().replace('_', ' ')
                        }
                    )
                }
            }
            else -> R.color.attention to buildString {
                append("No one in frame")
                append(if (serverStatus.startsWith("streaming")) " · connected" else "")
            }
        }
        stateText.text = extra ?: label
        stateDot.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(getColor(colour))
        }

        if (!debugVisible) return
        debugPanel.text = buildString {
            append("station   ").append(deviceId).append('\n')
            append("backend   ").append(backendLabel).append('\n')
            if (yolo26 == null) append("model     ").append(detectorStatus).append('\n')
            append("camera    %.1f fps, frame %d".format(fpsEma, frameCount)).append('\n')
            append("inference ")
            append(
                if (running.get())
                    "%.1f ms, %d person(s), thr %.2f, hit %d%%".format(
                        inferenceMsEma, lastDetections, scoreThreshold,
                        if (framesInferred > 0) framesWithDetection * 100 / framesInferred else 0,
                    )
                else "stopped"
            ).append('\n')
            append("pose      ")
            append(
                if (yolo26 != null && running.get())
                    "in-model, %d/17 keypoints, score %.2f%s".format(
                        lastVisibleKeypoints, lastPoseScore,
                        if (subjectTracker.switches > 0)
                            " — %d subject switch(es)".format(subjectTracker.switches) else "",
                    )
                else if (poseEstimator != null && running.get())
                    "%.1f ms, %d/13 keypoints, score %.2f".format(
                        poseMsEma, lastVisibleKeypoints, lastPoseScore)
                else poseStatus
            ).append('\n')
            // Which exercise is announced, and what the form model made of the
            // last frame. Both matter to whoever places the phone: the exercise
            // selects the server's scoring profile, and a station reporting
            // "no model" is scoring on geometry alone -- a real state, but not
            // one to discover from a flat dashboard.
            append("form      ").append(exercise).append(" — ")
            append(
                when {
                    exercise !in formClassifiers -> "no classifier for this exercise ($formClassifierStatus)"
                    !running.get() -> formClassifierStatus
                    else -> lastVerdict?.describe() ?: "no pose this frame"
                }
            ).append('\n')
            append("server    ").append(serverStatus)
            append("\n\nlong-press Debug to import a model")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        client?.disconnect()
        detector?.close()
        poseEstimator?.close()
        yolo26?.close()
        analysisExecutor.shutdown()
    }
}
