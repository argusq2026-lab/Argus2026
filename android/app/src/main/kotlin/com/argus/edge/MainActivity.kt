package com.argus.edge

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
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

                subject?.let {
                    client?.send(
                        Observation.fromDetection(
                            ts = System.currentTimeMillis() / 1000.0,
                            det = it, pose = subjectPose,
                            frameWidth = upright.width, frameHeight = upright.height,
                        )
                    )
                }
                upright.recycle()
            }
        } finally {
            image.close()
        }
        if (frameCount % 10L == 0L) runOnUiThread { render() }
    }

    private fun showConnectDialog() {
        val prefs = getSharedPreferences("argus_edge_server", MODE_PRIVATE)
        val urlInput = EditText(this).apply {
            hint = "ws://192.168.1.20:8765"
            setText(prefs.getString("url", ""))
            inputType = InputType.TYPE_TEXT_VARIATION_URI
        }
        val traineeInput = EditText(this).apply {
            hint = "trainee id"
            setText(prefs.getString("trainee", deviceId))
        }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(urlInput)
            addView(traineeInput)
        }
        AlertDialog.Builder(this)
            .setTitle("Ingest server (docs/PROTOCOL.md)")
            .setView(column)
            .setPositiveButton("Connect") { _, _ ->
                val url = urlInput.text.toString().trim()
                val trainee = traineeInput.text.toString().trim().ifEmpty { deviceId }
                prefs.edit().putString("url", url).putString("trainee", trainee).apply()
                client?.disconnect()
                client = IngestClient(url, stationId = deviceId, traineeId = trainee) { s ->
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
