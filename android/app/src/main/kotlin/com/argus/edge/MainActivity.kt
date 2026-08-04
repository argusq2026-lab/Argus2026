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
import java.util.concurrent.atomic.AtomicBoolean

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
    private lateinit var statusView: TextView
    private lateinit var toggleButton: Button
    private lateinit var thresholdLabel: TextView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var imageAnalysis: ImageAnalysis? = null
    private val running = AtomicBoolean(false)

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
        statusView = findViewById(R.id.status)
        toggleButton = findViewById(R.id.toggle)
        thresholdLabel = findViewById(R.id.thresholdLabel)

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
            running.set(now)
            toggleButton.text = if (now) "Stop" else "Start"
            if (!now) {
                overlay.update(emptyList(), null, null, emptyList(), 1, 1)
                subjectTracker.reset()
                lastGood = null
                framesInferred = 0; framesWithDetection = 0
            }
            render()
        }
        findViewById<Button>(R.id.importModel).setOnClickListener {
            pickModelFiles.launch(arrayOf("*/*"))
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
        val status = findViewById<TextView>(R.id.status)
        val controls = findViewById<LinearLayout>(R.id.controls)
        val thresholdRow = findViewById<LinearLayout>(R.id.thresholdRow)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            status.setPadding(
                status.paddingLeft.coerceAtLeast(bars.left + 12),
                bars.top + 12, status.paddingRight, status.paddingBottom,
            )
            for (row in listOf(thresholdRow, controls)) {
                row.setPadding(bars.left + 8, row.paddingTop, bars.right + 8, row.paddingBottom)
            }
            controls.setPadding(
                controls.paddingLeft, controls.paddingTop,
                controls.paddingRight, bars.bottom + 8,
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

    private fun render(extra: String? = null) {
        statusView.text = buildString {
            append("station   ").append(deviceId).append('\n')
            append("backend   ").append(backendLabel).append('\n')
            // Only name the two-model artifacts when they are the ones running.
            // Showing "model 'yolox'" while the single-stage backend does the
            // work states something untrue about what produced the boxes.
            if (yolo26 == null) append("model     ").append(detectorStatus).append('\n')
            append("camera    %.1f fps, frame %d".format(fpsEma, frameCount)).append('\n')
            append("inference ")
            append(
                if (running.get())
                    // Hit rate is the number to watch when the overlay looks
                    // intermittent: a low value is the detector losing the
                    // person on blurred frames, not the pipeline being slow.
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
                    // 13 of 17 is the ceiling: the 25-point export has no
                    // knees or ankles, so those four never light up.
                    "%.1f ms, %d/13 keypoints, score %.2f%s"
                        .format(
                            poseMsEma, lastVisibleKeypoints, lastPoseScore,
                            if (subjectTracker.switches > 0)
                                " — %d subject switch(es)".format(subjectTracker.switches) else "",
                        )
                else poseStatus
            ).append('\n')
            append("server    ").append(serverStatus)
            extra?.let { append('\n').append(it) }
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
