package com.argus.edge

import android.content.Context
import android.content.res.Configuration
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * A nursing station: camera, pose, and a CPR compression stream to the laptop.
 *
 * A second plain Activity rather than a mode inside [MainActivity], for the
 * same reason [DashboardActivity] is one — [MainActivity] is the fitness
 * screen, welded to an exercise picker, a rep counter and a form classifier
 * that mean nothing here, and threading a use case through all of it would
 * make one long file answer two questions.
 *
 * What it *reuses* is everything that was already its own class: [ModelStore],
 * [QnnDetector] / [Yolo26PoseEstimator] / [PoseEstimator], [SubjectTracker],
 * [DetectionOverlayView] and [IngestClient]. What is written again here is the
 * camera bind and the frame loop, and both are materially smaller than
 * fitness's — one subject, no second classifier pass, no rep counting.
 *
 * The privacy property is unchanged and is the same wiring: the frame is a
 * local, and only the normalized box and keypoints ever reach the network.
 */
class NursingActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DetectionOverlayView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var imageAnalysis: ImageAnalysis? = null
    private val running = AtomicBoolean(false)

    /** Bumped on every stop, so a frame still on the NPU cannot repaint after it. */
    private val sessionEpoch = AtomicInteger(0)

    private lateinit var modelStore: ModelStore
    private var detector: QnnDetector? = null
    private var poseEstimator: PoseEstimator? = null
    private var yolo26: Yolo26PoseEstimator? = null
    private var modelStatus: String = "no model"
    private var backendLabel: String = "yolox+blazepose"

    private val subjectTracker = SubjectTracker()
    private var client: IngestClient? = null
    private var serverStatus: String = "disconnected"

    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var frameCount = 0L
    private var fpsEma = 0.0
    private var lastFrameNanos = 0L
    private var inferenceMsEma = 0.0
    private var debugVisible = false
    private var lastIdleSentMs = 0L

    /** Trailing wrist heights, for the on-screen rate echo. See [localRate]. */
    private val wristTrail = ArrayDeque<Pair<Double, Double>>()

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) bindCamera() else render("camera permission denied") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_nursing)
        title = getString(R.string.tile_nursing_title)

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        modelStore = ModelStore(this)

        openModels()
        applyWindowInsets()

        findViewById<Button>(R.id.toggle).setOnClickListener { view ->
            val next = !running.get()
            running.set(next)
            if (!next) {
                // Stopping clears the screen and invalidates in-flight frames,
                // so nothing lands after the stop and reads as current.
                sessionEpoch.incrementAndGet()
                wristTrail.clear()
                overlay.update(emptyList(), null, null, emptyList(), 1, 1)
            }
            (view as Button).text =
                getString(if (next) R.string.nursing_stop else R.string.nursing_start)
            render()
        }
        findViewById<Button>(R.id.connect).setOnClickListener { showConnectDialog() }
        findViewById<Button>(R.id.debug).setOnClickListener {
            debugVisible = !debugVisible
            findViewById<TextView>(R.id.debugPanel).visibility =
                if (debugVisible) View.VISIBLE else View.GONE
            render()
        }

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            bindCamera()
        } else {
            requestCamera.launch(android.Manifest.permission.CAMERA)
        }
        render()
    }

    private fun applyWindowInsets() {
        val chip = findViewById<LinearLayout>(R.id.statusChip)
        val bottom = findViewById<LinearLayout>(R.id.bottomBar)
        val density = resources.displayMetrics.density
        fun dp(value: Int) = (value * density).toInt()
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            (chip.layoutParams as android.widget.FrameLayout.LayoutParams).apply {
                topMargin = bars.top + dp(12)
                marginStart = bars.left + dp(12)
            }
            bottom.setPadding(bars.left + dp(16), dp(16), bars.right + dp(16), bars.bottom + dp(16))
            chip.requestLayout()
            insets
        }
    }

    /**
     * Load whatever pose stack is staged, preferring the single-stage export.
     *
     * Same file-presence selection as [MainActivity]: staging the artifact is
     * already a deliberate act, so it needs no second setting.
     */
    private fun openModels() {
        val backend = File(applicationInfo.nativeLibraryDir, "libQnnHtp.so")

        val single = File(modelStore.modelsDir, "yolo26_pose_fp32.onnx")
        if (single.isFile) {
            try {
                yolo26 = Yolo26PoseEstimator(single, backend)
                backendLabel = "yolo26-pose (single-stage)"
                modelStatus = "yolo26 on NPU"
                return
            } catch (e: NpuUnavailableException) {
                modelStatus = "yolo26 FAILED — ${e.message?.take(90)}"
            }
        }

        val entry = modelStore.active()
        modelStatus = if (entry == null || !entry.usable) {
            "no model staged — see android/README.md"
        } else {
            try {
                detector = QnnDetector(entry.onnx, entry.sidecar, backend)
                val pose = File(modelStore.modelsDir, "pose_landmark_fp32.onnx")
                if (pose.isFile) poseEstimator = PoseEstimator(pose, backend)
                backendLabel = "yolox+blazepose"
                "'${entry.name}' on NPU" + if (poseEstimator == null) " (no pose model)" else ""
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

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalysis?.targetRotation = display.rotation
        ViewCompat.requestApplyInsets(findViewById(R.id.root))
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
                    val tensor = toNchwRgbFloats(canvas)
                    canvas.recycle()
                    val pairs = single.detectAndPose(
                        tensor, info, upright.width, upright.height, SCORE_THRESHOLD.toFloat(),
                    )
                    detections = pairs.map { it.first }
                    singlePoses = pairs.toMap()
                } else {
                    val (canvas, info) = letterbox(upright, det!!.inputSize, det.sidecar.letterboxPadValue)
                    val tensor = toNchwRgbBytes(canvas)
                    canvas.recycle()
                    detections = det.detect(
                        tensor, info, upright.width, upright.height, SCORE_THRESHOLD,
                    )
                }
                inferenceMsEma = ((System.nanoTime() - started) / 1e6).let {
                    if (inferenceMsEma == 0.0) it else 0.9 * inferenceMsEma + 0.1 * it
                }

                val subject = subjectTracker.select(detections)

                // Only the subject is posed. Fitness poses runners-up so the
                // overlay can show the rest of the floor; a nursing station is
                // watching one rescuer, and a second pose would cost NPU time
                // for a skeleton nothing scores.
                var subjectPose: PoseResult? = null
                if (single != null) {
                    subjectPose = subject?.let { singlePoses[it] }
                } else {
                    val estimator = poseEstimator
                    if (estimator != null && subject != null) {
                        subjectPose = estimator.estimate(upright, subject)
                    }
                }

                if (epoch != sessionEpoch.get()) { upright.recycle(); return }

                overlay.update(
                    detections, subject, subjectPose, emptyList(),
                    upright.width, upright.height, ageMs = 0L,
                )

                if (subject != null) {
                    noteWrist(subjectPose, upright.height)
                    client?.send(
                        Observation.fromDetection(
                            ts = System.currentTimeMillis() / 1000.0,
                            det = subject, pose = subjectPose,
                            frameWidth = upright.width, frameHeight = upright.height,
                            useCase = USE_CASE_NURSING,
                            procedure = PROCEDURE_CPR,
                        )
                    )
                } else {
                    wristTrail.clear()
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

    /** Keep the last [RATE_WINDOW_S] seconds of the better-tracked wrist. */
    private fun noteWrist(pose: PoseResult?, frameHeight: Int) {
        if (pose == null) return
        val left = pose.keypointsConf[KP_LEFT_WRIST]
        val right = pose.keypointsConf[KP_RIGHT_WRIST]
        val idx = if (left >= right) KP_LEFT_WRIST else KP_RIGHT_WRIST
        if (pose.keypointsConf[idx] < 0.3f) return
        val now = System.currentTimeMillis() / 1000.0
        wristTrail.addLast(now to pose.keypointsXy[idx * 2 + 1].toDouble() / frameHeight)
        while (wristTrail.isNotEmpty() && now - wristTrail.first().first > RATE_WINDOW_S) {
            wristTrail.removeFirst()
        }
    }

    /**
     * Compressions per minute, by counting peaks in the wrist trail.
     *
     * **Advisory only — the laptop is the authority.** This exists because a
     * rescuer cannot read a console across the room while their hands are on a
     * chest, and rate is the one CPR number that is actionable in the moment.
     * It is deliberately the simpler of the laptop's two estimators, and does
     * not attempt the autocorrelation cross-check that
     * `argus.triage.estimate_compression_rate` uses to decide whether a rate is
     * trustworthy at all. Where the two disagree, the laptop is right.
     */
    private fun localRate(): Int? {
        if (wristTrail.size < 12) return null
        val trail = wristTrail.toList()
        val span = trail.last().first - trail.first().first
        if (span < RATE_WINDOW_S / 2) return null

        val ys = trail.map { it.second }
        val mean = ys.average()
        val amplitude = kotlin.math.sqrt(ys.sumOf { (it - mean) * (it - mean) } / ys.size)
        if (amplitude < MIN_AMPLITUDE) return null

        val peaks = ArrayList<Double>()
        for (i in 1 until trail.size - 1) {
            val y = ys[i]
            if (y - mean < amplitude * 0.5) continue
            if (y < ys[i - 1] || y < ys[i + 1]) continue
            if (peaks.isNotEmpty() && trail[i].first - peaks.last() < MIN_PEAK_SEPARATION_S) continue
            peaks.add(trail[i].first)
        }
        if (peaks.size < 3) return null
        val intervals = peaks.zipWithNext { a, b -> b - a }.sorted()
        val median = intervals[intervals.size / 2]
        if (median <= 0.0) return null
        return (60.0 / median).roundToInt()
    }

    private fun showConnectDialog() {
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val field = EditText(this).apply {
            hint = getString(R.string.nursing_server_hint)
            setText(prefs.getString(KEY_URL, "") ?: "")
            setSingleLine()
        }
        val wrapper = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = (20 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad, pad, 0)
            addView(field)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.nursing_connect)
            .setView(wrapper)
            .setPositiveButton(R.string.nursing_connect) { _, _ ->
                val url = field.text.toString().trim()
                if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
                    serverStatus = "not a ws:// address"
                    render()
                    return@setPositiveButton
                }
                prefs.edit().putString(KEY_URL, url).apply()
                client?.disconnect()
                val id = DeviceIdentity.deviceId(this)
                client = IngestClient(
                    serverUrl = url,
                    stationId = id,
                    traineeId = id,
                    displayName = getString(R.string.tile_nursing_title),
                    // The laptop refuses this handshake unless its own
                    // [session] use_case is nursing -- which is the point.
                    useCase = USE_CASE_NURSING,
                    // Kept for the pre-connect case and for terminal errors;
                    // the live line comes from `describe()` in `render`.
                    onStateChange = { s -> runOnUiThread { serverStatus = s; render() } },
                ).also { it.connect() }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun render(extra: String? = null) {
        val dot = findViewById<View>(R.id.stateDot)
        val streaming = client?.state == IngestClient.State.STREAMING
        val colour = ContextCompat.getColor(
            this, if (running.get() && streaming) R.color.live else R.color.on_surface_dim
        )
        (dot.background?.mutate() as? GradientDrawable)?.setColor(colour)
            ?: dot.setBackgroundColor(colour)

        findViewById<TextView>(R.id.stateText).text = buildString {
            append(if (running.get()) "CPR · watching" else "CPR · stopped")
            // Ask the client rather than reusing the string from the last
            // transition: `onStateChange` fires when the *state* changes, so a
            // cached copy freezes at "streaming (0 sent)" and stays there while
            // hundreds of observations go out -- which reads as a broken link.
            append(" · ").append(client?.describe() ?: serverStatus)
            extra?.let { append(" · ").append(it) }
        }

        val rate = localRate()
        findViewById<TextView>(R.id.rateReadout).apply {
            text = if (rate == null) "--/min" else "$rate/min"
            setTextColor(
                ContextCompat.getColor(
                    this@NursingActivity,
                    if (rate != null && abs(rate - TARGET_MID) <= TARGET_HALF_WIDTH) R.color.live
                    else R.color.on_surface,
                )
            )
        }

        if (debugVisible) {
            findViewById<TextView>(R.id.debugPanel).text = buildString {
                append("backend  ").append(backendLabel).append('\n')
                append("model    ").append(modelStatus).append('\n')
                append("camera   %.1f fps, frame %d".format(fpsEma, frameCount)).append('\n')
                append("infer    %.1f ms".format(inferenceMsEma)).append('\n')
                append("wrist    ").append(wristTrail.size).append(" samples\n")
                append("sent     ").append(client?.observationsSent ?: 0)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        sessionEpoch.incrementAndGet()
        running.set(false)
        client?.disconnect()
        detector?.close()
        poseEstimator?.close()
        yolo26?.close()
        analysisExecutor.shutdown()
    }

    private companion object {
        const val PREFS = "argus_nursing"
        const val KEY_URL = "server_url"
        const val SCORE_THRESHOLD = 0.35
        const val IDLE_INTERVAL_MS = 1_000L

        /** Matches `[scoring] cpr_window_s` on the laptop. */
        const val RATE_WINDOW_S = 8.0

        /** 180/min, the fastest the laptop will look for. */
        const val MIN_PEAK_SEPARATION_S = 60.0 / 180.0

        /** Matches the laptop's `_CPR_MIN_AMPLITUDE`. */
        const val MIN_AMPLITUDE = 0.005

        /** The AHA band, 100-120, expressed as a midpoint for the readout tint. */
        const val TARGET_MID = 110
        const val TARGET_HALF_WIDTH = 10

        const val KP_LEFT_WRIST = 9
        const val KP_RIGHT_WRIST = 10
    }
}
