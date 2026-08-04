package com.argus.edge

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/**
 * The station's own screen.
 *
 * This is a capture-and-compute node, not a dashboard: the ranked help queue
 * lives on the PC, because what an instructor needs is one ordered list for the
 * floor rather than one per phone. What is shown here is only what the person
 * holding the handset needs in order to trust it -- which station this is,
 * whether frames are actually arriving, and whether the NPU came up.
 *
 * That last line is the point. If the detector cannot reach the Hexagon NPU the
 * status reads the failure rather than showing a calm preview that implies the
 * station is working, and no envelopes are sent at all -- a station that is
 * silent is one the PC will flag as stale, which is the honest signal. Sending
 * empty ranks from a phone whose model never loaded would tell the aggregator
 * "nobody here needs help", which is a lie it has no way to detect.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusView: TextView
    private lateinit var previewView: PreviewView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val framesSeen = AtomicLong(0)

    private lateinit var deviceId: String
    private var detector: PersonDetector? = null
    private var npuStatus: String = "not initialised"

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else render("camera permission denied") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        statusView = findViewById(R.id.status)
        previewView = findViewById(R.id.preview)

        deviceId = DeviceIdentity.deviceId(this)
        npuStatus = initialiseDetector()
        render()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Bring up the NPU detector, or record exactly why it could not come up.
     *
     * The exception is caught here and *only* here, to put it on screen. It is
     * not caught to substitute a working-looking fallback: [detector] stays
     * null, so nothing downstream can mistake this station for a healthy one.
     */
    private fun initialiseDetector(): String {
        val models = File(filesDir, "models")
        val artifact = File(models, "yolox_sm8750.onnx")
        val backend = File(applicationInfo.nativeLibraryDir, "libQnnHtp.so")
        return try {
            detector = QnnDetector(artifact, backend)
            "NPU ready"
        } catch (e: NpuUnavailableException) {
            "NPU UNAVAILABLE — ${e.message?.take(220)}"
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                // 640 square matches the detector's letterbox canvas, so the
                // capture path does not resize twice.
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 640),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                            )
                        )
                        .build()
                )
                // Drop frames rather than queue them: a station that falls behind
                // should report the present late, not the past on time.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::onFrame) }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            render()
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * One captured frame.
     *
     * The frame is a local and stays one. It is closed at the end of this
     * method and never stored on a field, handed to a sink, or serialised --
     * the same property `tests/test_privacy.py` asserts of the PC's tracks.
     */
    private fun onFrame(image: ImageProxy) {
        try {
            framesSeen.incrementAndGet()
            // Inference, tracking, and scoring attach here. Which of them run on
            // this device is still an open architecture decision; the wire
            // contract in Wire.kt is deliberately the same either way, since it
            // carries scored records only.
        } finally {
            image.close()
        }
        if (framesSeen.get() % 15L == 0L) runOnUiThread { render() }
    }

    private fun render(extra: String? = null) {
        statusView.text = buildString {
            append("station   ").append(deviceId).append('\n')
            append("frames    ").append(framesSeen.get()).append('\n')
            append("detector  ").append(npuStatus)
            extra?.let { append('\n').append(it) }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        analysisExecutor.shutdown()
    }
}
