package com.argus.edge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * BlazePose landmarks after YOLO: the second NPU stage, per the architecture
 * on main — the phone supplies bbox + COCO-17 keypoints, the laptop scores.
 *
 * The detector stage of MediaPipe's pipeline is deliberately absent. It only
 * ever contributed a centre and scale for the landmark ROI (the PC pipeline
 * had already dropped ROI rotation — `d3bd15e:src/argus/vision/pose.py` states
 * it as a design simplification), and YOLO's person box provides both. One
 * extra NPU call per frame, not two.
 *
 * ## The decode contract, pinned to source
 *
 * The exported network ends with `landmarks.view(batch, 31, 4) / 256`
 * (zmurez MediaPipePyTorch `blazepose_landmark.py`, cropped to 25 by the
 * qai-hub-models wrapper), so *every* channel arrives divided by 256:
 *
 *  - x, y: raw values are crop pixels in [0, 256], so post-division they are
 *    normalized [0, 1] of the ROI — mapped to frame pixels here.
 *  - visibility: raw value is a *logit*, so post-division the wire value is
 *    `logit / 256`. Confidence is therefore `sigmoid(v * 256)`. The old w8a8
 *    artifact shipped visibility already-activated
 *    (`landmark_visibility_is_logit = false` in the old config) — the same
 *    number means different things per export, which is exactly why this is
 *    written down next to the arithmetic instead of discovered in production.
 *  - z is dropped, as the PC decode dropped it: scoring is in the image plane.
 */
private const val LANDMARK_COUNT = 25
private const val LANDMARK_STRIDE = 4

/** BlazePose landmark indices (25-point upper-body export). */
private val BLAZEPOSE_NAMES = listOf(
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
)

/**
 * COCO index -> BlazePose index, -1 where the 25-point export has no source
 * landmark. Ported verbatim from `d3bd15e:src/argus/vision/keypoints.py`;
 * both layouts label left/right from the subject's perspective.
 */
val COCO_FROM_BLAZEPOSE: IntArray = intArrayOf(
    BLAZEPOSE_NAMES.indexOf("nose"),            // 0
    BLAZEPOSE_NAMES.indexOf("left_eye"),        // 1
    BLAZEPOSE_NAMES.indexOf("right_eye"),       // 2
    BLAZEPOSE_NAMES.indexOf("left_ear"),        // 3
    BLAZEPOSE_NAMES.indexOf("right_ear"),       // 4
    BLAZEPOSE_NAMES.indexOf("left_shoulder"),   // 5
    BLAZEPOSE_NAMES.indexOf("right_shoulder"),  // 6
    BLAZEPOSE_NAMES.indexOf("left_elbow"),      // 7
    BLAZEPOSE_NAMES.indexOf("right_elbow"),     // 8
    BLAZEPOSE_NAMES.indexOf("left_wrist"),      // 9
    BLAZEPOSE_NAMES.indexOf("right_wrist"),     // 10
    BLAZEPOSE_NAMES.indexOf("left_hip"),        // 11
    BLAZEPOSE_NAMES.indexOf("right_hip"),       // 12
    -1, -1, -1, -1,                             // 13-16 knees/ankles: no source
)

/** COCO-17 keypoints in frame pixels, plus per-keypoint confidence. */
data class PoseResult(
    val keypointsXy: FloatArray,   // 17 * 2, frame pixels; (0,0) where unmapped
    val keypointsConf: FloatArray, // 17
    val poseScore: Float,
)

/** An axis-aligned square ROI around a detection, clamped to sane bounds. */
data class PoseRoi(val x0: Float, val y0: Float, val side: Float)

fun squareRoiFor(det: Detection, scale: Float = 1.25f): PoseRoi {
    val cx = (det.x0 + det.x1) / 2f
    val cy = (det.y0 + det.y1) / 2f
    val side = maxOf(det.x1 - det.x0, det.y1 - det.y0) * scale
    return PoseRoi(cx - side / 2f, cy - side / 2f, maxOf(side, 1f))
}

/**
 * Pure decode: raw (25*4) landmark floats + the ROI they were computed in →
 * COCO-17 frame-pixel keypoints and confidences. Host-testable without ORT.
 */
fun decodePose(landmarks: FloatArray, roi: PoseRoi, poseScore: Float): PoseResult {
    require(landmarks.size == LANDMARK_COUNT * LANDMARK_STRIDE) {
        "expected ${LANDMARK_COUNT * LANDMARK_STRIDE} floats, got ${landmarks.size}"
    }
    val xy = FloatArray(17 * 2)
    val conf = FloatArray(17)
    for (coco in 0 until 17) {
        val src = COCO_FROM_BLAZEPOSE[coco]
        if (src < 0) continue // no source landmark: (0,0) at confidence 0
        val base = src * LANDMARK_STRIDE
        // x,y are normalized [0,1] of the ROI after the network's /256.
        xy[coco * 2] = roi.x0 + landmarks[base] * roi.side
        xy[coco * 2 + 1] = roi.y0 + landmarks[base + 1] * roi.side
        // visibility arrives as logit/256; activate it.
        conf[coco] = sigmoid(landmarks[base + 3] * 256f)
    }
    return PoseResult(xy, conf, poseScore)
}

fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

/**
 * The landmark model on the NPU. Float32 artifact, fp16 on the HTP —
 * deliberately unquantized until real calibration footage exists
 * (docs/VALIDATION.md §1 is what happens otherwise).
 */
class PoseEstimator(modelFile: File, backendFile: File) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputSize: Int

    init {
        if (!modelFile.isFile) throw NpuUnavailableException("pose model not found at $modelFile")
        session = try {
            val options = OrtSession.SessionOptions()
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            options.addQnn(
                mapOf(
                    "backend_path" to backendFile.absolutePath,
                    "htp_performance_mode" to "burst",
                    "enable_htp_fp16_precision" to "1",
                )
            )
            env.createSession(modelFile.absolutePath, options)
        } catch (t: Throwable) {
            throw NpuUnavailableException(
                "pose model could not be placed on the NPU: ${t.message}. Not falling back to CPU.", t,
            )
        }

        // Contract from the artifact, checked at open — names, shapes, count.
        if (session.inputNames.toSet() != setOf("image") ||
            session.outputNames.toSet() != setOf("scores", "landmarks")
        ) {
            session.close()
            throw NpuUnavailableException(
                "pose model contract mismatch: inputs ${session.inputNames}, outputs ${session.outputNames}"
            )
        }
        val shape = (session.inputInfo.getValue("image").info as TensorInfo).shape
        if (shape.size != 4 || shape[1] != 3L || shape[2] != shape[3]) {
            session.close()
            throw NpuUnavailableException("pose model input is ${shape.toList()}, expected (1,3,S,S)")
        }
        inputSize = shape[2].toInt()
    }

    /** Landmark the person inside `det`'s ROI. Returns null below `minScore`. */
    fun estimate(upright: Bitmap, det: Detection, minScore: Float = 0.5f): PoseResult? {
        val roi = squareRoiFor(det)
        val tensor = cropToFloatNchw(upright, roi, inputSize)
        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(tensor), longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        ).use { input ->
            session.run(mapOf("image" to input)).use { result ->
                val score = (result.get("scores").get() as OnnxTensor).floatBuffer.get(0)
                if (score < minScore) return null
                val lm = FloatArray(LANDMARK_COUNT * LANDMARK_STRIDE)
                (result.get("landmarks").get() as OnnxTensor).floatBuffer.get(lm)
                return decodePose(lm, roi, score)
            }
        }
    }

    fun close() = session.close()
}

/** Crop `roi` (zero-padded outside the frame), resize to S², RGB floats [0,1] NCHW. */
fun cropToFloatNchw(src: Bitmap, roi: PoseRoi, size: Int): FloatArray {
    val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(out) // starts transparent black == zero padding
    val scale = size / roi.side
    val srcRect = Rect(0, 0, src.width, src.height)
    val dst = android.graphics.RectF(
        -roi.x0 * scale, -roi.y0 * scale,
        (src.width - roi.x0) * scale, (src.height - roi.y0) * scale,
    )
    canvas.drawBitmap(src, srcRect, dst, Paint(Paint.FILTER_BITMAP_FLAG))

    val pixels = IntArray(size * size)
    out.getPixels(pixels, 0, size, 0, 0, size, size)
    out.recycle()
    val plane = size * size
    val tensor = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val p = pixels[i]
        tensor[i] = ((p shr 16) and 0xFF) / 255f
        tensor[plane + i] = ((p shr 8) and 0xFF) / 255f
        tensor[2 * plane + i] = (p and 0xFF) / 255f
    }
    return tensor
}
