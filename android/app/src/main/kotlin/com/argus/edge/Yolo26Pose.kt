package com.argus.edge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.FloatBuffer

/**
 * Single-stage detection + COCO-17 pose, as an alternative to YOLO-X + BlazePose.
 *
 * **Prototype, behind a flag.** Selected only when its artifact is staged; the
 * two-model path stays intact and is used otherwise, so nothing regresses while
 * this is being evaluated. See android/README.md for the comparison that
 * motivated it.
 *
 * Why it exists: the 25-point BlazePose export is upper-body only, and four of
 * the seven `form_reason_codes` in the protocol vocabulary — `insufficient_depth`,
 * `knee_valgus`, `heels_rising`, and partly `incomplete_lockout` — need knees or
 * ankles it structurally cannot produce. That is a coverage problem no amount of
 * tuning fixes.
 *
 * Why it is simpler: this model emits everything in one pass, in one coordinate
 * space, already activated. Compared with the path it replaces there is no ROI
 * to derive (the bug that made pose look broken), no crop, no 25-to-COCO remap,
 * no visibility logit convention to get right, no quantization parameters, and
 * one NPU dispatch instead of two — which matters given the ~500 µs fixed
 * dispatch cost measured on this device.
 *
 * ## Licence
 *
 * The weights are Ultralytics YOLO26, **AGPL-3.0**, while this repository is
 * MIT. Shipping this commercially requires either releasing the application
 * under AGPL or obtaining an Ultralytics commercial licence. That is a business
 * decision, not a technical one, and it is why this is a flag rather than a
 * replacement. `rtmpose_body2d` is the Apache-2.0 alternative if the licence
 * is unacceptable.
 *
 * ## Contract
 *
 * Verified against the live session at open, the same discipline `QnnDetector`
 * applies. Unlike the w8a8 detector there are no quantization parameters to
 * read, so there is no sidecar: the contract is entirely I/O names and shapes.
 *
 *  - `image`      `(1, 3, 640, 640)` float32, RGB, **[0, 1]** — note the
 *                 normalisation; the w8a8 detector takes raw uint8 instead.
 *  - `boxes`      `(1, 8400, 4)` xyxy in letterboxed-canvas pixels
 *  - `scores`     `(1, 8400)` already in [0, 1]
 *  - `keypoints`  `(1, 8400, 17, 3)` — x, y in canvas pixels, confidence
 *                 already activated. COCO-17 order, so it is exactly what
 *                 `docs/PROTOCOL.md` asks for with no remap.
 */
class Yolo26PoseEstimator(
    modelFile: File,
    backendFile: File,
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val inputSize: Int

    /** Carried over from the detector config; the same knob the UI slider drives. */
    var nmsIouThreshold: Float = 0.45f
    var maxDetections: Int = 64

    init {
        if (!modelFile.isFile) {
            throw NpuUnavailableException("yolo26-pose model not found at $modelFile")
        }
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
                "yolo26-pose could not be placed on the NPU: ${t.message}. Not falling back to CPU.",
                t,
            )
        }

        if (session.inputNames.toSet() != setOf("image") ||
            session.outputNames.toSet() != setOf("boxes", "scores", "keypoints")
        ) {
            session.close()
            throw NpuUnavailableException(
                "yolo26-pose contract mismatch: inputs ${session.inputNames}, " +
                    "outputs ${session.outputNames}"
            )
        }
        val shape = (session.inputInfo.getValue("image").info as TensorInfo).shape
        if (shape.size != 4 || shape[1] != 3L || shape[2] != shape[3]) {
            session.close()
            throw NpuUnavailableException("yolo26-pose input is ${shape.toList()}, expected (1,3,S,S)")
        }
        inputSize = shape[2].toInt()
    }

    /** One pass: detections with their poses, in source-frame pixels. */
    fun detectAndPose(
        nchwFloat: FloatArray,
        letterbox: LetterboxInfo,
        srcWidth: Int,
        srcHeight: Int,
        scoreThreshold: Float,
    ): List<Pair<Detection, PoseResult>> {
        val n = 3 * inputSize * inputSize
        require(nchwFloat.size == n) { "tensor is ${nchwFloat.size} floats, contract wants $n" }
        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(nchwFloat),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        ).use { input ->
            session.run(mapOf("image" to input)).use { result ->
                fun floats(name: String): FloatArray {
                    val buf = (result.get(name).get() as OnnxTensor).floatBuffer
                    return FloatArray(buf.remaining()).also { buf.get(it) }
                }
                return decodeYolo26Pose(
                    floats("boxes"), floats("scores"), floats("keypoints"),
                    scoreThreshold, nmsIouThreshold, maxDetections,
                    letterbox, srcWidth, srcHeight,
                )
            }
        }
    }

    fun close() = session.close()
}

/**
 * Pure decode: threshold, greedy NMS, letterbox undo, clip. Host-testable.
 *
 * Deliberately mirrors [decodeDetections]'s structure so the two paths stay
 * comparable, minus everything this model makes unnecessary: no dequantize, no
 * class filter (the model is person-only), no separate pose stage.
 */
fun decodeYolo26Pose(
    boxes: FloatArray,      // 8400 * 4
    scores: FloatArray,     // 8400
    keypoints: FloatArray,  // 8400 * 17 * 3
    scoreThreshold: Float,
    nmsIouThreshold: Float,
    maxDetections: Int,
    letterbox: LetterboxInfo,
    srcWidth: Int,
    srcHeight: Int,
): List<Pair<Detection, PoseResult>> {
    val anchors = scores.size
    data class Candidate(val idx: Int, val score: Float, val box: FloatArray)

    val candidates = ArrayList<Candidate>()
    for (i in 0 until anchors) {
        if (scores[i] < scoreThreshold) continue
        candidates.add(
            Candidate(i, scores[i], floatArrayOf(
                boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3],
            ))
        )
    }
    if (candidates.isEmpty()) return emptyList()

    val order = candidates.sortedWith(compareByDescending<Candidate> { it.score }.thenBy { it.idx })
    val kept = ArrayList<Candidate>()
    for (c in order) {
        if (kept.size >= maxDetections) break
        if (kept.none { iouOf(it.box, c.box) > nmsIouThreshold }) kept.add(c)
    }

    val out = ArrayList<Pair<Detection, PoseResult>>(kept.size)
    for (c in kept) {
        val m = undoLetterbox(c.box[0], c.box[1], c.box[2], c.box[3], letterbox)
        val x0 = m[0].coerceIn(0f, srcWidth.toFloat())
        val y0 = m[1].coerceIn(0f, srcHeight.toFloat())
        val x1 = m[2].coerceIn(0f, srcWidth.toFloat())
        val y1 = m[3].coerceIn(0f, srcHeight.toFloat())
        if (x1 - x0 < 1f || y1 - y0 < 1f) continue

        val xy = FloatArray(17 * 2)
        val conf = FloatArray(17)
        val base = c.idx * 17 * 3
        for (k in 0 until 17) {
            val kx = keypoints[base + k * 3]
            val ky = keypoints[base + k * 3 + 1]
            // Keypoints share the boxes' canvas space, so the same inverse
            // letterbox applies -- one transform, not two.
            val p = undoLetterbox(kx, ky, kx, ky, letterbox)
            xy[k * 2] = p[0]
            xy[k * 2 + 1] = p[1]
            conf[k] = keypoints[base + k * 3 + 2]   // already activated
        }
        out.add(Detection(x0, y0, x1, y1, c.score) to PoseResult(xy, conf, c.score))
    }
    return out
}

private fun iouOf(a: FloatArray, b: FloatArray): Float {
    val xx0 = maxOf(a[0], b[0]); val yy0 = maxOf(a[1], b[1])
    val xx1 = minOf(a[2], b[2]); val yy1 = minOf(a[3], b[3])
    val inter = maxOf(0f, xx1 - xx0) * maxOf(0f, yy1 - yy0)
    val areaA = maxOf(a[2] - a[0], 0f) * maxOf(a[3] - a[1], 0f)
    val areaB = maxOf(b[2] - b[0], 0f) * maxOf(b[3] - b[1], 0f)
    return inter / maxOf(areaA + areaB - inter, 1e-9f)
}

/** ARGB letterboxed bitmap -> NCHW RGB float32 in [0, 1], this model's input. */
fun toNchwRgbFloats(bitmap: android.graphics.Bitmap): FloatArray {
    val size = bitmap.width
    require(bitmap.height == size) { "expected a square letterboxed bitmap" }
    val pixels = IntArray(size * size)
    bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
    val plane = size * size
    val out = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val p = pixels[i]
        out[i] = ((p shr 16) and 0xFF) / 255f
        out[plane + i] = ((p shr 8) and 0xFF) / 255f
        out[2 * plane + i] = (p and 0xFF) / 255f
    }
    return out
}
