package com.argus.edge

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer

/** One person detection, in source-frame pixel coordinates. */
data class Detection(val x0: Float, val y0: Float, val x1: Float, val y1: Float, val score: Float)

/**
 * Raised when the NPU cannot be reached or a contract does not hold. Never
 * caught internally: the PC engine refused rather than falling back to CPU
 * (`allow_cpu_fallback = false`), because CPU-speed results that look like NPU
 * results are worse than an error. The phone inherits that rule verbatim.
 */
class NpuUnavailableException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

/**
 * YOLO-X on the Hexagon NPU via ONNX Runtime's QNN execution provider.
 *
 * The two `<uses-native-library>` manifest entries are what make this possible
 * on a stock retail phone — QNN dlopens the vendor cDSP client from the vendor
 * namespace, granted only to apps that declare it. See android/README.md for
 * the full account.
 *
 * Every contract is read from the sidecar and re-verified against the live
 * session at open; the decode itself is [decodeDetections], a pure function
 * mirroring the deleted PC reference (`d3bd15e:src/argus/vision/detect.py` +
 * `nms.py`) and pinned against it by the parity fixture's crafted cases.
 */
class QnnDetector(
    modelFile: File,
    sidecarFile: File,
    backendFile: File,
) {
    val sidecar: ModelSidecar = ModelSidecar.load(sidecarFile)
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val inputSize: Int get() = sidecar.inputSize

    init {
        if (!modelFile.isFile) {
            throw NpuUnavailableException(
                "model not found at $modelFile — stage it with the adb commands in " +
                    "android/README.md or import it from the app"
            )
        }
        if (!backendFile.isFile) {
            throw NpuUnavailableException(
                "QNN backend missing at $backendFile — expected from " +
                    "com.qualcomm.qti:qnn-runtime; check abiFilters and useLegacyPackaging"
            )
        }

        session = try {
            val options = OrtSession.SessionOptions()
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            options.addQnn(
                mapOf(
                    "backend_path" to backendFile.absolutePath,
                    "htp_performance_mode" to "burst",
                )
            )
            env.createSession(modelFile.absolutePath, options)
        } catch (t: Throwable) {
            throw NpuUnavailableException(
                "the QNN execution provider could not place this graph on the NPU: " +
                    "${t.message}. Not falling back to CPU.", t,
            )
        }

        // The live session must agree with the sidecar, tensor by tensor —
        // a re-export that changed a shape fails here, naming the mismatch,
        // not 200 frames into a session with garbage boxes.
        val inputNames = session.inputNames.toSet()
        if (inputNames != setOf(sidecar.inputName)) {
            failContract("inputs are $inputNames, sidecar declares '${sidecar.inputName}'")
        }
        val outputNames = session.outputNames.toSet()
        if (outputNames != setOf("boxes", "scores", "class_idx")) {
            failContract("outputs are $outputNames, expected boxes/scores/class_idx")
        }
        val inputInfo = session.inputInfo.getValue(sidecar.inputName).info
        val shape = (inputInfo as ai.onnxruntime.TensorInfo).shape.map { it.toInt() }
        if (shape != sidecar.inputShape) {
            failContract("input shape $shape != sidecar ${sidecar.inputShape}")
        }
    }

    private fun failContract(detail: String): Nothing {
        session.close()
        throw NpuUnavailableException("model/sidecar contract mismatch: $detail")
    }

    /** Run one letterboxed NCHW tensor; returns detections in source pixels. */
    fun detect(
        nchw: ByteArray,
        letterbox: LetterboxInfo,
        srcWidth: Int,
        srcHeight: Int,
        scoreThreshold: Double = sidecar.scoreThreshold,
    ): List<Detection> {
        val (boxesQ, scoresQ, classQ) = runRaw(nchw)
        return decodeDetections(
            boxesQ, scoresQ, classQ, sidecar, scoreThreshold,
            letterbox, srcWidth, srcHeight,
        )
    }

    /** The raw quantized outputs — exposed for the on-device parity test. */
    fun runRaw(nchw: ByteArray): Triple<ByteArray, ByteArray, ByteArray> {
        val n = sidecar.inputShape.reduce(Int::times)
        require(nchw.size == n) { "tensor is ${nchw.size} bytes, contract wants $n" }
        OnnxTensor.createTensor(
            env, ByteBuffer.wrap(nchw),
            sidecar.inputShape.map { it.toLong() }.toLongArray(),
            OnnxJavaType.UINT8,
        ).use { tensor ->
            session.run(mapOf(sidecar.inputName to tensor)).use { result ->
                fun bytes(name: String): ByteArray {
                    val t = result.get(name).get() as OnnxTensor
                    val buf = t.byteBuffer
                    val out = ByteArray(buf.remaining())
                    buf.get(out)
                    return out
                }
                return Triple(bytes("boxes"), bytes("scores"), bytes("class_idx"))
            }
        }
    }

    fun close() = session.close()
}

/**
 * Pure decode: quantized outputs → person boxes in source pixels.
 *
 * Mirrors the PC reference line for line: dequantize, threshold + person
 * filter, greedy NMS (descending score), cap at maxDetections, undo letterbox,
 * clip to frame, drop degenerate boxes. Pinned by the fixture's crafted cases
 * on both the host JVM and the device.
 */
fun decodeDetections(
    boxesQ: ByteArray,
    scoresQ: ByteArray,
    classQ: ByteArray,
    sidecar: ModelSidecar,
    scoreThreshold: Double,
    letterbox: LetterboxInfo,
    srcWidth: Int,
    srcHeight: Int,
): List<Detection> {
    val anchors = scoresQ.size
    data class Candidate(val idx: Int, val score: Float, val box: FloatArray)

    val candidates = ArrayList<Candidate>()
    for (i in 0 until anchors) {
        val cls = classQ[i].toInt() and 0xFF
        if (cls != sidecar.personClassIndex) continue
        val score = sidecar.scores.dequantize(scoresQ[i].toInt() and 0xFF)
        if (score < scoreThreshold) continue
        val box = FloatArray(4) { k ->
            sidecar.boxes.dequantize(boxesQ[i * 4 + k].toInt() and 0xFF)
        }
        candidates.add(Candidate(i, score, box))
    }
    if (candidates.isEmpty()) return emptyList()

    // Greedy NMS, highest score first; stable order like the reference.
    val order = candidates.sortedWith(compareByDescending<Candidate> { it.score }.thenBy { it.idx })
    val kept = ArrayList<Candidate>()
    for (c in order) {
        if (kept.size >= sidecar.maxDetections) break
        val suppressed = kept.any { k -> iou(k.box, c.box) > sidecar.nmsIouThreshold }
        if (!suppressed) kept.add(c)
    }

    val out = ArrayList<Detection>(kept.size)
    for (c in kept) {
        val m = undoLetterbox(c.box[0], c.box[1], c.box[2], c.box[3], letterbox)
        val x0 = m[0].coerceIn(0f, srcWidth.toFloat())
        val y0 = m[1].coerceIn(0f, srcHeight.toFloat())
        val x1 = m[2].coerceIn(0f, srcWidth.toFloat())
        val y1 = m[3].coerceIn(0f, srcHeight.toFloat())
        if (x1 - x0 < 1f || y1 - y0 < 1f) continue  // degenerate after clipping
        out.add(Detection(x0, y0, x1, y1, c.score))
    }
    return out
}

private fun iou(a: FloatArray, b: FloatArray): Float {
    val xx0 = maxOf(a[0], b[0]); val yy0 = maxOf(a[1], b[1])
    val xx1 = minOf(a[2], b[2]); val yy1 = minOf(a[3], b[3])
    val inter = maxOf(0f, xx1 - xx0) * maxOf(0f, yy1 - yy0)
    val areaA = maxOf(a[2] - a[0], 0f) * maxOf(a[3] - a[1], 0f)
    val areaB = maxOf(b[2] - b[0], 0f) * maxOf(b[3] - b[1], 0f)
    return inter / maxOf(areaA + areaB - inter, 1e-9f)
}
