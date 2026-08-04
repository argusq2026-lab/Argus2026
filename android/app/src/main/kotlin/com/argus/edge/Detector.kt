package com.argus.edge

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File

/** One person detection, in source-frame pixels. */
data class Detection(val x0: Float, val y0: Float, val x1: Float, val y1: Float, val score: Float)

interface PersonDetector {
    /** @param nchw a (1, 3, S, S) uint8 letterboxed tensor, RGB. */
    fun detect(nchw: ByteArray, size: Int, letterbox: LetterboxInfo): List<Detection>
    fun close()
}

/**
 * Raised when the NPU cannot be reached. Never caught internally.
 *
 * The PC engine refuses rather than falling back to CPU, because CPU-speed
 * results that look like NPU results are worse than an error -- see
 * `argus.engines.ort_qnn` and `configs/argus.default.toml`'s
 * `allow_cpu_fallback = false`. The phone inherits that rule verbatim: a
 * station that quietly drops to CPU would report plausible, late, degraded
 * scores that the PC has no way to distinguish from healthy ones.
 */
class NpuUnavailableException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

/**
 * YOLO-X on the Hexagon NPU, via ONNX Runtime's QNN execution provider.
 *
 * The QNN backend libraries are **not** something an operator has to stage.
 * `onnxruntime-android-qnn` depends on `com.qualcomm.qti:qnn-runtime`, which
 * Qualcomm publishes to Maven Central, so `libQnnHtp.so`, `libQnnSystem.so` and
 * the per-Hexagon-version skels — including `libQnnHtpV79Skel.so`, the one the
 * Snapdragon 8 Elite needs — are packaged into the APK by Gradle and unpacked
 * into `nativeLibraryDir` at install time. No Qualcomm account, no QAIRT SDK
 * download.
 *
 * What *does* have to be staged is the model: a w8a8 detector artifact compiled
 * for this SoC. The repository's existing artifacts target `sc8380xp` /
 * Hexagon v73 (Snapdragon X Elite); the Snapdragon 8 Elite is `sm8750` /
 * Hexagon **v79**. That is the same class of failure `docs/VALIDATION.md` §4
 * records for QAIRT version skew, one step worse because the chip family
 * differs too.
 *
 * Note the bundled runtime is QAIRT 2.33.0. A QNN *context binary* is tied to
 * its producing runtime, so compiling one against a different QAIRT would
 * reproduce exactly the skew §4 describes. Targeting plain QDQ ONNX instead
 * lets the execution provider compile the graph for whatever HTP it finds at
 * session init, which sidesteps the pinning entirely — at the cost of a slower
 * first load.
 *
 * When the artifact is missing, or the graph cannot be placed on the NPU, this
 * throws [NpuUnavailableException]. It does not substitute a CPU session, and
 * it does not return empty detections — a station reporting "nobody here"
 * because its model failed to load is the exact silent degradation the design
 * forbids.
 */
class QnnDetector(
    modelPath: File,
    backendPath: File,
) : PersonDetector {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        if (!modelPath.isFile) {
            throw NpuUnavailableException(
                "detector artifact not found at $modelPath -- stage a build compiled " +
                    "for this SoC (Snapdragon 8 Elite / sm8750, Hexagon v79); " +
                    "the sc8380xp artifacts will not load here"
            )
        }
        if (!backendPath.isFile) {
            // Should be unreachable on a correctly packaged build: Gradle pulls
            // com.qualcomm.qti:qnn-runtime in transitively and the installer
            // unpacks it here. If it is missing, the APK was assembled without
            // that dependency, which is a build defect worth naming as one
            // rather than silently degrading to CPU.
            throw NpuUnavailableException(
                "QNN backend not found at $backendPath -- expected it to be " +
                    "packaged from com.qualcomm.qti:qnn-runtime. Check that " +
                    "abiFilters still includes arm64-v8a and that the " +
                    "onnxruntime-android-qnn dependency resolved."
            )
        }

        session = try {
            val options = OrtSession.SessionOptions()
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            options.addQnn(mapOf("backend_path" to backendPath.absolutePath))
            env.createSession(modelPath.absolutePath, options)
        } catch (t: Throwable) {
            throw NpuUnavailableException(
                "the QNN execution provider could not place this graph on the NPU: " +
                    "${t.message}. Not falling back to CPU.",
                t,
            )
        }
    }

    override fun detect(nchw: ByteArray, size: Int, letterbox: LetterboxInfo): List<Detection> {
        // The decode is deliberately not written yet: the real tensor contract
        // must be read from the artifact's metadata.json, exactly as
        // argus.engines.metadata does on the PC, and no SM8750 artifact exists
        // to read one from. Guessing shapes here is precisely the mistake
        // ARCHITECTURE.md records the prototype making -- three wrong contracts
        // that were never caught because the path had never executed.
        throw NotImplementedError(
            "detector post-processing pending a real SM8750 artifact to read " +
                "metadata.json from -- see android/README.md"
        )
    }

    override fun close() {
        session.close()
    }
}
