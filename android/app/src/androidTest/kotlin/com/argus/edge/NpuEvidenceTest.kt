package com.argus.edge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.FloatBuffer

/**
 * Evidence that inference is on the NPU, not merely that a session was created.
 *
 * "The session opened with CPU fallback disabled" is good evidence but it is
 * indirect, and this project has already been burned once by a confident
 * inference drawn from an indirect measurement. So this gathers two independent
 * signals that do not depend on each other:
 *
 *  1. **Node placement.** ORT logs which execution provider each node was
 *     assigned to. Verbose logging makes that visible in logcat, so the claim
 *     can be read off the runtime's own account rather than deduced.
 *  2. **Timing.** The QNN and CPU providers run the identical graph on identical
 *     input. Different hardware should not produce identical throughput. This
 *     is a sanity signal, not a benchmark: the model is a single tiny Conv, so
 *     the numbers are dominated by dispatch overhead and are not a statement
 *     about YOLO-X.
 *
 * Neither alone is conclusive. Together, with the fallback-disabled session
 * creation, they are.
 */
@RunWith(AndroidJUnit4::class)
class NpuEvidenceTest {

    private val backend: File
        get() {
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            return File(ctx.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
        }

    private fun modelBytes(): ByteArray =
        InstrumentationRegistry.getInstrumentation().context.assets
            .open("qnn_smoke_w8a8.onnx").use { it.readBytes() }

    private fun timeRuns(options: OrtSession.SessionOptions, runs: Int): Pair<Double, Any?> {
        val env = OrtEnvironment.getEnvironment()
        env.createSession(modelBytes(), options).use { session ->
            val input = FloatBuffer.allocate(1 * 3 * 32 * 32).apply {
                repeat(capacity()) { put(0.5f) }; rewind()
            }
            OnnxTensor.createTensor(env, input, longArrayOf(1, 3, 32, 32)).use { tensor ->
                val feed = mapOf("x" to tensor)
                repeat(10) { session.run(feed).close() }          // warm up
                var last: Any? = null
                val started = System.nanoTime()
                repeat(runs) { session.run(feed).use { last = it[0].value } }
                val micros = (System.nanoTime() - started) / 1000.0 / runs
                return micros to last
            }
        }
    }

    @Test
    fun qnnAndCpuBothProduceOutputAndDifferInThroughput() {
        val runs = 200

        val qnnOptions = OrtSession.SessionOptions().apply {
            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
            setSessionLogVerbosityLevel(1)
            addConfigEntry("session.disable_cpu_ep_fallback", "1")
            addQnn(mapOf("backend_path" to backend.absolutePath, "htp_performance_mode" to "burst"))
        }
        val (qnnMicros, qnnOut) = timeRuns(qnnOptions, runs)

        val cpuOptions = OrtSession.SessionOptions()
        val (cpuMicros, cpuOut) = timeRuns(cpuOptions, runs)

        val report = buildString {
            append("NPU EVIDENCE\n")
            append("  QNN/HTP : %.1f us/inference\n".format(qnnMicros))
            append("  CPU     : %.1f us/inference\n".format(cpuMicros))
            append("  ratio   : %.2fx\n".format(cpuMicros / qnnMicros))
            append("  QNN produced output: ").append(qnnOut != null).append('\n')
            append("  CPU produced output: ").append(cpuOut != null)
        }
        println(report)

        assertTrue("QNN produced no output", qnnOut != null)
        assertTrue("CPU produced no output", cpuOut != null)
        // Identical timing would mean both paths are the same hardware.
        assertTrue(
            "QNN and CPU timings are indistinguishable, which is what running on " +
                "the same hardware twice looks like:\n$report",
            kotlin.math.abs(qnnMicros - cpuMicros) / maxOf(qnnMicros, cpuMicros) > 0.10,
        )
    }
}
