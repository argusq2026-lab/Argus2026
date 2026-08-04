package com.argus.edge

import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Can this app actually open the DSP?
 *
 * Everything upstream of this is inference from file permissions, and file
 * permissions are the wrong thing to reason from: `/dev/fastrpc-cdsp` is an
 * ioctl-only character device, so `cat` failing on it says nothing about
 * whether `open(2)` would succeed, and a DAC mode bit says nothing about what
 * SELinux and supplementary groups actually permit.
 *
 * `android.system.Os.open` surfaces the real errno, which is the only thing
 * that settles it:
 *
 *  - `EACCES` / `EPERM` — the app genuinely cannot reach the DSP, and QNN on a
 *    retail handset is a hardware or signing problem.
 *  - success — the app *can* reach the DSP, and the QNN failure is a
 *    configuration problem after all, somewhere above the device node.
 *
 * Those two conclusions lead to completely different plans, which is why this
 * is measured rather than argued.
 */
@RunWith(AndroidJUnit4::class)
class DspAccessTest {

    private val nodes = listOf(
        "/dev/fastrpc-cdsp",
        "/dev/fastrpc-adsp",
        "/dev/fastrpc-cdsp-secure",
    )

    @Test
    fun reportWhetherThisAppCanOpenTheDspNodes() {
        val report = StringBuilder("DSP node access from untrusted_app:")
        for (path in nodes) {
            val exists = File(path).exists()
            if (!exists) {
                report.append("\n  ABSENT  ").append(path)
                continue
            }
            val outcome = try {
                val fd = Os.open(path, OsConstants.O_RDWR, 0)
                Os.close(fd)
                "OPENED  (O_RDWR succeeded)"
            } catch (e: ErrnoException) {
                val name = OsConstants.errnoName(e.errno) ?: e.errno.toString()
                "DENIED  $name — ${e.message}"
            }
            report.append("\n  ").append(outcome).append("  ").append(path)
        }

        // Whether the vendor's own client library can be loaded at all is a
        // second, independent signal: it lives in /vendor/lib64 and is only on
        // an app's linker path if the vendor made it public.
        val libs = listOf("libcdsprpc.so", "libadsprpc.so")
        for (lib in libs) {
            val outcome = try {
                System.loadLibrary(lib.removePrefix("lib").removeSuffix(".so"))
                "LOADED"
            } catch (t: Throwable) {
                "UNAVAILABLE — ${t.message?.take(100)}"
            }
            report.append("\n  ").append(outcome).append("  ").append(lib)
        }

        println(report)
        // Reporting test: it records the platform's answer rather than asserting
        // one, because either answer is a legitimate state of the world and the
        // point is to know which.
    }
}
