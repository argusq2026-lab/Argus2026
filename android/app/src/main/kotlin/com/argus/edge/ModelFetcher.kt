package com.argus.edge

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * Pull the model weights from the instructor's own laptop.
 *
 * The weights are deliberately not in the APK (licensing —
 * `android/models.json`), and asking whoever sets up a floor to cable every
 * phone was the worst step of first-time setup. The laptop the phone already
 * connects to serves them read-only at `GET /models/` on the ingest port —
 * the same host and port as the WebSocket URL in the Server dialog, so there
 * is nothing new to configure and nothing new to trust: this phone was
 * already sending that laptop its observations.
 *
 * Every download is verified against the sha256 the server's manifest
 * declares before it is moved into the model store. That catches torn
 * transfers, not a hostile server — a station's trust in its laptop is
 * established at Connect, not here.
 */
object ModelFetcher {

    private const val TAG = "ArgusModelFetcher"

    private val http = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        // Weight files are tens of MB; the default 10 s read timeout would
        // turn a slow AP into a spurious failure.
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    data class Progress(val file: String, val index: Int, val total: Int)

    /** `ws://host:port` (or wss) → `http://host:port` — same listener. */
    fun baseUrlFromWs(wsUrl: String): String? {
        val trimmed = wsUrl.trim()
        return when {
            trimmed.startsWith("ws://") -> "http" + trimmed.removePrefix("ws")
            trimmed.startsWith("wss://") -> "https" + trimmed.removePrefix("wss")
            else -> null
        }
    }

    /**
     * Download everything the server offers into [targetDir]. Blocking —
     * call from a background executor. Returns a human-readable summary.
     *
     * Files land as `.part` and are renamed only after the hash matches, so
     * a failed download can never masquerade as a staged model — the exact
     * property the model store's "refuses a model without a sidecar" posture
     * expects from anything that writes into its directory.
     */
    fun fetchAll(
        wsUrl: String,
        targetDir: File,
        onProgress: (Progress) -> Unit = {},
    ): String {
        val base = baseUrlFromWs(wsUrl)
            ?: return "server address is not a ws:// URL"

        val manifest = try {
            http.newCall(Request.Builder().url("$base/models/manifest.json").build())
                .execute().use { response ->
                    if (!response.isSuccessful) return "server answered ${response.code} for the manifest"
                    JSONObject(response.body!!.string())
                }
        } catch (e: Exception) {
            return "could not reach the server: ${e.message}"
        }

        val files = manifest.getJSONArray("files")
        if (files.length() == 0) {
            return "the server has no models to offer — run `argus fetch-models` on the laptop first"
        }

        targetDir.mkdirs()
        var fetched = 0
        for (i in 0 until files.length()) {
            val entry = files.getJSONObject(i)
            val name = entry.getString("name")
            val expected = entry.getString("sha256")
            onProgress(Progress(name, i + 1, files.length()))

            val final = File(targetDir, name)
            if (final.isFile && sha256(final) == expected) {
                Log.i(TAG, "$name already present and verified")
                continue
            }
            val part = File(targetDir, "$name.part")
            try {
                http.newCall(Request.Builder().url("$base/models/$name").build())
                    .execute().use { response ->
                        if (!response.isSuccessful) {
                            return "server answered ${response.code} for $name"
                        }
                        part.outputStream().use { out ->
                            response.body!!.byteStream().copyTo(out, 1 shl 20)
                        }
                    }
                val actual = sha256(part)
                if (actual != expected) {
                    part.delete()
                    return "$name failed verification (got $actual); nothing was staged"
                }
                if (!part.renameTo(final)) {
                    part.delete()
                    return "could not move $name into the model store"
                }
                fetched += 1
                Log.i(TAG, "fetched and verified $name (${final.length()} bytes)")
            } catch (e: Exception) {
                part.delete()
                return "download of $name failed: ${e.message}"
            }
        }
        return "fetched $fetched file(s), ${files.length()} verified — from $base"
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.joinToString("") { "%02x".format(it) }
    }

    private fun MessageDigest.joinToString(
        separator: String, transform: (Byte) -> String,
    ): String = digest().joinToString(separator, transform = transform)
}
