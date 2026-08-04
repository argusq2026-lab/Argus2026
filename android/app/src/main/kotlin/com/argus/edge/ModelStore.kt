package com.argus.edge

import android.content.Context
import android.net.Uri
import java.io.File

/**
 * Local, pluggable model management.
 *
 * Models live in `filesDir/models/` as a triple: `<name>.onnx` (graph),
 * `<name>.data` (external weights, if the export produced one), and
 * `<name>.json` (the contract sidecar — required, see [ModelSidecar]).
 *
 * Two ways in, same directory:
 *  - the in-app importer ([importFromUris]) — the user multi-selects the files
 *    in the system picker; nothing is fetched from a network, ever;
 *  - `adb push` per android/README.md, for development.
 *
 * The active model is a preference; a model whose sidecar is missing is listed
 * as unusable rather than hidden, so a half-staged model is a visible state,
 * not a mystery.
 */
class ModelStore(private val context: Context) {

    private val prefs = context.getSharedPreferences("argus_edge_models", Context.MODE_PRIVATE)
    val modelsDir: File = File(context.filesDir, "models").apply { mkdirs() }

    data class Entry(val name: String, val onnx: File, val sidecar: File, val data: File?) {
        val usable: Boolean get() = onnx.isFile && sidecar.isFile
    }

    fun list(): List<Entry> =
        (modelsDir.listFiles() ?: emptyArray())
            .filter { it.extension == "onnx" }
            .map { onnx ->
                val stem = onnx.nameWithoutExtension
                Entry(
                    name = stem,
                    onnx = onnx,
                    sidecar = File(modelsDir, "$stem.json"),
                    data = File(modelsDir, "$stem.data").takeIf { it.isFile },
                )
            }
            .sortedBy { it.name }

    var activeName: String?
        get() = prefs.getString("active", null)
        set(value) = prefs.edit().putString("active", value).apply()

    /** The active entry, falling back to the only usable model if unset. */
    fun active(): Entry? {
        val all = list()
        val chosen = activeName?.let { name -> all.firstOrNull { it.name == name } }
        return chosen ?: all.firstOrNull { it.usable }?.also { activeName = it.name }
    }

    /**
     * Copy user-picked files in. Returns a human-readable summary line.
     * Files keep their display names; the .onnx basename becomes the entry.
     */
    fun importFromUris(uris: List<Uri>): String {
        var imported = 0
        var newActive: String? = null
        for (uri in uris) {
            val name = displayName(uri) ?: continue
            if (!name.endsWith(".onnx") && !name.endsWith(".data") && !name.endsWith(".json")) continue
            context.contentResolver.openInputStream(uri)?.use { input ->
                File(modelsDir, name).outputStream().use { input.copyTo(it) }
                imported += 1
                if (name.endsWith(".onnx")) newActive = name.removeSuffix(".onnx")
            }
        }
        newActive?.let { activeName = it }
        return "imported $imported file(s)" + (newActive?.let { ", active: $it" } ?: "")
    }

    private fun displayName(uri: Uri): String? =
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && cursor.moveToFirst()) cursor.getString(idx) else null
        }
}
