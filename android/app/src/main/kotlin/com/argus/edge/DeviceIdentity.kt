package com.argus.edge

import android.content.Context
import java.util.UUID

/**
 * This station's stable identity.
 *
 * `trainee_id` is a triage key -- an instructor is dispatched to a *specific*
 * person -- so on a multi-device floor the device half of that id has to be
 * stable across app restarts, or the same trainee reappears under a new id
 * every time a phone is picked up and their triage history resets mid-class.
 *
 * The id is generated once and persisted. It is deliberately not derived from
 * `Build.SERIAL`, the advertising id, or anything else that identifies the
 * handset or its owner: the PC needs to tell stations apart, which is a much
 * weaker requirement than knowing which handset a station is.
 *
 * The generated form satisfies `wire.checkDeviceId` by construction -- 12 hex
 * characters after a `phone` prefix, so no `-` can appear and the device half
 * of a namespaced trainee id stays unambiguous.
 */
object DeviceIdentity {
    private const val PREFS = "argus_edge_identity"
    private const val KEY_DEVICE_ID = "device_id"

    fun deviceId(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs.getString(KEY_DEVICE_ID, null)?.let { return it }

        val generated = generate()
        prefs.edit().putString(KEY_DEVICE_ID, generated).apply()
        return generated
    }

    /** Overwrite the stored identity. Re-pairing a station, not routine use. */
    fun reset(context: Context): String {
        val generated = generate()
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY_DEVICE_ID, generated).apply()
        return generated
    }

    private fun generate(): String {
        val hex = UUID.randomUUID().toString().replace("-", "").take(12)
        return "phone$hex".also { checkDeviceId(it) }
    }
}
