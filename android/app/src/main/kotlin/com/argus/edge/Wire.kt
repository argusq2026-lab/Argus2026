package com.argus.edge

/**
 * The phone half of the phone -> PC contract.
 *
 * This is the Kotlin mirror of `src/argus/edge/wire.py`, and it is deliberately
 * the *only* type in this app that a transport can accept. The privacy
 * guarantee is the same one the Python side makes, enforced the same way: not
 * by filtering frames out on the way to a socket, but by there being no
 * parameter anywhere on this path that could hold one.
 *
 * Concretely:
 *
 *  - [TriageRecord] has four scalar fields and no constructor that accepts an
 *    image, a bitmap, a byte buffer, or a caption.
 *  - [DeviceEnvelope] holds station metadata *around* records, so the record
 *    itself never grows a field to carry it.
 *  - [EdgeTransport.send] takes exactly one [DeviceEnvelope]. A transport
 *    written against that interface cannot be handed pixels.
 *
 * Encoding is one-directional on purpose. The phone only ever sends; the PC
 * only ever receives. There is no decoder here, so there is no code path by
 * which the phone could be told what to do by whatever it is talking to.
 *
 * The encoder emits the same field set and the same key order as the Python
 * `encode_envelope`, and `WireVectorsTest` checks that against the shared
 * fixture in `tests/data/wire_vectors.json`. Byte-identical output is
 * explicitly *not* the contract -- Kotlin and Python format doubles
 * differently -- so what is asserted is that both sides parse to equal values.
 */

/** Bumped whenever the envelope's shape changes meaning. Must match wire.py. */
const val WIRE_VERSION: Int = 1

/** Device ids are `[A-Za-z0-9_]+`; `-` is reserved as the device/track separator. */
private val DEVICE_ID_RE = Regex("^[A-Za-z0-9_]+$")

class WireFormatException(message: String) : IllegalArgumentException(message)

/**
 * The only thing allowed to leave this device.
 *
 * Four scalar fields, closed set, mirroring `argus.triage.TriageRecord`. There
 * is deliberately no field that could carry a frame, a crop, a caption, or even
 * a bounding box.
 */
data class TriageRecord(
    val traineeId: String,
    val score: Double,
    val reasonCodes: List<String>,
    val ts: Double,
) {
    init {
        require(traineeId.isNotEmpty()) { "traineeId must be non-empty" }
        if (!score.isFinite() || score < 0.0 || score > 1.0) {
            // A NaN would be worse than merely wrong: it compares false against
            // everything, so it would silently corrupt the PC's merged rank
            // ordering rather than surfacing as a bad value.
            throw WireFormatException("score must be finite and in [0, 1], got $score")
        }
        if (!ts.isFinite()) throw WireFormatException("ts must be finite, got $ts")
    }
}

/**
 * One phone's report for one tick.
 *
 * [seq] is a per-device monotonic counter so the PC can reject a replayed or
 * out-of-order datagram without consulting a clock the two do not share.
 */
data class DeviceEnvelope(
    val deviceId: String,
    val seq: Int,
    val sentTs: Double,
    val records: List<TriageRecord> = emptyList(),
) {
    init {
        checkDeviceId(deviceId)
        if (seq < 0) throw WireFormatException("seq must be non-negative, got $seq")
        if (!sentTs.isFinite()) throw WireFormatException("sentTs must be finite, got $sentTs")
        // A device may only speak about its own trainees. Without this, a phone
        // could emit records namespaced to another station and the PC would
        // merge them as that station's -- an integrity property that holds
        // regardless of what authentication the transport later grows.
        records.forEach { record ->
            val owner = deviceOf(record.traineeId)
            if (owner != deviceId) {
                throw WireFormatException(
                    "record '${record.traineeId}' is not namespaced to device '$deviceId'"
                )
            }
        }
    }
}

/**
 * What every transport client must satisfy.
 *
 * The parameter type *is* the privacy guarantee. Widening this interface is the
 * only way to make it possible for anything else to leave the device, which
 * makes that a visible change to this file rather than an accident inside a
 * networking helper.
 */
fun interface EdgeTransport {
    fun send(envelope: DeviceEnvelope)
}

fun checkDeviceId(deviceId: String) {
    if (!DEVICE_ID_RE.matches(deviceId)) {
        throw WireFormatException(
            "device_id must match ${DEVICE_ID_RE.pattern} (got '$deviceId'); " +
                "'-' is reserved as the device/track separator"
        )
    }
}

/** Join a device id and a device-local track id into a global trainee id. */
fun namespaced(deviceId: String, localTrackId: String): String {
    checkDeviceId(deviceId)
    if (localTrackId.isEmpty()) throw WireFormatException("localTrackId must be non-empty")
    return "$deviceId-$localTrackId"
}

/** Recover the device id a trainee id was namespaced under. */
fun deviceOf(traineeId: String): String {
    val index = traineeId.indexOf('-')
    if (index < 0) {
        throw WireFormatException(
            "traineeId '$traineeId' is not device-namespaced (expected 'device-track')"
        )
    }
    return traineeId.substring(0, index)
}

/**
 * Serialise to one JSON line, with the same keys in the same order as
 * `argus.edge.wire.encode_envelope`.
 *
 * Hand-rolled rather than delegating to a JSON library, for the same reason the
 * Python side spells its payload out: this function is the complete and
 * auditable statement of what leaves the device. A reflective serialiser would
 * emit whatever fields a type happened to gain later.
 */
fun encodeEnvelope(envelope: DeviceEnvelope): String {
    val records = envelope.records.joinToString(",") { record ->
        val reasons = record.reasonCodes.joinToString(",") { quote(it) }
        """{"reason_codes": [$reasons], "score": ${num(record.score)}, """ +
            """"trainee_id": ${quote(record.traineeId)}, "ts": ${num(record.ts)}}"""
    }
    return """{"device_id": ${quote(envelope.deviceId)}, "records": [$records], """ +
        """"seq": ${envelope.seq}, "sent_ts": ${num(envelope.sentTs)}, """ +
        """"wire_version": $WIRE_VERSION}"""
}

private fun num(value: Double): String {
    // JSON has no NaN or Infinity; the guards above make these unreachable, but
    // an encoder that could emit them would produce a payload the PC's strict
    // decoder rejects at the far end instead of here.
    if (!value.isFinite()) throw WireFormatException("cannot encode non-finite $value")
    return if (value == Math.floor(value) && Math.abs(value) < 1e15) {
        "${value.toLong()}.0"
    } else {
        value.toString()
    }
}

private fun quote(text: String): String {
    val out = StringBuilder(text.length + 2)
    out.append('"')
    for (ch in text) {
        when (ch) {
            '"' -> out.append("\\\"")
            '\\' -> out.append("\\\\")
            '\n' -> out.append("\\n")
            '\r' -> out.append("\\r")
            '\t' -> out.append("\\t")
            else -> if (ch < ' ') out.append("\\u%04x".format(ch.code)) else out.append(ch)
        }
    }
    out.append('"')
    return out.toString()
}
