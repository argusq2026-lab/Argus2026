package com.argus.edge

import org.json.JSONArray
import org.json.JSONObject

/**
 * The phone side of `docs/PROTOCOL.md` — encode only.
 *
 * The laptop validates (`argus.ingest.protocol`); this file must produce
 * messages that validator accepts, and `ProtocolTest` checks exactly that
 * against `tests/data/protocol_vectors.json`, the same fixture the server's
 * own tests run.
 *
 * There is deliberately no observation decoder here. The server only ever
 * sends `hello_ack` and `error` (see "What the server never sends back" in
 * PROTOCOL.md), so the only parsing on this side is those two control frames —
 * a phone that could be instructed by its server would be a wider surface than
 * the protocol defines.
 *
 * Privacy posture, carried over from the deleted envelope layer: nothing in
 * this file can hold a frame. An observation is scalars and fixed-length
 * lists; the encoder has no parameter through which pixels could travel.
 */
const val PROTOCOL_VERSION = 1

/** COCO-17 — every observation carries all 17 slots, always. */
const val NUM_KEYPOINTS = 17

class ProtocolException(message: String) : IllegalArgumentException(message)

/**
 * The use cases this app can stream. A laptop refuses a `hello` whose use case
 * is not the one its session is set to, so this is not a cosmetic label — it
 * decides whether the station is admitted at all. See docs/PROTOCOL.md.
 */
const val USE_CASE_FITNESS = "fitness"
const val USE_CASE_NURSING = "nursing"

/** The one nursing procedure the laptop can score today. */
const val PROCEDURE_CPR = "cpr"

/**
 * One observation, in the protocol's normalized coordinate space.
 *
 * `bboxNorm` and every keypoint are fractions of the phone's own frame in
 * [0, 1] — resolution-independent by construction, which is why the conversion
 * from detector pixels happens *here*, where the invariant can be enforced,
 * and not scattered at call sites.
 *
 * A phone with no pose model reports all 17 keypoints at zero confidence —
 * PROTOCOL.md is explicit that unknown keypoints are low-confidence, never
 * omitted.
 */
data class Observation(
    val ts: Double,
    val bboxNorm: List<Double>,          // [x0, y0, x1, y1]
    val keypointsXy: List<List<Double>> = List(NUM_KEYPOINTS) { listOf(0.0, 0.0) },
    val keypointsConf: List<Double> = List(NUM_KEYPOINTS) { 0.0 },
    /**
     * Which use case this station is running. Selects the parser *and* the
     * scorer on the laptop, so it decides what the rest of this object means.
     */
    val useCase: String = USE_CASE_FITNESS,
    /** Nursing's counterpart to [exercise] — "cpr". Null for other use cases. */
    val procedure: String? = null,
    val exercise: String? = null,
    val repCount: Int? = null,
    val formOk: Boolean? = null,
    val formReasonCodes: List<String> = emptyList(),
) {
    init {
        if (!ts.isFinite()) throw ProtocolException("ts must be finite")
        // Fitness's fields are fitness's. A nursing observation carrying a rep
        // count or a form code would be sending a vocabulary its own parser
        // does not read, and the laptop would silently drop it -- so it is
        // refused here, where the mistake is visible, rather than on the wire.
        if (useCase != USE_CASE_FITNESS) {
            if (exercise != null || repCount != null || formOk != null || formReasonCodes.isNotEmpty()) {
                throw ProtocolException(
                    "exercise/rep_count/form_ok/form_reason_codes belong to fitness; " +
                        "use_case '$useCase' must not set them"
                )
            }
        }
        if (procedure != null && useCase != USE_CASE_NURSING) {
            throw ProtocolException("procedure is nursing's field, not '$useCase'")
        }
        if (bboxNorm.size != 4 || bboxNorm.any { !it.isFinite() || it < 0.0 || it > 1.0 }) {
            throw ProtocolException("bbox_xyxy must be 4 normalized values in [0, 1], got $bboxNorm")
        }
        if (keypointsXy.size != NUM_KEYPOINTS || keypointsXy.any { it.size != 2 }) {
            throw ProtocolException("keypoints_xy must be $NUM_KEYPOINTS [x, y] pairs")
        }
        if (keypointsConf.size != NUM_KEYPOINTS) {
            throw ProtocolException("keypoints_conf must be $NUM_KEYPOINTS values")
        }
    }

    companion object {
        /**
         * Normalize a detection (and its pose, when the pose model produced
         * one) into protocol space. Without a pose the observation carries the
         * protocol's zero-confidence keypoints; unmapped joints (knees/ankles
         * in the 25-point export) arrive the same way, so the laptop's
         * confidence gate excludes them rather than scoring a fabricated
         * position.
         */
        fun fromDetection(
            ts: Double, det: Detection, pose: PoseResult?,
            frameWidth: Int, frameHeight: Int,
            useCase: String = USE_CASE_FITNESS,
            procedure: String? = null,
        ): Observation {
            val w = frameWidth.toDouble()
            val h = frameHeight.toDouble()
            fun clamp01(v: Double) = v.coerceIn(0.0, 1.0)
            val keypointsXy = pose?.let { p ->
                List(NUM_KEYPOINTS) { k ->
                    listOf(
                        clamp01(p.keypointsXy[k * 2].toDouble() / w),
                        clamp01(p.keypointsXy[k * 2 + 1].toDouble() / h),
                    )
                }
            } ?: List(NUM_KEYPOINTS) { listOf(0.0, 0.0) }
            val keypointsConf = pose?.let { p ->
                List(NUM_KEYPOINTS) { k -> p.keypointsConf[k].toDouble().coerceIn(0.0, 1.0) }
            } ?: List(NUM_KEYPOINTS) { 0.0 }
            return Observation(
                ts = ts,
                bboxNorm = listOf(
                    clamp01(det.x0 / w), clamp01(det.y0 / h),
                    clamp01(det.x1 / w), clamp01(det.y1 / h),
                ),
                keypointsXy = keypointsXy,
                keypointsConf = keypointsConf,
                useCase = useCase,
                procedure = procedure,
            )
        }
    }
}

fun encodeHello(
    stationId: String,
    traineeId: String,
    exercisePlan: String = "",
    /** Label shown on the instructor's approval prompt. Optional. */
    displayName: String = "",
    /**
     * The session this phone believes it is joining, when it learned one from
     * a beacon. Sent so the server can refuse a mismatch: on a floor with two
     * laptops, silently joining the wrong one is a trainee monitored by an
     * instructor who is not watching them.
     */
    sessionName: String = "",
    /**
     * What this station is running. The laptop rejects a `hello` that does not
     * match its own `[session] use_case`, which is the check that stops a
     * fitness phone being admitted onto a nursing floor and scored by nothing.
     *
     * Omitted from the message when it is fitness: a server that predates the
     * field defaults an absent `use_case` to fitness anyway, so leaving it out
     * keeps the fitness handshake byte-identical to what shipped.
     */
    useCase: String = USE_CASE_FITNESS,
): String {
    if (stationId.isEmpty()) throw ProtocolException("station_id must not be empty")
    if (traineeId.isEmpty()) throw ProtocolException("trainee_id must not be empty")
    val obj = JSONObject()
        .put("type", "hello")
        .put("protocol_version", PROTOCOL_VERSION)
        .put("station_id", stationId)
        .put("trainee_id", traineeId)
    if (exercisePlan.isNotEmpty()) obj.put("exercise_plan", exercisePlan)
    if (displayName.isNotEmpty()) obj.put("display_name", displayName)
    if (sessionName.isNotEmpty()) obj.put("session_name", sessionName)
    if (useCase != USE_CASE_FITNESS) obj.put("use_case", useCase)
    return obj.toString()
}

/**
 * "I am here and watching, and there is nobody in frame."
 *
 * Deliberately not an observation with the subject nulled: an observation
 * asserts a reading about a person, this asserts that there is no person to
 * read. See docs/PROTOCOL.md.
 */
fun encodeIdle(ts: Double): String =
    JSONObject().put("type", "idle").put("ts", ts).toString()

fun encodeObservation(obs: Observation): String {
    val obj = JSONObject()
        .put("type", "observation")
        .put("ts", obs.ts)
        .put("bbox_xyxy", JSONArray(obs.bboxNorm))
        .put("keypoints_xy", JSONArray(obs.keypointsXy.map { JSONArray(it) }))
        .put("keypoints_conf", JSONArray(obs.keypointsConf))
    // Same reasoning as `encodeHello`: omitted when fitness, so the shape that
    // shipped stays exactly as it was.
    if (obs.useCase != USE_CASE_FITNESS) obj.put("use_case", obs.useCase)
    obs.procedure?.let { obj.put("procedure", it) }
    obs.exercise?.let { obj.put("exercise", it) }
    obs.repCount?.let { obj.put("rep_count", it) }
    obs.formOk?.let { obj.put("form_ok", it) }
    if (obs.formReasonCodes.isNotEmpty()) {
        obj.put("form_reason_codes", JSONArray(obs.formReasonCodes))
    }
    return obj.toString()
}

/** The two control frames the server can send; anything else is a violation. */
sealed class ServerReply {
    data object HelloAck : ServerReply()
    data class Error(val message: String) : ServerReply()

    /**
     * The instructor has to approve this station before it can stream.
     *
     * Sent instead of [HelloAck] when the session runs
     * `session.approval = "manual"`. It is not a refusal and not an
     * acknowledgement: the connection stays open and a decision follows, or
     * an [Error] does when nobody answers within [timeoutS].
     */
    data class JoinPending(
        val sessionName: String,
        val requestId: String,
        val timeoutS: Double,
    ) : ServerReply()

    companion object {
        fun parse(text: String): ServerReply {
            val obj = JSONObject(text)
            return when (val type = obj.optString("type")) {
                "hello_ack" -> HelloAck
                "error" -> Error(obj.optString("message", "(no message)"))
                "join_pending" -> JoinPending(
                    obj.optString("session_name"),
                    obj.optString("request_id"),
                    obj.optDouble("timeout_s", 0.0),
                )
                else -> throw ProtocolException("server sent unexpected frame type '$type'")
            }
        }
    }
}
