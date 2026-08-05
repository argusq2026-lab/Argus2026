package com.argus.edge

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * WebSocket client for the laptop's ingest server, per `docs/PROTOCOL.md`.
 *
 * The lifecycle mirrors the spec exactly: connect → send `hello` as the first
 * frame → wait for `hello_ack` → stream observations. An `error` frame or a
 * close means this session is over; the protocol's reconnection rule is a
 * *fresh* connection with a *fresh* hello (the server's `track_ttl_s` grace
 * window is what preserves the trainee's history across the gap, not anything
 * this client retains).
 *
 * Nothing degrades quietly: every state transition lands in [state] and the
 * listener, so the UI can render "connected", "rejected: …", or "reconnecting"
 * truthfully. Observations sent while not streaming are dropped and *counted*
 * (`droppedWhileDown`), never buffered — a stale observation is not a claim
 * about now, the same reasoning the old aggregator applied to stale devices.
 */
class IngestClient(
    private val serverUrl: String,          // ws://<laptop-ip>:8765
    private val stationId: String,
    private val traineeId: String,
    /**
     * Announced once at handshake and informational on the server. The
     * load-bearing copy is the `exercise` field on every observation, which
     * selects the scoring weight profile -- this one only tells the operator's
     * logs what the station was set up to watch.
     */
    private val exercisePlan: String = "",
    /** Label for the instructor's approval prompt. Optional. */
    private val displayName: String = "",
    /** The session this phone means to join, if a beacon named one. */
    private val sessionName: String = "",
    private val onStateChange: (String) -> Unit,
) {
    enum class State {
        IDLE, CONNECTING, AWAITING_ACK,
        /** The instructor has been asked and has not answered yet. */
        AWAITING_APPROVAL,
        STREAMING, REJECTED, CLOSED,
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    private val stateRef = AtomicReference(State.IDLE)
    @Volatile private var socket: WebSocket? = null
    @Volatile var lastError: String? = null
        private set
    @Volatile var observationsSent: Long = 0
        private set
    @Volatile var droppedWhileDown: Long = 0
        private set

    /** Set by [disconnect]; suppresses the reconnect loop for a deliberate stop. */
    @Volatile private var stopped = false
    @Volatile private var pendingSession: String = ""
    @Volatile private var approvalDeadlineMs: Long = 0L
    @Volatile private var reconnects: Long = 0
    private var backoffMs = 1_000L
    private val reconnectTimer = java.util.Timer("ingest-reconnect", true)

    val state: State get() = stateRef.get()

    fun connect() {
        stopped = false
        if (state == State.CONNECTING || state == State.AWAITING_ACK || state == State.STREAMING) return
        transition(State.CONNECTING)
        val request = Request.Builder().url(serverUrl).build()
        client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                socket = webSocket
                // The first frame on every connection must be hello.
                webSocket.send(
                    encodeHello(
                        stationId,
                        traineeId,
                        exercisePlan = exercisePlan,
                        displayName = displayName,
                        sessionName = sessionName,
                    )
                )
                transition(State.AWAITING_ACK)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                when (val reply = runCatching { ServerReply.parse(text) }.getOrNull()) {
                    is ServerReply.HelloAck -> {
                        backoffMs = 1_000L   // a good handshake resets the backoff
                        transition(State.STREAMING)
                    }
                    is ServerReply.JoinPending -> {
                        // Not an ack and not a refusal: a human has to press a
                        // button. Surfacing it as its own state is what keeps
                        // whoever is standing at the rack from reading a
                        // perfectly healthy connection as a hang and
                        // restarting the station -- which would only queue
                        // another request behind the first.
                        pendingSession = reply.sessionName
                        approvalDeadlineMs =
                            System.currentTimeMillis() + (reply.timeoutS * 1000).toLong()
                        transition(State.AWAITING_APPROVAL)
                    }
                    is ServerReply.Error -> {
                        lastError = reply.message
                        transition(State.REJECTED)
                        // A protocol-level refusal (bad version, id collision,
                        // unknown form code, a declined or unanswered join)
                        // will not fix itself by retrying: retrying would
                        // hammer the server and hide the cause. A join that
                        // was never answered is the one worth reading twice --
                        // the fix is on the instructor's console, not here.
                    }
                    null -> {
                        lastError = "unparseable server frame"
                        transition(State.REJECTED)
                    }
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                lastError = t.message ?: t.javaClass.simpleName
                socket = null
                transition(State.CLOSED)
                scheduleReconnect()
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                socket = null
                lastError = reason.ifEmpty { "closed by server (code $code)" }
                transition(State.CLOSED)
                scheduleReconnect()
            }
        })
    }

    /** Send one observation; drops (and counts) if the session is not live. */
    fun send(obs: Observation): Boolean {
        val ws = socket
        if (state != State.STREAMING || ws == null) {
            droppedWhileDown += 1
            return false
        }
        val ok = ws.send(encodeObservation(obs))
        if (ok) observationsSent += 1 else droppedWhileDown += 1
        return ok
    }

    /**
     * Reconnect after a transport-level drop, with capped exponential backoff.
     *
     * A station runs a whole class on a floor's Wi-Fi; a dropped connection is
     * expected, not exceptional. PROTOCOL.md is explicit that the recovery is a
     * *fresh* connection with a fresh `hello`, and that the server's
     * `track_ttl_s` grace window (not anything this client keeps) is what
     * preserves the trainee's rolling history across the gap. So there is
     * nothing to resume here — just reconnect and re-introduce.
     *
     * Only transport failures retry. A protocol refusal stays REJECTED,
     * because retrying a version mismatch or an id collision cannot succeed and
     * would bury the reason under reconnect noise.
     */
    private fun scheduleReconnect() {
        if (stopped) return
        val delay = backoffMs
        backoffMs = (backoffMs * 2).coerceAtMost(30_000L)
        reconnects += 1
        reconnectTimer.schedule(object : java.util.TimerTask() {
            override fun run() { if (!stopped) connect() }
        }, delay)
    }

    fun disconnect() {
        stopped = true
        socket?.close(1000, "station stopping")
        socket = null
        transition(State.IDLE)
    }

    private fun transition(next: State) {
        stateRef.set(next)
        onStateChange(describe())
    }

    fun describe(): String {
        val retries = if (reconnects > 0) " [${reconnects} reconnect(s)]" else ""
        return when (state) {
            State.IDLE -> "disconnected"
            State.CONNECTING -> "connecting to $serverUrl$retries"
            State.AWAITING_ACK -> "awaiting hello_ack$retries"
            State.AWAITING_APPROVAL -> {
                val left = ((approvalDeadlineMs - System.currentTimeMillis()) / 1000)
                    .coerceAtLeast(0)
                val who = pendingSession.ifEmpty { "the instructor" }
                "waiting for $who to approve this station (${left}s)"
            }
            State.STREAMING -> "streaming ($observationsSent sent$retries)"
            // A refusal is terminal on purpose — see scheduleReconnect.
            State.REJECTED -> "REJECTED — ${lastError ?: "?"} (will not retry)"
            State.CLOSED ->
                "closed${lastError?.let { " — $it" } ?: ""} — retrying in ${backoffMs / 1000}s$retries"
        }
    }
}
