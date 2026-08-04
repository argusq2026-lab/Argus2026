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
    private val onStateChange: (String) -> Unit,
) {
    enum class State { IDLE, CONNECTING, AWAITING_ACK, STREAMING, REJECTED, CLOSED }

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

    val state: State get() = stateRef.get()

    fun connect() {
        if (state == State.CONNECTING || state == State.AWAITING_ACK || state == State.STREAMING) return
        transition(State.CONNECTING)
        val request = Request.Builder().url(serverUrl).build()
        client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                socket = webSocket
                // The first frame on every connection must be hello.
                webSocket.send(encodeHello(stationId, traineeId))
                transition(State.AWAITING_ACK)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                when (val reply = runCatching { ServerReply.parse(text) }.getOrNull()) {
                    is ServerReply.HelloAck -> transition(State.STREAMING)
                    is ServerReply.Error -> {
                        lastError = reply.message
                        transition(State.REJECTED)
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
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                socket = null
                transition(State.CLOSED)
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

    fun disconnect() {
        socket?.close(1000, "station stopping")
        socket = null
        transition(State.IDLE)
    }

    private fun transition(next: State) {
        stateRef.set(next)
        onStateChange(describe())
    }

    fun describe(): String = when (state) {
        State.IDLE -> "disconnected"
        State.CONNECTING -> "connecting to $serverUrl"
        State.AWAITING_ACK -> "awaiting hello_ack"
        State.STREAMING -> "streaming ($observationsSent sent)"
        State.REJECTED -> "REJECTED — ${lastError ?: "?"}"
        State.CLOSED -> "closed${lastError?.let { " — $it" } ?: ""} (reconnect sends a fresh hello)"
    }
}
