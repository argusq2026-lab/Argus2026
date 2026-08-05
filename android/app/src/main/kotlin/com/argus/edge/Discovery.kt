package com.argus.edge

import android.content.Context
import android.net.wifi.WifiManager
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.SocketTimeoutException

/**
 * Finding the laptop without anyone typing its IP into this phone.
 *
 * The laptop broadcasts a small UDP datagram saying where its WebSocket
 * listener is (see `docs/PROTOCOL.md`, "Discovery", and `argus/discovery.py`
 * for the sending half). This listens for one. That is the whole mechanism:
 * no mDNS, no service registry, no dependency on either end.
 *
 * Two properties matter more than the convenience:
 *
 * - **It is a hint, not an authority.** Everything here is optional. A phone
 *   that hears nothing — guest Wi-Fi with broadcast filtering, a laptop on a
 *   different VLAN — falls back to the address field, which still works
 *   exactly as it did. Discovery failing must never be the reason a station
 *   cannot be set up.
 * - **It parses hostile input.** Any process on the network can send to this
 *   port. Every rejection here returns null rather than throwing, and the
 *   protocol version is checked *before* the address is offered, so a
 *   mismatched server shows up during setup instead of as a handshake
 *   rejection after someone has already placed the phone on a rack.
 *
 * A beacon is never auto-connected to. It fills the field in; a human still
 * presses Connect. An unauthenticated datagram from the local network should
 * not be able to point this phone at an arbitrary server on its own.
 */
object Discovery {

    /** Marks a datagram as ours. Anything else on this port is noise. */
    const val BEACON_TYPE = "argus_beacon"

    /** Matches `discovery.port` in the laptop's `configs/argus.default.toml`. */
    const val DEFAULT_PORT = 8766

    data class Server(
        val wsUrl: String,
        val protocolVersion: Int,
        /**
         * The instructor's session name, when the laptop has one set. This is
         * what makes a room with three laptops choosable rather than a list of
         * IP addresses.
         */
        val sessionName: String?,
        /**
         * `"auto"` or `"manual"`. Carried so the app can say "the instructor
         * will approve this" up front, instead of looking hung while a join
         * request sits on someone else's screen.
         */
        val approval: String,
    ) {
        /** What to show in a picker. Falls back to the address, which works. */
        val label: String get() = sessionName ?: wsUrl

        val needsApproval: Boolean get() = approval == "manual"
    }

    /**
     * Validate one datagram. Null for anything that is not a usable beacon.
     *
     * Deliberately total: malformed input is the normal case on a shared
     * network, not an error worth reporting to whoever is setting up a phone.
     */
    fun parse(raw: ByteArray, length: Int, expectedProtocolVersion: Int?): Server? {
        val payload = try {
            JSONObject(String(raw, 0, length, Charsets.UTF_8))
        } catch (_: Exception) {
            return null
        }
        if (payload.optString("type") != BEACON_TYPE) return null

        val wsUrl = payload.optString("ws_url")
        if (!wsUrl.startsWith("ws://")) return null

        if (!payload.has("protocol_version")) return null
        val version = payload.optInt("protocol_version", Int.MIN_VALUE)
        if (version == Int.MIN_VALUE) return null
        if (expectedProtocolVersion != null && version != expectedProtocolVersion) return null

        val sessionName = payload.optString("session_name").ifEmpty { null }
        val approval = payload.optString("approval").ifEmpty { "auto" }
        return Server(wsUrl, version, sessionName, approval)
    }

    /**
     * Listen for beacons, blocking for up to [timeoutMs]. **Not for the main
     * thread** — call it on a background executor.
     *
     * Holds a multicast lock for the duration. Wi-Fi firmware drops broadcast
     * frames to save power unless something asks it not to, so without this
     * the socket is correct and simply never receives anything — the most
     * confusing possible failure, because nothing reports an error.
     */
    fun listen(
        context: Context,
        port: Int = DEFAULT_PORT,
        timeoutMs: Long = 3_000,
        expectedProtocolVersion: Int? = PROTOCOL_VERSION,
    ): List<Server> {
        val wifi = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
        val lock = wifi?.createMulticastLock("argus-discovery")?.apply {
            setReferenceCounted(true)
            acquire()
        }

        val found = LinkedHashMap<String, Server>()
        var socket: DatagramSocket? = null
        try {
            socket = DatagramSocket(null).apply {
                reuseAddress = true
                broadcast = true
                bind(java.net.InetSocketAddress(port))
                soTimeout = 500
            }
            val buffer = ByteArray(2048)
            val deadline = System.currentTimeMillis() + timeoutMs
            while (System.currentTimeMillis() < deadline) {
                val packet = DatagramPacket(buffer, buffer.size)
                try {
                    socket.receive(packet)
                } catch (_: SocketTimeoutException) {
                    continue
                }
                val server = parse(packet.data, packet.length, expectedProtocolVersion)
                if (server != null) found.putIfAbsent(server.wsUrl, server)
            }
        } catch (_: Exception) {
            // A port already in use, or no network at all. The caller's
            // fallback is the address field, so this is not worth surfacing
            // as anything louder than "no server found".
        } finally {
            socket?.close()
            try {
                lock?.release()
            } catch (_: RuntimeException) {
                // Already released; nothing to do.
            }
        }
        return found.values.toList()
    }
}
