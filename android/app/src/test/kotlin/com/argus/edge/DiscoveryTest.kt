package com.argus.edge

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Beacon parsing (`Discovery.parse`), the half of discovery that runs on
 * hostile input.
 *
 * Anything on the Wi-Fi can send to the discovery port, so the property under
 * test is mostly about what is *rejected*: parsing must be total, and a
 * mismatched protocol version must be dropped rather than offered, or someone
 * setting up a station gets an address whose handshake is going to fail after
 * they have already placed the phone.
 *
 * The mirror of `tests/test_discovery.py` on the server side; the format
 * itself is specified in `docs/PROTOCOL.md`.
 */
class DiscoveryTest {

    private fun datagram(build: JSONObject.() -> Unit = {}): ByteArray {
        val payload = JSONObject()
            .put("type", Discovery.BEACON_TYPE)
            .put("protocol_version", PROTOCOL_VERSION)
            .put("ws_url", "ws://10.0.0.5:8765")
        payload.build()
        return payload.toString().toByteArray(Charsets.UTF_8)
    }

    private fun parse(raw: ByteArray, expected: Int? = PROTOCOL_VERSION) =
        Discovery.parse(raw, raw.size, expected)

    @Test
    fun `a valid beacon parses`() {
        val server = parse(datagram())!!
        assertEquals("ws://10.0.0.5:8765", server.wsUrl)
        assertEquals(PROTOCOL_VERSION, server.protocolVersion)
        assertNull(server.sessionName)
    }

    @Test
    fun `a session name is carried through so a picker can name the laptop`() {
        val server = parse(datagram { put("session_name", "Coach Riley") })!!
        assertEquals("Coach Riley", server.sessionName)
        assertEquals("Coach Riley", server.label)
    }

    @Test
    fun `an unnamed session falls back to its address rather than blank`() {
        assertEquals("ws://10.0.0.5:8765", parse(datagram())!!.label)
    }

    @Test
    fun `the approval mode is read so the app can warn before connecting`() {
        assertEquals(false, parse(datagram())!!.needsApproval)
        assertEquals(true, parse(datagram { put("approval", "manual") })!!.needsApproval)
    }

    @Test
    fun `a beacon from an older server without an approval field defaults to auto`() {
        assertEquals("auto", parse(datagram())!!.approval)
    }

    @Test
    fun `unrelated traffic on the port is dropped rather than throwing`() {
        for (raw in listOf("not json", "[1,2,3]", "", "{}")) {
            assertNull(raw, parse(raw.toByteArray(Charsets.UTF_8)))
        }
        assertNull(parse(byteArrayOf(-1, -2, 0, 3)))
    }

    @Test
    fun `a datagram of another type is dropped`() {
        assertNull(parse(datagram { put("type", "something_else") }))
    }

    @Test
    fun `a beacon without a ws url is dropped`() {
        assertNull(parse(datagram { put("ws_url", "http://10.0.0.5:8765") }))
        assertNull(parse(datagram { remove("ws_url") }))
    }

    @Test
    fun `a mismatched protocol version is dropped`() {
        assertNull(parse(datagram { put("protocol_version", PROTOCOL_VERSION + 1) }))
    }

    @Test
    fun `a missing protocol version is dropped`() {
        assertNull(parse(datagram { remove("protocol_version") }))
    }

    @Test
    fun `the version check can be waived for diagnostics`() {
        val server = parse(datagram { put("protocol_version", 99) }, expected = null)!!
        assertEquals(99, server.protocolVersion)
    }

    @Test
    fun `only the bytes actually received are parsed`() {
        // DatagramPacket hands back a reused buffer with trailing rubbish; a
        // parser that read the whole array instead of packet.length would
        // choke on the previous datagram's tail.
        val real = datagram()
        val buffer = ByteArray(2048) { 0x7A }
        real.copyInto(buffer)
        assertEquals("ws://10.0.0.5:8765", Discovery.parse(buffer, real.size, PROTOCOL_VERSION)!!.wsUrl)
    }
}

/**
 * The join-approval half of the handshake, on the phone's side.
 *
 * `join_pending` is neither an ack nor a refusal: the connection stays open
 * and a decision follows. Parsing it as a distinct reply is what lets the app
 * say "waiting for the instructor" instead of showing a healthy connection as
 * a hang — and a station that looks hung gets restarted by whoever is next to
 * it, which only queues another request behind the first.
 *
 * Mirrors `tests/test_admission.py` on the server side.
 */
class JoinPendingTest {

    @Test
    fun `join_pending parses as its own reply`() {
        val reply = ServerReply.parse(
            JSONObject()
                .put("type", "join_pending")
                .put("session_name", "Coach Riley")
                .put("request_id", "join-1")
                .put("timeout_s", 120.0)
                .toString()
        )
        assertEquals(ServerReply.JoinPending("Coach Riley", "join-1", 120.0), reply)
    }

    @Test
    fun `an ack is still an ack`() {
        assertEquals(
            ServerReply.HelloAck,
            ServerReply.parse(JSONObject().put("type", "hello_ack").toString()),
        )
    }

    @Test
    fun `a declined join arrives as an error, not as a pending`() {
        val reply = ServerReply.parse(
            JSONObject()
                .put("type", "error")
                .put("message", "the instructor declined this join request")
                .toString()
        )
        assertEquals(
            ServerReply.Error("the instructor declined this join request"),
            reply,
        )
    }

    @Test
    fun `hello carries the display name and session only when they are set`() {
        val bare = JSONObject(encodeHello("s0", "t0"))
        assertEquals(false, bare.has("display_name"))
        assertEquals(false, bare.has("session_name"))

        val full = JSONObject(
            encodeHello("s0", "t0", displayName = "Alex", sessionName = "Coach Riley")
        )
        assertEquals("Alex", full.getString("display_name"))
        assertEquals("Coach Riley", full.getString("session_name"))
    }
}

/** The empty-station heartbeat. Mirrors `tests/test_admission.py`. */
class IdleTest {

    @Test
    fun `idle encodes as its own message type`() {
        val obj = JSONObject(encodeIdle(12.5))
        assertEquals("idle", obj.getString("type"))
        assertEquals(12.5, obj.getDouble("ts"), 1e-9)
    }

    @Test
    fun `idle carries no subject fields at all`() {
        // Not an observation with the pose nulled: an observation asserts a
        // reading about a person, and a null inside one gets scored as a zero.
        val obj = JSONObject(encodeIdle(1.0))
        for (key in listOf("bbox_xyxy", "keypoints_xy", "keypoints_conf",
                           "form_reason_codes", "exercise")) {
            assertEquals(key, false, obj.has(key))
        }
    }
}
