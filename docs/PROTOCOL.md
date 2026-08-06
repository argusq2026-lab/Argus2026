# Argus wire protocol — phone to laptop

This is the contract a phone app must implement to feed Argus. The Android
client that implements it lives in [`android/`](../android/) in this same
repository (see `android/README.md`); an iOS client does not exist. This
document stays precise enough to build a client against without reading the
server's code either way. `demo/replay_client.py` is a small reference
implementation of the client side of exactly this protocol, in Python; read
it alongside this document.

Everything here is enforced by `argus.ingest.protocol` and covered by
`tests/test_ingest_protocol.py` and `tests/test_ingest_server.py` — if this
document and that code ever disagree, the code is what actually runs, and
that's a documentation bug to fix.

---

## Transport

One WebSocket connection per trainee's phone, to `ws://<laptop-ip>:<ws_port>`
(default port `8765`, from `[ingest]` in the config). No URL path is
required or inspected. Every message, in both directions, is exactly one
WebSocket text frame containing one JSON object — there is no batching of
multiple logical messages into one frame, and no framing beyond what
WebSocket already provides.

`argus doctor` prints the LAN-reachable address(es) a phone should use — not
`127.0.0.1`, which only works from the laptop itself. A phone does not have to
be told the address by a human, though; see **Discovery** below.

---

## Discovery

Optional, and strictly a convenience: it removes the step where someone reads
an IP off the laptop and types it into every phone, once per phone, again
whenever the DHCP lease moves.

The server broadcasts one UDP datagram to port `discovery.port` (default
`8766`) every `discovery.interval_s` (default 1 s):

```json
{
  "type": "argus_beacon",
  "protocol_version": 1,
  "ws_url": "ws://10.73.51.76:8765",
  "session_name": "Coach Riley — 6pm HIIT",
  "approval": "manual"
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `type` | `"argus_beacon"` | yes | Marks the datagram as ours. Anything else on this port is unrelated traffic and must be dropped silently. |
| `protocol_version` | int | yes | The server's. A client should **check this before offering the address**, so a mismatch shows up during setup rather than as a handshake rejection after the phone has been placed on a rack. |
| `ws_url` | string | yes | Where to connect. Always `ws://host:port`. |
| `session_name` | string | no | The instructor's session name. This is what makes a room with three laptops choosable rather than a list of IP addresses. Absent when the laptop has none set, in which case show the address. |
| `approval` | string | yes | `"auto"` or `"manual"` — see **Admission**. Carried so a phone can say "the instructor will approve this" up front, instead of looking hung once it has connected. Treat an absent value as `"auto"`. |

Rules a client must follow:

* **Listen, never reply.** The server has no listener on this port; it only
  sends. Discovery adds no inbound surface to the laptop.
* **Suggest, never auto-connect.** A beacon is an unauthenticated datagram
  from whoever is on the Wi-Fi. It may fill an address field in; a human
  presses Connect. Anything else lets any host on the network point a phone
  at a server of its choosing.
* **Drop malformed input silently.** Any process can send to this port, so
  bad input is the normal case, not an error worth reporting — unlike a
  malformed `observation`, which comes from a phone that has already shaken
  hands and means a real disagreement.
* **Fall back to typing.** A phone that hears nothing — broadcast-filtered
  guest Wi-Fi, a laptop on another VLAN — must still accept a typed address.
  Discovery failing is an inconvenience, never the reason a station cannot be
  set up.

Nothing about a trainee is in the beacon, and structurally cannot be: the
payload is built once at startup from config, before any phone has connected.
It advertises a port that `ingest.ws_host = "0.0.0.0"` already left open, so
it publishes the *fact* of the service rather than new access to it. Set
`discovery.enabled = false` where even that is unwanted.

Both halves are implemented and testable without a phone in the room:
`argus.discovery` sends, `argus discover` listens and prints what it hears,
and `android/.../Discovery.kt` is the same listener on the device.

---

## Handshake

The **first message** on every connection must be `hello`:

```json
{
  "type": "hello",
  "protocol_version": 1,
  "station_id": "station-3",
  "trainee_id": "trainee-42",
  "exercise_plan": "squat"
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `type` | `"hello"` | yes | |
| `protocol_version` | int | yes | Must exactly equal the server's `ingest.protocol_version`. There is no negotiation — a mismatch is rejected outright, not downgraded. |
| `station_id` | string | yes | A label for this physical station/camera. Not required to be globally unique; purely diagnostic (shows up in server logs). |
| `trainee_id` | string | yes | **The triage key.** Must be globally unique across every trainee currently on the floor — this is what an alert is dispatched against. Two phones simultaneously claiming the same `trainee_id` is rejected as a collision, not merged. |
| `exercise_plan` | string | no | Free-form, informational only; not used in scoring today. |
| `display_name` | string | no | What to call this station on the instructor's approval prompt (e.g. `"Alex — rack 3"`). At most 64 characters. Display-only, never scored, never logged. Only reaches a human when the session requires approval, but harmless to always send. |
| `session_name` | string | no | The session this phone believes it is joining, if it learned one from a beacon (see **Discovery**). The server **rejects a mismatch**: on a floor with two laptops, silently joining the wrong one means a trainee is monitored by an instructor who is not watching them. Omit it if the address was typed by hand — a phone that never heard a beacon cannot know the name, and must still be able to connect. |

The server replies with exactly one of:

* `{"type": "hello_ack", "accepted": true}` — proceed to sending `observation`
  messages.
* `{"type": "join_pending", ...}` — the instructor has to approve this station
  first. See **Admission** below. The connection stays open; exactly one
  `hello_ack` or `error` follows.
* `{"type": "error", "message": "<why>"}`, immediately followed by the
  server closing the connection (WebSocket close code `1008`). Causes:
  wrong `protocol_version`, a missing/empty required field, a `trainee_id`
  that is already connected elsewhere, a `session_name` naming a different
  session, or a join request that was declined or went unanswered.

---

## Admission

A session runs one of two admission modes, set by `[session] approval` on the
laptop and advertised in the beacon so a phone knows before it connects.

**`"auto"` (the default)** — a well-formed `hello` is acknowledged
immediately. Nothing below applies; this is the handshake exactly as it was
before admission existed.

**`"manual"`** — the server replies:

```json
{
  "type": "join_pending",
  "session_name": "Coach Riley — 6pm HIIT",
  "request_id": "join-1",
  "timeout_s": 120.0
}
```

and holds the connection open while the request sits on the instructor's
console. Exactly one of these follows, and one always does:

| Then | Meaning |
|---|---|
| `hello_ack` | Approved. Stream observations as normal. |
| `error` — "the instructor declined this join request" | Refused by a human. |
| `error` — "no instructor answered…" | `timeout_s` elapsed with nobody deciding. |
| `error` — "a newer join request… replaced this one" | The same `trainee_id` asked again, almost always this phone reconnecting. The newer request is the live one. |

A client must:

* **Show the wait as a wait.** A phone that displays nothing after
  `join_pending` looks hung, and a station that looks hung gets restarted by
  whoever is standing next to it — which only queues another request behind
  the first. Say who is being waited on and for how long.
* **Not retry automatically.** All four outcomes above are protocol refusals,
  terminal like any other. A declined phone that reconnected in a loop would
  bury the instructor in prompts; an unanswered one would do the same. The fix
  for both is on the console, not on the phone.
* **Expect a decision at any point** inside `timeout_s`, including
  immediately.

Duplicate `trainee_id` is checked *before* the instructor is asked, so a phone
that would be refused anyway never costs anyone a decision.

If no `hello` arrives within 10 seconds of connecting, the server closes the
connection with code `1002` and no `hello_ack`/`error` is sent (there is
nothing yet to reply to).

---

## Observation messages

After a successful handshake, stream `observation` messages at roughly
**5–15 Hz** (no hard limit; the merged rank is recomputed on its own
schedule — `ingest.rank_interval_s`, default every 0.5 s — independent of how
fast any individual phone sends):

```json
{
  "type": "observation",
  "ts": 1730649600.125,
  "bbox_xyxy": [0.12, 0.08, 0.61, 0.97],
  "keypoints_xy": [[0.34, 0.10], [0.36, 0.09], "... 17 pairs total"],
  "keypoints_conf": [0.91, 0.87, "... 17 values total"],
  "exercise": "squat",
  "rep_count": 12,
  "form_ok": false,
  "form_reason_codes": ["insufficient_depth"]
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `type` | `"observation"` | yes | |
| `ts` | float | yes | Unix epoch seconds, from the **phone's own clock**. No cross-phone clock synchronization is assumed or enforced — see `docs/VALIDATION.md`. |
| `bbox_xyxy` | `[x0, y0, x1, y1]` | yes | The trainee's bounding box, **normalized to [0, 1]** of the phone's own camera frame — resolution-independent, so a 1080p and a 4K phone report the same numbers for the same framing. |
| `keypoints_xy` | 17 × `[x, y]` | yes | COCO-17 keypoints, normalized to [0, 1] the same way. See "Keypoint layout" below. |
| `keypoints_conf` | 17 × float | yes | Per-keypoint confidence in [0, 1]. A keypoint the phone's pose model has no estimate for should be reported at low/zero confidence, not omitted — all 17 slots are always present. |
| `exercise` | string | no | The exercise being performed (e.g. `"plank"`, `"squat"`). **Selects the server's scoring weight profile** — see "The exercise field is load-bearing" below. Lowercased on the server; unlike `form_reason_codes` it is *not* a closed vocabulary, and an exercise the server has no profile for scores on the default weights rather than being rejected. At most 64 characters — it is a classifier label, not free text, and a longer value is rejected like any other malformed field. It is also shown on the trainer console (see [`CONSOLE.md`](CONSOLE.md)). |
| `rep_count` | int | no | Running rep count for the current set. **Display-only**, shown on the trainer console, never scored. Must be a non-negative integer; `true`/`false` is rejected rather than counted as 1. |
| `form_ok` | bool | no | The phone's own correct/incorrect verdict. **Display-only** — the server derives whether form is flagged from `form_reason_codes` being non-empty, not from this field. Omitting it is **not** the same as sending `false`: the console shows an absent verdict as unknown, not as a pass, so send it when the phone has one and omit it when it does not. |

All three are validated and carried through to the trainer console. `rep_count`
and `form_ok` are display-only as a guarantee, not an accident: the rank must
stay a pure function of the numeric pose history, the closed-vocabulary form
codes, and `exercise`'s weight-profile selection below, so scoring a
phone-maintained rep counter would make it a function of an unauditable
device-side value. `exercise` is the one exception to "display-only" — see
below.
| `form_reason_codes` | list of strings | no (default `[]`) | The phone's on-device classifier's closed-vocabulary reasons for an incorrect rep. **Every code must appear in the server's `[scoring.form_error_vocab]`** (see below) — an unrecognised code is treated as a protocol/version mismatch, not scored as zero: the server sends an `error` and closes the connection. |

A malformed `observation` (wrong type, missing field, wrong-length array, or
an unrecognised `form_reason_codes` entry) gets the same treatment as a bad
`hello`: an `error` message, then the connection closes with code `1008`.
There is no partial acceptance of a malformed message.

## Idle — "nobody is in frame"

When the phone has no subject, it must say so rather than going quiet:

```json
{ "type": "idle", "ts": 1730649600.125 }
```

Send it about once a second — it is a liveness fact, not a measurement, and
only has to arrive comfortably inside `ingest.track_ttl_s`.

**This is not optional for a station that will ever be pointed at an empty
rack**, which is every station: they get set up before their trainee arrives.
Without it, a healthy phone watching nobody is indistinguishable from a dead
one — both send nothing — so the server evicts the session at `track_ttl_s`,
refuses the phone's next message, and the phone reconnects. That flap runs for
as long as the rack is empty.

It is deliberately **not** an observation with the subject fields nulled. An
observation asserts a reading about a person; this asserts that there is no
person to read. Making the difference a null inside a message everything else
treats as a measurement is how a null ends up scored as a zero.

On the server it refreshes the session exactly as an observation does, and
marks the trainee absent: the station stops being ranked (its last two seconds
of pose describe somebody who has left, and scoring them would report
`prolonged_stillness` about an empty rack) and the console draws it as **ready**
rather than as silent. A phone that sends *neither* observations nor idles is
still evicted — `idle` is not a way for a wedged app to look alive.

---

### Keypoint layout

Standard COCO-17, in this exact order:

```
 0 nose            6 right_shoulder   12 right_hip
 1 left_eye        7 left_elbow       13 left_knee
 2 right_eye       8 right_elbow      14 right_knee
 3 left_ear        9 left_wrist       15 left_ankle
 4 right_ear      10 right_wrist      16 right_ankle
 5 left_shoulder  11 left_hip
```

**"Left" and "right" are the subject's own left and right, not the image's.**
A trainee facing the camera therefore has their *left* shoulder at the
*larger* image x-coordinate. This matters for `off_task_orientation` scoring
(`argus.config.ScoringConfig.off_task_reference_angle_deg`): getting this
backwards makes every attentive trainee score a full off-task deviation. If
the phone's own pose model (e.g. MediaPipe Pose, on-device) emits a different
landmark layout or count, the phone is responsible for remapping to COCO-17
before sending — the server does no remapping of its own.

### The `exercise` field is load-bearing

`exercise` was informational in earlier revisions of this document. It is not
any more: the server looks it up in `[scoring.exercise_weights]` to pick which
weight vector scores that trainee, because some features are wrong for some
movements.

The concrete case is the plank. `fall` fires on a bounding box wider than it
is tall; `stillness` fires on a centroid that stops moving; `off_task` fires
on a shoulder line away from the station-facing angle. A correct plank is all
three. Measured against the default weights, a textbook plank scores **0.42**
of a 0.5 alert threshold and reports `prolonged_stillness,
off_task_orientation` — an instructor is told a trainee holding perfect form
has stopped moving and is facing the wrong way, and the actual form signal is
worth 0.12 against 0.42 of noise. With `exercise: "plank"` the same
observation scores 0.0 and reports nothing, and a sagging plank scores 0.68.

So for any exercise with a profile, omitting `exercise` is not a small loss of
fidelity — it is the difference between a correct rep scoring 0.0 and scoring
most of an alert. A phone should send it on every observation, not only when
it changes: the server tracks the most recent value and has no way to
distinguish "still planking" from "stopped reporting".

Exercises with no configured profile are unaffected, and sending an unknown
one is safe by design — it is a label, not a code.

### The form-error vocabulary must match exactly

`[scoring.form_error_vocab]` in the shared config (`configs/argus.default.toml`)
is the single source of truth for both ends. Its keys are lowercased when the
server loads them; a phone should send codes already lowercase (e.g.
`"knee_valgus"`, not `"Knee_Valgus"`) since the comparison is case-sensitive
on the wire. Phone and laptop are expected to deploy from copies of the same
config file — there is no protocol-level vocabulary sync.

---

## Reconnection

* A `trainee_id` currently connected is exclusive: a second `hello` claiming
  it is rejected (`error` + close `1008`).
* A cleanly disconnected trainee's session is **not** deleted immediately.
  It stays in a grace window of `ingest.track_ttl_s` seconds (default 10s):
  reconnecting with the same `trainee_id` within that window resumes the
  same rolling observation history and alert-suppression state, so a brief
  Wi-Fi drop doesn't reset a trainee's triage history or cause an alert to
  re-fire from scratch.
* If the grace window elapses — whether the phone reconnects or not — the
  session is dropped. A phone whose connection technically stayed open but
  stopped sending observations for longer than `track_ttl_s` gets the same
  treatment: the *next* observation it sends after expiry is rejected with
  an `error` telling it to reconnect with a fresh `hello`.

---

## What the server never sends back

Nothing beyond `hello_ack` and `error`. In particular, the server never
echoes a trainee's own score or rank to their phone — ranking is for the
instructor's console (`GET /triage`, or the dashboard at `GET /`), not the
trainee's own device.

---

## Versioning

`protocol_version` is a single integer, bumped whenever this document's wire
format changes incompatibly. There is no partial compatibility — a client
built against version 1 talking to a server configured for version 2 (or
vice versa) is rejected at the handshake, loudly, rather than degrading.
