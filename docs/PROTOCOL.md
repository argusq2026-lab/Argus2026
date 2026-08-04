# Argus wire protocol — phone to laptop

This is the contract a phone app must implement to feed Argus. It is not
implemented in this repository — the Android (or iOS) client is a separate,
future project — so this document has to be precise enough to build against
without seeing the server's code. `demo/replay_client.py` is a small
reference implementation of the client side of exactly this protocol, in
Python; read it alongside this document.

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
`127.0.0.1`, which only works from the laptop itself.

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

The server replies with exactly one of:

* `{"type": "hello_ack", "accepted": true}` — proceed to sending `observation`
  messages.
* `{"type": "error", "message": "<why>"}`, immediately followed by the
  server closing the connection (WebSocket close code `1008`). Causes:
  wrong `protocol_version`, a missing/empty required field, or a
  `trainee_id` that is already connected elsewhere.

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
| `exercise` | string | no | The classified exercise (e.g. `"squat"`, `"burpee"`). Informational only today — not used in scoring, only reserved for a future dashboard column. |
| `rep_count` | int | no | Running rep count for the current set. Informational only, same as `exercise`. |
| `form_ok` | bool | no | The phone's own correct/incorrect verdict. Informational only — the server derives whether form is flagged from `form_reason_codes` being non-empty, not from this field. A phone should still send it accurately; it is reserved for future display. |
| `form_reason_codes` | list of strings | no (default `[]`) | The phone's on-device classifier's closed-vocabulary reasons for an incorrect rep. **Every code must appear in the server's `[scoring.form_error_vocab]`** (see below) — an unrecognised code is treated as a protocol/version mismatch, not scored as zero: the server sends an `error` and closes the connection. |

A malformed `observation` (wrong type, missing field, wrong-length array, or
an unrecognised `form_reason_codes` entry) gets the same treatment as a bad
`hello`: an `error` message, then the connection closes with code `1008`.
There is no partial acceptance of a malformed message.

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
