# Handoff: the host side — trainer console and live phone↔laptop operation

Written against `main` at `e4e4b7c`. Every claim here was checked against the
tree rather than recalled; where something is unverified it says so.

Read in this order: [`docs/PROTOCOL.md`](PROTOCOL.md) — the contract the phone
already speaks — then `src/argus/ingest/`, then
[`ARCHITECTURE.md`](../ARCHITECTURE.md) and
[`docs/VALIDATION.md`](VALIDATION.md).

---

## 1. This is not a greenfield

The laptop already accepts connections, validates them, scores them, and serves
the result. Read the code before writing any: the most likely way to waste a day
here is to rebuild something that works.

| Module | What it already does |
|---|---|
| `argus.ingest.server` | `IngestServer`, one WebSocket connection per trainee; `hello` → `hello_ack` → `observation` stream |
| `argus.ingest.protocol` | `parse_hello` / `parse_observation`. Strict: rejects a mismatched `protocol_version` or a `form_reason_code` outside the configured vocabulary rather than scoring it as zero |
| `argus.ingest.session` | `SessionRegistry`, `StationSession`. Per-trainee rolling history, plus a `track_ttl_s` grace window so a brief Wi-Fi drop does not reset a trainee's triage state |
| `argus.triage` | The deterministic scorer. Five features: fall, stillness, occlusion, off-task, `form_error` |
| `argus.outputs` | `TriageHTTPServer` — `GET /triage` (JSON), `GET /healthz`, `GET /` (dashboard); `JsonLogSink` |
| `demo/replay_client.py` | A reference phone. `--fixture --speed --ws-host --ws-port`. Exercises the whole path with no device involved |
| CLI | `argus run`, `argus demo`, `argus doctor`, `argus config` |

**Already verified end to end**: a real phone connected over `adb reverse` and
appeared in `GET /triage` as a scored record. The plumbing is proven. What is
missing sits above it.

---

## 2. What is actually missing

**The dashboard is a stub.** `_DASHBOARD_HTML` in `outputs.py` is 46 lines: it
polls `/triage` with `fetch` and `setInterval` and renders a list. It has no
keypoint rendering, no per-station view, no history, and no staleness
indication. This is the main build.

**`form_error` contributes nothing.** The phone always sends
`form_reason_codes: []`, so a feature weighted 0.15 is inert and the rank is
driven entirely by pose and motion. The vocabulary in
`[scoring.form_error_vocab]` is squat-centric (`insufficient_depth`,
`knee_valgus`, `heels_rising`, …) and needs an on-device classifier — that is
edge work, but build the console so the codes display the moment they arrive.

**Three protocol fields are accepted and ignored**: `exercise`, `rep_count`,
`form_ok`. `parse_observation` validates them; nothing surfaces them.

**Nothing has ever run over real Wi-Fi.** Every phone↔laptop session so far went
through `adb reverse` over USB. The phone's reconnect-with-backoff is therefore
unexercised against real LAN behaviour — roaming, NAT, sleep, packet loss.

---

## 3. What the phone sends today

Per `docs/PROTOCOL.md`, at roughly 5–15 Hz, one message per frame:

```json
{
  "type": "observation",
  "ts": 1730649600.125,
  "bbox_xyxy": [0.12, 0.08, 0.61, 0.97],
  "keypoints_xy": [[0.34, 0.10], "… 17 pairs"],
  "keypoints_conf": [0.91, "… 17 values"],
  "form_reason_codes": []
}
```

Coordinates are **normalized to [0, 1]** of the phone's own frame, so a 1080p
and a 4K phone report the same numbers for the same framing — the console can
render them without knowing anything about the camera.

The keypoints are real: the phone runs YOLO26-pose single-stage on its NPU at
~22 ms/frame and emits full COCO-17 **including knees and ankles**. Left and
right are the *subject's*, not the image's.

Only the subject is reported. One connection is one trainee; an observation has
exactly one `bbox_xyxy` and one keypoint set. Rendering several people per
station would be a protocol change, not a console change.

---

## 4. Non-negotiables

These are why the codebase is trustworthy. Breaking one is a bigger decision
than it looks.

1. **The privacy boundary is structural, not a filter.** `tests/test_privacy.py`
   asserts by AST and type-hint inspection that boundary modules import no image
   library and that no public callable names an image-capable type;
   `TriageRecord` is a closed four-field set. A console that draws skeletons must
   do it from the numeric observation. Do not widen `TriageRecord`, and do not
   add a frame path to make rendering easier.
2. **Nothing degrades quietly.** A station that has gone silent must be visibly
   stale. A quiet station and a calm one produce the same empty rank, and the
   console is the only place that difference can be shown.
3. **The rank is deterministic.** `tests/test_determinism.py` is a required CI
   job. The scorer is a pure function of numeric history; keep it one.
4. **Config over code.** Every tunable lives in `configs/argus.default.toml`
   behind a `config_version` bump. Nothing in `src/argus/` hardcodes a weight,
   threshold, or cadence.

---

## 5. Gotchas that cost real time

- **`ws_host` defaults to `0.0.0.0`** so phones on the LAN can reach it.
  `argus doctor` prints the LAN-reachable address; `127.0.0.1` works only from
  the laptop itself.
- **Server-start races are real and they hide.** A test that `Popen`s the server
  and connects immediately passes locally and fails in CI with `ECONNREFUSED` —
  which reads like a broken server rather than a client that started too early.
  Poll the port. A fixed `sleep` is the same race with a longer fuse. The
  pattern is `_wait_until_listening` in `tests/test_cli.py`.
- **Phone and laptop clocks are not synchronised.** `ts` comes from the phone's
  own clock and no cross-device sync is assumed anywhere. Do not build
  cross-station logic that depends on a shared time base.
- **`trainee_id` is exclusive.** A second `hello` claiming a connected id is
  rejected, not merged.
- **Reconnection is a fresh `hello`.** There is no resume handshake; continuity
  comes from the server's `track_ttl_s` window, not from anything the client
  keeps.

---

## 6. Suggested first moves

1. **Get the loop running without a phone.** `argus run` in one terminal,
   `demo/replay_client.py` in another, `/triage` in a browser. That is the
   development loop, and it needs no device.
2. **Build the console properly.** Per-station cards; a live skeleton drawn from
   `keypoints_xy`; the ranked help queue as the primary view; **visible
   staleness**; and reason codes rendered as prose rather than raw enum strings —
   `possible_fall` is a value, not a sentence a trainer should read.
3. **Surface `exercise` / `rep_count` / `form_ok`.** The plumbing exists and
   nothing displays them.
4. **Then test against the real phone over Wi-Fi**, not `adb reverse`. That is
   the first time the reconnect path is exercised for real.

---

## 7. What is unresolved, and should stay stated

- **The scoring weights have never been fitted to an incident**
  ([VALIDATION.md](VALIDATION.md) §2). The rank is structurally sound and
  empirically unvalidated. A console that presents it as tuned would be
  overclaiming.
- **No accuracy claim is supported.** No real trainee footage exists (§1), so
  neither detection nor pose has a measured accuracy on this task.
- **Scope.** This is a hackathon build; see the note at the top of
  [`README.md`](../README.md).
- **An AGPL-3.0 weights file remains in git history** at `3071f26`, untracked at
  HEAD. Irrelevant to host-side work, relevant before anything ships — see
  [`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md).

---

## 8. The edge side, for context

Not your build, but worth knowing what is on the other end of the socket. See
[`android/README.md`](../android/README.md).

The station runs YOLO26-pose on the Snapdragon 8 Elite's Hexagon NPU, holds the
overlay ~400 ms across blurred frames, picks its subject by largest box with
hysteresis (so a passer-by does not steal the trainee), reconnects with capped
backoff on transport drops but **not** on protocol refusals, and keeps
diagnostics behind a Debug toggle. Its model artifacts are reproducible
byte-identically from a clone via `scripts/fetch_edge_models.py`.
