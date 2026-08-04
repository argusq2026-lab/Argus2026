# Argus — architecture

## Overview

Argus is a triage service for a HIIT floor with one phone in front of every
trainee. Each phone runs its own on-device pose model and form/exercise
classifier and streams the numeric result over a WebSocket; the laptop ranks
who needs a human instructor and gives the trainer a live view.

Three properties drive the design, and each one is enforced by structure
rather than by convention:

1. **The rank is reproducible.** The scorer is a pure function of numeric
   history. There is no free text or non-deterministic model decode anywhere
   in the system to neutralize — a phone's on-device classifier already
   emits a closed-vocabulary code, and the ingest server rejects anything
   else outright. Verified by two independent replays over the same
   observation history producing byte-identical output.
2. **Frames cannot leak, because none ever exist past the phone's own camera
   pipeline.** The laptop never opens a camera, never receives a pixel, and
   the modules on its alert boundary import no image library and cannot even
   name an image type. Verified by inspecting imports and type annotations.
3. **Nothing degrades quietly.** A phone whose protocol version doesn't match
   is rejected, not downgraded. A `form_reason_codes` entry outside the
   configured vocabulary closes the connection rather than scoring as zero.
   A config with an unknown key raises. Two phones claiming the same
   `trainee_id` is a rejected collision, not a silent overwrite.

## What changed from the single-machine design

The previous iteration of Argus ran entirely on one Windows AI PC: USB/IP
cameras watched a room, a Snapdragon X Elite NPU ran YOLO-X detection,
Kalman + appearance tracking, and BlazePose pose estimation locally, and a
VLM sampled flagged trainees. That premise assumed *one camera, many
trainees*, which is why most of its complexity — multi-object detection,
re-identification, an NPU inference stack, a VLM captioning gate — existed
at all.

The product's premise is now *one phone, one trainee*. That collapses most
of that complexity:

* **No detector.** There is nothing to find in a frame the laptop never
  receives — the phone already knows which person it's pointed at.
* **No re-identification.** A trainee's identity is their phone's own
  connection, not a re-associated track across frames. There is no "which
  blob is which person" problem to solve on the laptop anymore.
* **No local model runtime.** Pose estimation and form/exercise
  classification run on the phone. The laptop has no NPU workload, no ONNX
  Runtime, no QNN — it is pure Python, no compiled inference dependency at
  all.
* **No VLM gate.** The old design sampled a VLM caption for already-flagged
  trainees and matched it against a closed vocabulary, because that was the
  only way to get a semantic anomaly signal without an unbounded latency
  cost. A phone's own classifier already *is* that signal, computed for
  free as part of pose estimation, with no sampling gate needed.

What survives untouched is the deterministic scorer itself
(`argus.triage`): fall, stillness, occlusion, and off-task orientation are
still pure functions of bounding box and COCO-17 keypoint history, which a
phone now supplies instead of a local vision stack. The fifth feature is
renamed rather than replaced — `vlm_anomaly` (scored by matching a VLM
caption against a vocabulary) becomes `form_error` (scored by a direct
lookup of the phone's own closed-vocabulary reason codes) — but the
"numeric history in, one explainable score out" contract is identical.

---

## System diagram

```
  phone (trainee 1)         phone (trainee 2)         phone (trainee N)
  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
  │ camera             │     │ camera             │     │ camera             │
  │  -> on-device pose  │    │  -> on-device pose  │    │  -> on-device pose  │
  │  -> form/exercise   │    │  -> form/exercise   │    │  -> form/exercise   │
  │     classifier      │    │     classifier      │    │     classifier      │
  └──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
             │ WebSocket: hello, then     │                          │
             │ observation{ts, bbox,      │                          │
             │ keypoints, form_reason_codes} — numeric only, no frame
             ▼                            ▼                          ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  argus.ingest.server — one connection per trainee                      │
  │  parse + validate (argus.ingest.protocol) -> reject on mismatch        │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │ push FrameObservation
                                  ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  argus.ingest.session — one TrackState per trainee_id                  │
  │  disconnect grace window (ingest.track_ttl_s), then evicted            │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  every ingest.rank_interval_s
                                  ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  argus.triage — deterministic scorer, pure, stdlib only                │
  │  fall / stillness / occlusion / off_task / form_error -> one score     │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  merged rank across every connected trainee
        ══════════════════════════╪══════════════════════════  ALERT BOUNDARY
                                  │  only {trainee_id, score, reason_codes, ts}
             ┌──────────┬─────────┴────────┬──────────────┐
             ▼          ▼                  ▼              ▼
         console    JSON lines      HTTP /triage    trainer dashboard (/)
```

---

## Components

| Component | Module | Responsibility |
|---|---|---|
| Config | `argus.config` | Versioned TOML; every tunable, validated on load |
| Scorer | `argus.triage` | Pure functions over numeric history. stdlib only. |
| Synthetic fixture | `argus.synthetic` | The scene `argus demo` + tests replay — no phone needed |
| Wire protocol | `argus.ingest.protocol` | Message validation; closed-vocabulary and version enforcement |
| Session registry | `argus.ingest.session` | One `TrackState` per trainee; disconnect grace window; TTL eviction |
| Ingest server | `argus.ingest.server` | WebSocket listener + the periodic rank tick |
| Sinks | `argus.outputs`, `argus.alerts` | The alert boundary: console, JSON lines, HTTP, dashboard |
| CLI | `argus.cli` | `run`, `doctor`, `config`, `demo` |

---

## Design decisions

### Identity is the connection, not a re-association

`trainee_id` is asserted by the phone at handshake and is the triage key an
instructor is dispatched against. There is no tracker to fool: a phone that
claims a `trainee_id` already live is rejected outright
(`DuplicateTraineeError`), rather than silently taking over that trainee's
history — the network-era analogue of the old tracker's identity-swap
concern, solved by a stricter mechanism (a connection either is or isn't
already claiming an id) instead of a probabilistic one (motion + appearance
cost).

### Disconnection is not the same as leaving the floor

A phone drops Wi-Fi for a few seconds; a trainee steps away and comes back.
Deleting a trainee's `TrackState` the instant a socket closes would reset
their rolling history and re-arm every alert-suppression state on
reconnect — indistinguishable, from the dashboard, from a new trainee
walking up. `argus.ingest.session.SessionRegistry` instead keeps a
disconnected session alive for `ingest.track_ttl_s` (default 10 s); a
reconnecting `trainee_id` within that window resumes the same track. Only
silence longer than the TTL — connected or not — gets a session evicted.

### The form-error feature is a lookup, not a match

The prototype-derived design scored a VLM's free-text caption by testing
whether it contained one of nine fixed phrases, because a VLM's decoding is
not itself deterministic and the vocabulary match is what made the rank
reproducible anyway. A phone's on-device classifier has no such
non-determinism to neutralize — it already emits the closed-vocabulary code
directly. `argus.triage.score_form_codes` is therefore a plain dict lookup,
and the real enforcement moved to the boundary where it belongs:
`argus.ingest.protocol.parse_observation` rejects any code outside
`[scoring.form_error_vocab]` before it ever reaches the scorer, since an
unrecognised code from a phone is a version-skew bug, not an open-ended
observation to score as best it can.

### Privacy by wiring, now with less to guard

`TriageRecord` is still frozen with exactly four scalar fields, and
`argus.outputs`/`argus.alerts` still import no image library — unchanged
from the previous design, and still checked by
`tests/test_privacy.py` parsing module ASTs and inspecting type hints. What's
different is that the boundary now has less to hold back: no camera frame is
ever captured on the laptop in the first place, so there is no image-typed
value anywhere in `argus.ingest` either, which the privacy test now also
covers.

### Nothing degrades quietly, extended to the network

The old design's "raises rather than falls back" posture (an NPU engine that
can't load, a config with a typo'd key) extends naturally to a network
boundary: a protocol version mismatch, a malformed message, an unrecognised
form-error code, or a colliding `trainee_id` all close the connection with an
explicit `error` message rather than being coerced into "best effort"
behavior. See `docs/PROTOCOL.md`.

### Deterministic clock, revisited

The old design distinguished a `FrameClock` (reproducible replay over a file)
from a `WallClock` (live cameras). Every source is now live by construction —
a phone's own connection — so the ingest server always uses a real clock for
the merged rank's timestamp. Reproducibility is instead proven by replaying
the same *sequence of received messages* through the scoring path twice
(`tests/test_determinism.py`) and diffing the output — the network layer is
just framing around a function that was always pure.

---

## Deployment

`run.ps1` creates one plain venv — there is no split anymore, because there
is no OpenCV/NPU dependency to keep out of the app's own environment. Argus
runs on any host architecture that runs Python 3.11+.

`argus demo` generates a canned multi-station observation fixture with no
phone, model, or camera involved; `demo/replay_client.py` streams it over a
real WebSocket connection to a running `argus run`, which is also the
reference example a phone-app implementer should read alongside
`docs/PROTOCOL.md`.

## What is not yet built

The phone-side app (pose estimation, form/exercise classification, and the
WebSocket client half of `docs/PROTOCOL.md`) is a separate, future project
and does not exist in this repository. See `docs/VALIDATION.md` for the full
list of what that gap means in practice.
