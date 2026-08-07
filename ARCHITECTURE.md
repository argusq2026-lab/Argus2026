# Argus — architecture

## Overview

Argus is a triage service for a floor of people watched one phone at a time.
Each phone runs its own on-device perception — pose, plus whatever classifier
its domain needs — and streams the numeric result over a WebSocket; the laptop
ranks who needs a human and gives the supervisor a live view.

**What that floor is doing is a parameter.** A `use_case`, agreed at handshake,
selects which parser reads a message, which scorer ranks it, and which renderer
draws it. Fitness is the use case built out far enough to demonstrate the
system end to end — it is the worked example the rest of this document keeps
returning to, not the subject of the design. Nursing (CPR) is a second, scored
entirely differently. Welding is a registered placeholder. See **A use case is
a dispatch, not a fork** below.

Three properties drive the design, and each one is enforced by structure
rather than by convention:

1. **The rank is reproducible.** Every scorer is a pure function of numeric
   history. There is no free text or non-deterministic model decode anywhere
   in the system to neutralize — where a phone classifies (fitness), it emits
   a closed-vocabulary code and the ingest server rejects anything else
   outright; where the laptop derives the fault itself (nursing), it does so
   with arithmetic over the pose. Verified by two independent replays over the
   same observation history producing byte-identical output.
2. **Frames cannot leak, because none ever exist past the phone's own camera
   pipeline.** The laptop never opens a camera, never receives a pixel, and
   the modules on its alert boundary import no image library and cannot even
   name an image type. Verified by inspecting imports and type annotations.
   This is also what makes a **private, proprietary on-device model** a
   practical thing for a new use case to bring: it runs on the phone, and the
   laptop only ever sees the numbers its parser defines.
3. **Nothing degrades quietly.** A phone whose protocol version doesn't match
   is rejected, not downgraded. A phone whose `use_case` doesn't match the
   floor's is rejected rather than admitted and scored by nothing. A
   `form_reason_codes` entry outside the configured vocabulary closes the
   connection rather than scoring as zero. A config with an unknown key
   raises. Two phones claiming the same `trainee_id` is a rejected collision,
   not a silent overwrite.

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

Those five features are now understood as *fitness's* scorer rather than *the*
scorer — the contract they satisfy is what generalised, not the features
themselves.

---

## System diagram

```
  phone (station 1)         phone (station 2)         phone (station N)
  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
  │ camera             │     │ camera             │     │ camera             │
  │  -> on-device pose  │    │  -> on-device pose  │    │  -> on-device pose  │
  │  -> the use case's  │    │  -> the use case's  │    │  -> the use case's  │
  │     own classifier, │    │     own classifier, │    │     own classifier, │
  │     if it has one   │    │     if it has one   │    │     if it has one   │
  └──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
             │ WebSocket: hello{use_case}, then           │          │
             │ observation{use_case, ts, ...that use case's own fields}
             │                — numeric only, no frame —  │          │
             ▼                            ▼                          ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  argus.ingest.server — one connection per station                      │
  │  argus.ingest.protocol: _OBSERVATION_PARSERS[use_case] -> validate,    │
  │  reject on unknown use case, version mismatch, or mid-stream switch    │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │ push FrameObservation
                                  ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  argus.ingest.session — one TrackState per trainee_id                  │
  │  history sized by history_len_for(use_case)                            │
  │  disconnect grace window (ingest.track_ttl_s), then evicted            │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  every ingest.rank_interval_s
                                  ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  argus.triage — deterministic scorers, pure, stdlib only               │
  │  _SCORERS[use_case] -> one score + reason codes                        │
  │    fitness: fall/stillness/occlusion/off_task/form_error, weighted     │
  │    nursing: dispatch again on `procedure`; cpr = worst single fault    │
  │    welding: 0.0, always — a registered placeholder                     │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  merged rank across every connected station
        ══════════════════════════╪══════════════════════════  ALERT BOUNDARY
                                  │  only {trainee_id, score, reason_codes, ts}
             ┌──────────┬─────────┴────────┬──────────────┐
             ▼          ▼                  ▼              ▼
      stderr alerts  JSON lines     HTTP /triage       console (/)
                                                          ▲
                                                          │ also reads the live
                                                          │ numeric observation
                                                          │ via GET /console —
                                                          │ a skeleton cannot be
                                                          │ drawn from four
                                                          │ scalars. Drawing is
                                                          │ RENDERERS[use_case].
                                                          │ Still no frame, still
                                                          │ closed, loopback-bound.
                                                          │ See docs/CONSOLE.md.
```

---

## Components

| Component | Module | Responsibility |
|---|---|---|
| Config | `argus.config` | Versioned TOML; every tunable, validated on load — including `[session] use_case` against what this build can score |
| Scorers | `argus.triage` | Pure functions over numeric history, one per use case behind `_SCORERS`. stdlib only. |
| Synthetic fixture | `argus.synthetic` | The fitness scene `argus demo` + tests replay — no phone needed |
| Wire protocol | `argus.ingest.protocol` | Per-use-case message validation behind `_OBSERVATION_PARSERS`; closed-vocabulary and version enforcement |
| Session registry | `argus.ingest.session` | One `TrackState` per station; history sized per use case; disconnect grace window; TTL eviction |
| Admission | `argus.ingest.admission` | Who is allowed onto the floor: auto by default, or held for the supervisor to approve |
| Ingest server | `argus.ingest.server` | WebSocket listener + the periodic rank tick |
| Sinks | `argus.outputs`, `argus.alerts` | The alert boundary: stderr alerts, JSON lines, HTTP, and the console's snapshot |
| Console | `argus.console` | The page at `GET /`, and `RENDERERS` — one drawing function per use case. Static, no build step, reads one endpoint |
| LAN discovery | `argus.discovery` | The beacon that lets a phone find this laptop without being told, and check it agrees on the use case |
| CLI | `argus.cli` | `run`, `replay`, `discover`, `doctor`, `config`, `demo`; no arguments means `run --open` |

---

## Design decisions

### A use case is a dispatch, not a fork

The five features above were written against a gym floor, and the temptation
when a second domain arrives is to make them cover it — add optional fields to
the observation, add a branch inside `compute_triage`, weight the features that
still sort of apply. That produces one function that is subtly wrong for every
domain and auditable for none.

Instead `use_case` is agreed once at `hello` and selects an implementation at
each layer that has a domain-specific answer:

| Registry | Module | What a new use case supplies |
|---|---|---|
| `_OBSERVATION_PARSERS` | `argus.ingest.protocol` | Its own message body, validated field by field |
| `_SCORERS` | `argus.triage` | Its own scorer: evidence in, one score plus reason codes out |
| `history_len_for` | `argus.triage` | How much history its evidence needs |
| `RENDERERS` | `argus.console` | Its own station drawing, or the placeholder |
| `known_use_cases()` | derived from `_SCORERS` | Nothing — it is read, not written, so the config validator, the console dropdown and `POST /session/use_case` cannot drift from what is actually scoreable |

Nursing is the proof this is a real seam rather than three names for one
shape. It shares fitness's pose fields — a COCO-17 pose is genuinely the same
measurement — and shares none of the rest: no `exercise`, no `rep_count`, no
`form_reason_codes`, because a nursing fault is derived on the laptop from the
movement rather than classified on the phone against a vocabulary. Its history
is 512 frames against fitness's 30, because a rhythm cannot be read through a
window shorter than the rhythm, and its scorer takes the worst single fault
rather than a weighted sum, because averaging a dangerously slow compression
rate against well-locked elbows produces a number an instructor cannot act on.
None of that required touching fitness.

Three consequences are deliberate:

* **A phone cannot be admitted onto a floor running something else.** `hello`'s
  `use_case` is checked against `[session] use_case` strictly — including when
  the phone omits it, since omitting it still means `"fitness"`. Every
  subsequent `observation` is checked against the same value, so a connection
  is one use case for its whole life. Admitting a mismatched phone would
  silently score it by nothing, which looks identical to a calm station.
* **An unregistered use case is refused before any field is read.** Nothing
  tries to pull `bbox_xyxy` out of a nursing message and fail confusingly three
  fields in.
* **The laptop still loads no model.** Where a use case needs a classifier, it
  runs on the phone. A domain arriving with a private, proprietary model ships
  it inside a phone build and defines what numbers cross the wire; the laptop's
  half stays pure Python with no inference dependency, and the privacy property
  (§2 above) holds for the new use case for free rather than by re-argument.

### `welding` is registered, and does nothing, on purpose

There is no welding classifier and no welding data. Rather than leave the
extension point untested until a real second domain arrived, `welding` is
registered with a parser that validates only the shared envelope plus an opaque
`payload`, a scorer that returns `0.0` without reading it, and a renderer that
clears the canvas. What that proves is the whole *non-scoring* lifecycle — a
station can connect, stream, be ranked among the others, appear on the console,
go silent, and be evicted — with nothing asserted about anyone's technique.

A real welding use case replaces `_parse_welding_observation` and
`compute_triage_welding` with functions naming actual fields. It does not
extend `payload` indefinitely; the placeholder is scaffolding, and the shape of
the scaffolding is not the shape of the building.

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

### The form-error feature is a lookup, not a match *(fitness)*

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

### A feature can be wrong for an exercise, so weights are per-exercise

*Within* the fitness use case, the same "a default is an assumption" problem
recurs one level down, and gets a smaller version of the same answer.

The five features were written against a floor of standing, high-intensity
movements, where "horizontal" and "not moving" are both good evidence something
has gone wrong. Adding a plank classifier surfaced that this is an assumption rather
than a fact: a correct plank is horizontal *and* motionless *and* oriented
away from the station-facing reference, so `fall`, `stillness`, and `off_task`
all read a textbook rep as trouble. Scored on the default weights, a correct
plank reached 0.42 against a 0.5 alert threshold and reported
`prolonged_stillness, off_task_orientation`.

The fix is not a special case inside the scorer. `docs/PROTOCOL.md` already
carried an `exercise` field, so `compute_triage` looks up a whole weight
vector for the trainee's current exercise
(`ScoringConfig.weights_for`) and `[scoring.exercise_weights.plank]` names one
where `fall`, `stillness`, and `off_task` are 0 and `form_error` carries 0.85.
Occlusion survives at its default weight, because a trainee the phone cannot
see is worth flagging whatever they are doing.

Two properties fall out of doing it this way rather than with a suppression
list:

* **A profile is a complete vector, held to the same contract as the default** —
  every feature named, non-negative, summing to 1.0. So "what does a plank
  actually score on" has one answer in one place, and a score stays comparable
  against `alert_threshold` without renormalisation.
* **Zero weight also suppresses the reason code.** One number decides both
  whether a feature contributes and whether it may explain the result, so the
  dashboard cannot report `prolonged_stillness` for a score that contains no
  stillness term. A reason that explains no part of the score is worse than no
  reason.

`exercise` stays deliberately open, unlike `form_reason_codes`: an exercise
with no profile scores on the defaults rather than being rejected, because it
is a free-form label and always has been. The closed vocabulary is where
version skew must be caught; the exercise label is not.

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
`docs/PROTOCOL.md`. The fixture is a **fitness** scene — it exercises the
ingest -> triage -> alert path and fitness's scorer; nursing and welding are
driven by a phone or a hand-written client today.

## The phone side

The phone-side app — pose estimation, per-use-case classification, and the
WebSocket client half of `docs/PROTOCOL.md` — lives in `android/` in this
same repository; see `android/README.md` for what it runs and how it stages
onto a device. `DashboardActivity` is the launcher and picks the station:
`MainActivity` is the fitness station (YOLO-X + YOLO26-pose detection, three
form classifiers, geometric fault checks, rep counting), `NursingActivity` is
the CPR station (the same detection and pose, no phone-side classifier, sending
`procedure` instead of an exercise). The two are separate Activities on
purpose — threading a use case through one screen welded to an exercise picker
and a rep counter would make one long file answer two questions — and share
everything that was already its own class: the model store, the detectors, the
subject tracker, the overlay, and the ingest client.

Both have been built, installed, and run on a real phone against a real laptop
server, over both USB and LAN Wi-Fi.

## What is not yet validated

What that device testing does *not* establish is accuracy against a real
person: every fitness model's reported number is a held-out-frame figure from
upstream's own source recordings, the fitness weights have never been fitted to
a real incident, and no real CPR has been measured — the rate estimator is
tested against synthetic waveforms at known rates, which tests the arithmetic
and not a camera. See `docs/VALIDATION.md` for the full, itemized list of what
remains unvalidated and what closing each gap would take.
