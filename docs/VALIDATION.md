# Validation gaps

What Argus has **not** been shown to do. Each entry says what is unverified,
why it matters, and what closing it would take. Nothing here is a known bug
— these are claims the product cannot currently make.

Ordered by how much a wrong assumption would cost.

---

## 1. The phone app does not exist yet

**Status:** blocking for everything else on this list.

This repository is the laptop side only. [`docs/PROTOCOL.md`](PROTOCOL.md)
specifies exactly what a phone app must send, and
[`demo/replay_client.py`](../demo/replay_client.py) is a reference
implementation of the client half of that protocol — but no on-device pose
model, no form/exercise classifier, and no real camera has ever driven this
server. Every test in this repository exercises the ingest -> triage -> alert
path against a synthetic observation fixture (`argus.synthetic`), not a real
trainee.

**To close:** build the phone app against `docs/PROTOCOL.md`; pick and
validate an on-device pose model and a form/exercise classifier; run it
against real trainees and confirm the wire contract holds up (frame rate,
keypoint layout, timestamp behaviour) under a real Wi-Fi network, not a
loopback socket in a test.

**Owner decision needed:** which on-device pose model and form-classifier
architecture, and which specific form errors the closed vocabulary in
`[scoring.form_error_vocab]` should actually cover for the exercises this
floor runs — the seven entries shipped in the default config are
illustrative, not the result of any domain review.

---

## 2. The five scoring weights have never been fitted to an incident

**Status:** blocking for operational trust in the rank.

`fall 0.40 / stillness 0.20 / occlusion 0.15 / form_error 0.15 / off_task
0.10` are the prototype author's priors, carried forward unchanged through
the move to phone-based ingestion. No incident has ever been scored with
them. `alert_threshold = 0.5` is equally unvalidated: nobody has measured how
many alerts an instructor would receive per hour, or what fraction would be
real.

The weights live in `configs/argus.default.toml` rather than in code, so
retuning is a config edit and a `config_version` bump — the mechanism is
ready, the evidence is not.

**To close:** build a labelled-clip harness once real phone data exists: a
corpus of observation streams each labelled with what happened and whether an
instructor was in fact needed, a replay path that emits the full per-feature
breakdown (not just the final score), and a fitting step that searches the
weight simplex against a chosen operating point.

**Owner decision needed:** what counts as "needed an instructor" — the label
definition is the experiment, and it cannot be inferred from the code.

---

## 3. The WebSocket ingest server has no authentication or transport security

**Status:** open risk for any deployment beyond a trusted private LAN.

`ws://` is plaintext, and any device that can reach `ingest.ws_port` can send
a `hello` claiming *any* `trainee_id` — there is no credential binding a
specific phone to a specific trainee identity, only the "first connection to
claim an id wins, and it's exclusive while connected" rule in
`argus.ingest.session`. That rule stops two simultaneous claims from
colliding silently; it does not stop a misconfigured or malicious device on
the same network from impersonating a trainee it has no relationship to.

**To close:** at minimum, run this on a network the gym controls (not open
Wi-Fi); for anything beyond that, add `wss://` (TLS) and a per-phone
credential (e.g. a token issued at trainee check-in) checked at `hello`
time — the protocol has a natural extension point there, since `hello`
already carries identity claims.

---

## 4. No cross-phone clock synchronization

**Status:** open risk for cross-trainee timing comparisons.

Each `observation` message's `ts` is the phone's own clock
(`docs/PROTOCOL.md`); nothing enforces or measures agreement between two
phones' clocks. The merged rank's own timestamp comes from the laptop's
clock at each rank tick, so *ranking* is unaffected — but any feature that
someday compared two trainees' event timing directly (e.g. "did they start
their rep within N ms of each other") would inherit whatever clock skew
exists between their phones, unmeasured.

**To close:** if a feature ever needs cross-trainee timing precision, add an
NTP-style offset estimate to the handshake and either correct or flag skew
beyond a threshold.

---

## 5. Ingest capacity under many concurrent phones is unmeasured

**Status:** blocking for a capacity claim ("how many trainees per laptop").

The ingest server is a single asyncio event loop; `IngestServer.tick()`
recomputes the full merged rank every `ingest.rank_interval_s` over every
connected session. Nothing about this has been load-tested: not the number
of concurrent WebSocket connections one process can hold open, not the CPU
cost of `rank_trainees` at real class sizes, not behaviour when a rank tick
takes longer than `rank_interval_s` to compute (there is no back-pressure or
skip-if-still-running guard on the periodic task today).

**To close:** run a synthetic load test — many concurrent
`demo/replay_client.py`-style connections streaming at a realistic rate —
and measure rank-tick latency and memory as trainee count grows; add a
guard against overlapping ticks if it becomes an issue in practice.

---

## 6. No accuracy figure exists for pose or form/exercise classification

**Status:** inherited from §1 — there is no model to measure yet.

Whatever on-device pose model and form classifier the phone app eventually
uses will need its own accuracy validation (keypoint error, exercise
classification precision/recall, false-positive rate on
`form_reason_codes`) against real trainee footage before any claim about
correctness can be made. None of that exists today because the model choice
itself hasn't been made.

**Owner decision needed:** same as §1 — this is one gap, not two, restated
here because it is the reason §2's weight-fitting work can't start yet
either: there is no real per-feature signal to fit weights against until a
real classifier exists.
