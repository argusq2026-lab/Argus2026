# Adding a use case

How to point Argus at a domain it has never watched — a ward, a bay, a bench —
without touching the domain it already watches.

Argus's engine is a phone-per-subject ingest path, a deterministic ranker, and
a console that is honest about silence. None of that is fitness. Fitness is the
use case built out far enough to prove the engine works end to end; nursing is
the second, and it shares almost nothing with the first except a pose. This
document is the runbook for the third.

Two sibling runbooks, so you start in the right one:

| You want to | Read |
|---|---|
| Watch a **new domain** — its own evidence, its own faults, its own scorer | this document |
| Add another **exercise** to the existing fitness use case | [`ADDING_AN_EXERCISE.md`](ADDING_AN_EXERCISE.md) |
| Know exactly what goes on the wire | [`PROTOCOL.md`](PROTOCOL.md) |

---

## 1. The shape of the thing

A use case is a string agreed once at `hello`, and five registry entries it
selects. There is no `if use_case == ...` anywhere in the pipeline; adding one
means adding entries, not branches.

```
  [session] use_case = "<yours>"            ← the laptop's config
        │                                     validated at load against
        │                                     triage.known_use_cases()
        ▼
  hello{use_case} ──► rejected unless it matches                [handshake]
        │
        ▼
  observation{use_case, ...} ──► _OBSERVATION_PARSERS[use_case] [every frame]
        │                        argus/ingest/protocol.py
        │                        rejects an unregistered use case, and any
        │                        message switching use case mid-connection
        ▼
  FrameObservation ──► TrackState(history_len_for(use_case))    [session]
        │              argus/ingest/session.py
        ▼
  _SCORERS[use_case] ──► TriageRecord{score, reason_codes}      [every tick]
        │               argus/triage.py
        ▼
  merged rank ──► sinks, and RENDERERS[use_case] on the console
                  argus/console.py
```

| Concern | Where |
|---|---|
| What your phone sends, and how it is validated | `_OBSERVATION_PARSERS` in [`src/argus/ingest/protocol.py`](../src/argus/ingest/protocol.py) |
| How that becomes one score plus reason codes | `_SCORERS` in [`src/argus/triage.py`](../src/argus/triage.py) |
| How much history your evidence needs | `history_len_for` in the same file |
| What a station card draws | `RENDERERS` in [`src/argus/console.py`](../src/argus/console.py) |
| What the config, the dropdown and `POST /session/use_case` accept | nothing — `known_use_cases()` reads `_SCORERS` |
| Reason-code prose on the console | `PROSE` in `src/argus/console.py` |
| Any tunables your scorer needs | `[scoring]` in [`configs/argus.default.toml`](../configs/argus.default.toml), plus a `config_version` bump |
| The phone station | a new Activity in [`android/`](../android/), alongside `MainActivity` (fitness) and `NursingActivity` |

**What you do not touch:** the session registry, admission, discovery, the
alert boundary and its sinks, the CLI, the binary build, or any other use
case's parser and scorer. If you find yourself editing
`_parse_fitness_observation` or `compute_triage_fitness`, stop — that is the
mistake this structure exists to prevent.

---

## 2. Bring your own model

The single most useful property here is one you get for free: **the laptop
loads no model.** It has no NPU workload, no ONNX Runtime, no inference
dependency at all — it is pure Python over numbers.

So a domain arriving with a private, proprietary classifier does not have to
disclose it, export it, or hand it to this repository. It ships inside a phone
build, runs on the phone's own NPU, and the only thing that becomes public is
the *shape of the numbers it emits* — the parser you write in step 4. Nursing
takes the other option and ships no phone-side classifier at all, deriving its
fault on the laptop from the pose. Both are supported; the choice is yours, and
it is the first thing to decide.

| Where the judgement happens | Wire carries | Example | Trade |
|---|---|---|---|
| **On the phone** | a verdict from a closed vocabulary | fitness's `form_reason_codes` | Your model stays private. The rank depends on a device-side value nobody downstream can audit, so the vocabulary must be closed and version-checked. |
| **On the laptop** | raw numeric evidence | nursing's pose, scored into a compression rate | Fully auditable and replayable from the recorded stream. Everything you need must survive as numbers on the wire. |

The privacy property holds either way: no frame ever exists past the phone's
own camera pipeline, and `tests/test_privacy.py` asserts by AST that no module
on the boundary can even name an image type. A new use case inherits that
rather than having to re-argue it — provided your parser carries numbers and
not pixels, which the same test will hold you to.

---

## 3. The recipe

### Step 1 — Decide what the evidence actually is

Before any code, answer two questions in writing:

1. **What does a fault look like, in numbers a phone can produce?** Not "bad
   technique" — the specific measurable thing.
2. **What can this camera honestly not measure?** Write it down now, while it
   is a design note rather than a retraction. CPR's is instructive: the AHA
   wants 5–6 cm of chest travel, and recovering that from an uncalibrated
   monocular camera needs a scale reference the frame does not contain. So
   `compute_triage_cpr` has **no depth term at all** — not a rough one, not a
   flagged one. A number that cannot be trusted is worse than a missing one,
   because a missing number is visibly missing.

### Step 2 — Pick the string

Lowercase, at most 32 characters, and the same token on the phone, in the
config, and in the beacon. It is not a display label — it is an identity three
independent components compare for equality.

### Step 3 — Reuse a field only when the measurement is genuinely the same

Nursing shares fitness's `bbox_xyxy` / `keypoints_xy` / `keypoints_conf`
through `_parse_pose_fields`, because a COCO-17 pose is the same measurement in
both. It shares nothing else — no `exercise`, no `rep_count`, no
`form_reason_codes`.

The test is whether the *meaning* is identical, not whether the *type* is. Two
`float` fields that both hold "a confidence" are not the same field if one
saturates and the other does not; the fitness classifier's most expensive bug
was exactly that mistake made across two pose models
([`ADDING_AN_EXERCISE.md`](ADDING_AN_EXERCISE.md) §4 Trap 1). Copying a field
because it is convenient is how one use case inherits another's assumptions
silently.

### Step 4 — Write the parser

One function, registered in `_OBSERVATION_PARSERS`:

```python
def _parse_<yours>_observation(
    raw: Mapping[str, Any], use_case: str, form_error_vocab: Mapping[str, float]
) -> FrameObservation:
    ts = _require(raw, "ts", float)
    ...
    return FrameObservation(ts=ts, use_case=use_case, ...)
```

Rules the existing parsers follow, and the reasons:

* **Reject, do not coerce.** A wrong type, a missing field, a wrong-length
  array raises `ProtocolError`, which sends an `error` and closes the
  connection. There is no partial acceptance.
* **Bound every free-form label.** `exercise` and `procedure` are both
  length-capped: they are classifier labels, not free text, and the console
  renders them.
* **Choose deliberately between a closed vocabulary and an open label.** A
  closed vocabulary (`form_reason_codes`) catches version skew between phone
  and laptop, and *must* be closed for anything that feeds the score. An open
  label (`exercise`, `procedure`) is right when an unrecognised value should
  degrade gracefully rather than refuse a station. Nursing gets this split
  right in a way worth copying: an unimplemented `procedure` scores `0.0` and
  still shows on the console, because a ward running something this build
  cannot grade should not be turned away at the door.
* **Never read another use case's fields.** `test_fitness_fields_are_not_read_
  from_a_nursing_observation` pins this.

### Step 5 — Write the scorer

One function, registered in `_SCORERS`, with the shared signature
`(trainee_id, track, ts, cfg, reference_angle_deg) -> TriageRecord`. It must be
**pure**: same history in, byte-identical record out, no clock, no randomness,
no I/O. `tests/test_determinism.py` replays a history twice and diffs.

Two structural choices to make consciously rather than by imitation:

* **Weighted sum, or worst single fault?** Fitness sums five weighted features
  because they are commensurable evidence about one question ("is this person
  in trouble"). CPR takes the **max**, because each of its faults is
  independently a reason to send someone — a dangerously slow rate diluted by
  well-locked elbows produces a 0.4 an instructor cannot act on. If your faults
  are not exchangeable, do not average them.
* **What does "no reading" score?** `0.0`, and say why in the docstring. A
  station set up before its subject arrives, a paused cycle, an unimplemented
  procedure — all normal, none a fault. Scoring absence as trouble puts every
  idle station on the queue and teaches the supervisor to ignore it.

And one rule that came out of CPR the hard way: **refuse a measurement the
stream cannot resolve.** Below about three samples per cycle, the true period
and twice the true period become indistinguishable, so a correct 120/min reads
back as 60/min — and an instructor told "too slow" would coach someone into
making it worse. `estimate_compression_rate` computes what its own sample rate
can resolve and returns no `bpm` when the target band is outside it. If your
evidence is a rhythm, a velocity, or anything else with a sampling floor, work
out that floor, enforce it, and document it in `PROTOCOL.md` as a hard
requirement on the phone's frame rate.

### Step 6 — Size the history

`history_len_for(use_case, cfg)`. Fitness's 30 frames is ~2 s of posture.
Nursing keeps 512, because a compression rate cannot be read through a window
shorter than the rhythm — at the 28 Hz a phone actually delivers, 30 frames
*is* one second, which is one compression cycle. A scorer running on too short
a buffer looks like it is working while measuring nothing at all.

Size generously in frames and window by timestamp inside the scorer (`_cpr_
window` does), so the buffer holds enough at any plausible frame rate while the
scorer still judges a fixed amount of wall time.

### Step 7 — Add tunables to the config, not to the code

Every threshold your scorer uses goes in `[scoring]` in
`configs/argus.default.toml`, with a `config_version` bump. Nothing in
`src/argus/` may hardcode one. In the comment, separate what has an external
source from what is your own arithmetic — the CPR block does this explicitly
(the 100–120/min band is the AHA's published figure; the 40 bpm
full-deviation scale is a local choice), and that distinction is what
`VALIDATION.md` is built on.

### Step 8 — Add a renderer, or the placeholder

`RENDERERS[use_case]` in `console.py`. Nursing reuses `renderFitness` because
it streams the same skeleton; welding uses `renderPlaceholder`, which clears
the canvas, so its card reads "nothing to show" rather than "broken". Do not
teach `renderFitness` a second shape — add a function.

Add your reason codes to `PROSE` in the same file. Anything not listed there is
prettified mechanically (`cpr_rate_slow` → "Cpr rate slow"), which is fine for
a code that reads as English and not fine for jargon or an acronym.

### Step 9 — Build the phone station

A new Activity, alongside `MainActivity` (fitness) and `NursingActivity`
(CPR) — not a mode inside one of them. `NursingActivity`'s header explains the
reasoning: `MainActivity` is welded to an exercise picker, a rep counter, and a
form classifier that mean nothing to another domain, and threading a use case
through all of it makes one long file answer two questions.

What you reuse, because it was already its own class: `ModelStore`,
`QnnDetector` / `Yolo26PoseEstimator` / `PoseEstimator`, `SubjectTracker`,
`DetectionOverlayView`, `Discovery`, and `IngestClient`. What you write again
is the camera bind and the frame loop, and if your use case has no second
classifier pass it is materially smaller than fitness's.

On the wire side, `Protocol.kt` already enforces that a use case cannot set
another's fields (a nursing observation carrying a `rep_count` throws). Extend
that check rather than working around it, and add your station's tile to
`DashboardActivity`.

### Step 10 — Write the validation entry before you demo it

A section in [`VALIDATION.md`](VALIDATION.md) saying what is *not* established:
which numbers have an external source, which are your own priors, what is
deliberately not measured at all, and what closing each gap would take. §2b
(nursing) is the model to copy — it separates those four categories
explicitly, and it was written alongside the feature rather than after someone
asked.

---

## 4. What welding is for

`welding` is registered and scores nothing. Its parser validates only the
shared envelope (`type`, `ts`) plus one opaque `payload` object it carries
through uninterpreted; `compute_triage_welding` returns `0.0` without reading
`payload`; its renderer clears the canvas.

That is not an unfinished feature — it is a test of the seam. It proves the
entire **non-scoring lifecycle** works for a use case with no classifier and no
data: a station can connect, stream, be ranked among the others, appear on the
console, go silent, be drawn as silent, and be evicted. Everything except the
judgement, which is the only part that needs domain evidence.

Use it as scaffolding while your parser and scorer are being built. Then
**replace** `_parse_welding_observation` and `compute_triage_welding` with
functions that name your actual fields. Do not extend `payload` indefinitely:
an opaque blob is exactly the "degrade quietly" failure the rest of this system
is built to refuse, and it is tolerable only because nothing currently reads
it.

---

## 5. The five mistakes this structure exists to prevent

1. **Bolting optional fields onto fitness's message.** A shape that is
   `bbox + keypoints + eight fields, most null` documents nothing and validates
   nothing. Separate parsers, separate shapes.
2. **Branching inside another use case's scorer.** `fall`, `stillness` and
   `off_task` are read off a standing person's bbox and centroid. They do not
   mean "wrong" in a domain where the subject kneels, leans, or holds still on
   purpose — as the plank proved *inside* fitness
   ([`ADDING_AN_EXERCISE.md`](ADDING_AN_EXERCISE.md) §4 Trap 3).
3. **Admitting a mismatched phone.** A fitness phone on a nursing floor parses,
   streams, and scores by nothing — indistinguishable on the console from a
   calm station. Hence the strict `hello` check, including against phones that
   omit `use_case` entirely (omitting it still means fitness), and the
   per-message check that a connection cannot switch mid-stream.
4. **Reusing a history length.** See step 6.
5. **Reporting what the sensor cannot resolve.** See step 5. The failure is not
   that the number is imprecise; it is that it is confidently, actionably
   wrong.

---

## 6. Checklist

1. `_OBSERVATION_PARSERS` entry, with tests for: a well-formed message, each
   missing/malformed field, an unregistered use case, and a mid-stream switch.
2. `_SCORERS` entry, pure, with the no-evidence case scoring `0.0` and a test
   that says so.
3. `history_len_for` reviewed for your evidence, with a stated reason.
4. Config entries added, `config_version` bumped, external sources separated
   from local choices in the comment.
5. `RENDERERS` entry and `PROSE` lines.
6. `argus config` shows your use case; `argus doctor` is clean; a config naming
   an unimplemented use case still fails at load (`known_use_cases()` is
   derived, so this should need no work — confirm it).
7. The console's header dropdown offers it, and `POST /session/use_case`
   accepts it.
8. Phone station built, tile added, and a real device streams to a real laptop
   end to end.
9. `pytest tests/ -q` and `cd android && ./gradlew testDebugUnitTest` both
   clean.
10. `VALIDATION.md` section written, and `PROTOCOL.md` documents your message
    body and any frame-rate floor it requires.
