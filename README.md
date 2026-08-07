# Argus — many eyes, one mind

One instructor cannot watch twelve people at once. That is the whole problem:
wherever trainees outnumber the person responsible for them — a gym floor, a
nursing skills lab, a welding shop — the instructor's attention is the scarce
resource, and whoever is quietly struggling in the corner is exactly who does
not get it.

Argus is a triage system for that attention. One phone watches each trainee,
runs perception on its own NPU, and streams a numeric observation to a
laptop; the laptop scores every station with a **deterministic, explainable
scorer** and answers one question on the instructor's console: **who needs a
human right now, and why.** No video ever leaves any phone.

What "needs a human" means is per use case, not hardcoded: the same
pipeline dispatches on what a floor is running. Fitness is the first fully
built instance — pose triage plus on-device form classifiers for plank,
bicep curl, and lunge. Nursing is the second: a CPR station scored against
the published 100–120/min compression band, that refuses to report a rate
its camera cannot resolve. Welding is a wired-through placeholder that
connects, streams, and deliberately asserts nothing until it has a
classifier — a station that claims nothing beats one that guesses. See
[Use cases](#use-cases) and
[Bringing your own use case](#bringing-your-own-use-case).

> **Start here.** [**SETUP.md**](SETUP.md) installs everything from scratch —
> laptop and phone, every dependency named and where it comes from.
> [**USAGE.md**](USAGE.md) is how to run it, with a phone or without one.
> The laptop half is a five-minute install with a single dependency, and it
> demonstrates itself with no phone involved.

> **Scope: this is a hackathon build.** It is a working system, not a shipped
> product — the engineering is real and the measurements are real, but three
> things would need revisiting before it went anywhere near a real floor. The
> fitness scoring weights have never been fitted to an incident
> ([VALIDATION.md](docs/VALIDATION.md) §2). No pose or detection accuracy claim
> is supported, because the app has been run on real devices but has never
> watched a real subject whose activity was labelled and checked against ([§1](docs/VALIDATION.md)).
> And the phone's pose model is AGPL-3.0, which is fine to demo and develop
> against but is a licensing decision before distributing an application
> built on it — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), where
> the permissively licensed alternatives are already scouted.

**Privacy is a property of the wiring.** No frame ever exists past a phone's
own camera pipeline, and every sink's signature accepts only
`TriageRecord{trainee_id, score, reason_codes, ts}` — the modules on that
boundary import no image library at all. This is enforced structurally and
checked by [`tests/test_privacy.py`](tests/test_privacy.py), not by a runtime
redaction filter the next contributor could forget. It is also what makes a
**private, proprietary model** practical to bring: it runs on the phone, and
the laptop only ever sees the numbers it emits.

The Android phone app lives in [`android/`](android/) in this same
repository — a dashboard front door opens the station for a floor's use case
(Fitness and Nursing are real screens; Lab and Welding are named
placeholders), which runs detection, pose, and — per use case — form
classification or CPR cadence on the phone's own NPU, and streams the
numeric result to this server. See [`android/README.md`](android/README.md)
for what it runs and [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the exact wire
contract it implements. [`demo/replay_client.py`](demo/replay_client.py) is a
reference client that stands in for a real phone when developing the laptop
side without one — an iOS app does not exist and is not started.

---

## Use cases

The engine is use-case-agnostic on purpose: one `[session] use_case` per floor
names what that floor is running, the beacon advertises it, every phone
declares it at `hello` and is refused at the handshake on mismatch (rather than
admitted and scored by nothing), and the scorer dispatches to that use case's
own definition of "needs a human". Adding a floor type is a scorer and a config
section, not a fork. What each one is today:

| Use case | Wire payload | Scored by | State |
|---|---|---|---|
| **`fitness`** | COCO-17 pose + `exercise`, `rep_count`, `form_ok`, closed-vocabulary `form_reason_codes` | Five weighted features on the laptop, with per-exercise weight profiles; form judged **on the phone** | **The worked example.** Three on-device form classifiers (plank, bicep, lunge), two geometric fault checks, rep counting. Weights unfitted to real incidents ([VALIDATION.md](docs/VALIDATION.md)) |
| **`nursing`** | COCO-17 pose + `procedure` | The laptop, per procedure. `cpr` measures compression rate, cadence regularity, and elbow lock from the pose itself | **Real, one procedure.** No phone-side classifier at all — the fault is derived from the movement. Thresholds beyond the published 100–120/min band are unfitted priors, and depth is deliberately **not** measured: a monocular uncalibrated camera cannot resolve it, and the scorer refuses rates the frame cadence cannot support |
| **`welding`** | Opaque `payload` object, carried through uninterpreted | Nothing — the scorer returns `0.0`, always | **Registered placeholder.** Proves the non-scoring lifecycle (connect, stream, appear, evict) without asserting anything unvalidated |

Fitness and nursing are deliberately *not* variations on one message. Fitness
trusts the phone's classifier for form and takes a closed-vocabulary verdict;
nursing sends no verdict at all and has the laptop measure a rhythm. That
difference is the reason the dispatch exists rather than a growing set of
optional fields bolted onto one shape.

The parts that carry over unchanged between floors are the point: the privacy
boundary, discovery and admission, the rolling session score, and staleness on
the console all work the same whether the person at the station is squatting
or compressing a manikin. What does *not* carry over is each use case's own
judgement — the fitness-side "cannot hold the movement" escalation, for
instance, is deliberately part of the fitness scorer, not the shared engine.

---

## Bringing your own use case

The engine is a phone-per-subject ingest path, a deterministic ranker, and an
honest console. None of that is fitness. Adding a use case means adding your
own parser, your own scorer, and (if you have something to draw) your own
renderer — five registries, no branching in anyone else's code:

| Extension point | Where | What it decides |
|---|---|---|
| `_OBSERVATION_PARSERS` | [`src/argus/ingest/protocol.py`](src/argus/ingest/protocol.py) | The fields your `observation` carries and how they are validated |
| `_SCORERS` | [`src/argus/triage.py`](src/argus/triage.py) | How evidence becomes one score plus reason codes |
| `history_len_for` | [`src/argus/triage.py`](src/argus/triage.py) | How much history a track keeps — a rhythm needs far more than a posture |
| `RENDERERS` | [`src/argus/console.py`](src/argus/console.py) | What a station card draws, if anything |
| `known_use_cases()` | derived from `_SCORERS` | Validates `[session] use_case` at load, fills the console's dropdown, and is what `POST /session/use_case` will accept |

Two properties are worth stating because they are what make "bring your own
private models" a real claim and not a slogan:

* **Your model stays on your phone.** The laptop loads no model, has no NPU
  workload and no inference dependency — it is pure Python. A proprietary
  classifier can ship inside a phone build with nothing about it disclosed
  here; what crosses the wire is the numeric observation your parser defines.
* **A use case cannot silently borrow another's assumptions.** A phone whose
  `hello` names a use case the laptop is not running is rejected, an
  `observation` that switches use case mid-stream is rejected, and a use case
  with no registered parser is rejected before any field is read. Nursing's
  512-frame history and fitness's 30 frames are separate answers to separate
  questions, not one tuned compromise.

The full runbook, including what welding deliberately does *not* do, is
[`docs/ADDING_A_USE_CASE.md`](docs/ADDING_A_USE_CASE.md). Adding an *exercise*
inside the existing fitness use case is a much smaller job with its own
runbook: [`docs/ADDING_AN_EXERCISE.md`](docs/ADDING_AN_EXERCISE.md).

---

## Quick start

Full instructions — prerequisites, both halves, every dependency, and the
troubleshooting table — are in [SETUP.md](SETUP.md) and [USAGE.md](USAGE.md).
This is the short version.

### Install the laptop side

Requires Python 3.11+ and git. Nothing else: no GPU, no model runtime, no
OpenCV, no camera library.

```powershell
.\run.ps1                     # Windows: .venv + deps + editable install
```

```bash
python -m venv .venv && source .venv/bin/activate    # any OS, stdlib only
pip install -r requirements.txt && pip install -e .
```

The `argus` command below assumes that venv is **activated**
(`.venv\Scripts\Activate.ps1` on Windows); without activating it, every
`argus …` is `.venv\Scripts\python.exe -m argus.cli …` instead.

### Run it with no phone at all

Generate a canned multi-station fixture and replay it over a real WebSocket
connection against a running server:

```bash
argus demo --ticks 900
argus run --http-port 8080     # in one terminal
argus replay --speed 1.0       # in another

# Watch the ranked, redacted output live:
#   http://127.0.0.1:8080/         (console)
#   http://127.0.0.1:8080/triage   (raw JSON)
```

`--speed 1.0` streams at the fixture's own pace, which is what makes the
console behave like a real floor rather than filling instantly. The synthetic
fixture is a **fitness** scene (`argus.synthetic`) — it exercises the ingest ->
triage -> alert path and the fitness scorer; a nursing or welding station is
driven by a phone or a hand-written client today.

```bash
# Diagnose the config, the ingest port, and whether phones can discover this
# laptop; also prints the LAN address a phone would connect to:
argus doctor

# What a phone does at setup: listen for laptops advertising themselves.
argus discover
```

### As a binary

The laptop side builds to a single executable with no Python install
required. Launched with no arguments it starts the ingest server and opens
the console, so it can be handed to someone who runs a floor rather than a
terminal.

```bash
pip install -e ".[build]"
python scripts/build_binary.py     # writes dist/argus (dist\argus.exe on Windows)

./dist/argus                       # server + console, no arguments needed
```

The build self-checks that the binary runs from a directory that is not the
repo, because a binary that only works beside its own source tree is exactly
the failure worth catching before shipping one.

### Installing the phone app

An Android phone with a Qualcomm Hexagon NPU, JDK 17, and the Android SDK;
then `cd android && ./gradlew installDebug`, and stage the perception models,
which are exported locally rather than committed because one of them is
AGPL-3.0. Step by step, including the toolchain and the phone itself:
[SETUP.md, Part 2](SETUP.md#part-2--the-phone).

### Connecting a phone

Name the session and say what this floor is running, so phones can find it and
check they agree:

```toml
[session]
name = "Coach Riley — 6pm class"
approval = "auto"      # or "manual" to approve each phone yourself
use_case = "fitness"   # "nursing", "welding" — see Use cases above
```

The laptop broadcasts where it is, so nobody types an IP: press **Find server
on this network** in the app's connect dialog, pick the session by name, and
the address fills itself in. The beacon carries `use_case` too, so a phone can
say "that laptop is running nursing, not fitness" during setup rather than
being refused at handshake after it has been placed on a rack. A human still
presses Connect — a beacon is an unauthenticated datagram, so it gets to make a
suggestion, not a decision. Typing the address by hand still works, which is
what a broadcast-filtered guest network falls back to. See
[Discovery](docs/PROTOCOL.md#discovery).

With `approval = "manual"`, a phone that asks to join waits while the request
sits at the top of the console with **Approve** and **Decline**. Auto is the
default because the failure modes are not symmetric: an unwanted phone on the
console is a nuisance you can see and disconnect, whereas a person standing at
a station unmonitored because nobody noticed a prompt is the thing Argus exists
to prevent. See [Admission](docs/PROTOCOL.md#admission).

**Over USB, for development** — no Wi-Fi needed, and no ambiguity about which
network the phone is on:

```powershell
adb reverse tcp:8765 tcp:8765
```

then type `ws://localhost:8765` as the server address in the app instead of
a LAN IP. See [`android/README.md`](android/README.md#windows--powershell-without-stagesh)
for the full device build/stage/connect workflow.

---

## Commands

| Command | Purpose |
|---|---|
| `argus run` | Start the WebSocket ingest server + triage ranking |
| `argus replay` | Replay a fixture into a running server, standing in for phones |
| `argus discover` | Listen for Argus beacons on the LAN — what a phone does at setup |
| `argus doctor` | Check config validity, port availability, and LAN reachability |
| `argus config` | Print the effective configuration after flag overrides |
| `argus demo` | Generate a canned multi-station observation fixture |

Useful `run` flags: `--ws-host`, `--ws-port`, `--json-log`, `--http-port`,
`--max-ticks` (bound the run — mainly for tests/CI), `--quiet`, `--use-case`
(what this floor is running; a phone whose app disagrees is rejected at
handshake — see [`docs/PROTOCOL.md`](docs/PROTOCOL.md)). The use case can also
be changed at runtime from the console's header dropdown, which affects every
*future* handshake and leaves connected phones on the use case they agreed to.

Flags override config; config never overrides flags. `argus config` shows
what a run is actually tuned with.

---

## How the score works

Every use case answers the same question — *one score in [0, 1] plus the reason
codes that explain it* — and answers it its own way. `compute_triage` is a
dispatch on `track.use_case`; nothing below is shared arithmetic.

### Fitness

Five weighted features, computed from numeric keypoints and boxes only —
nothing here has ever seen a pixel:

| Feature | Weight | Signal |
|---|---:|---|
| `fall` | 0.40 | Sudden torso-centroid drop + bbox aspect flip (wider than tall) |
| `stillness` | 0.20 | Fraction of the ~2 s window with near-zero centroid motion |
| `occlusion` | 0.15 | Both hands *and* face below the keypoint-confidence threshold |
| `form_error` | 0.15 | The phone's own on-device form/exercise classifier reason code, looked up against a closed vocabulary |
| `off_task` | 0.10 | Shoulder-line deviation from the station-facing angle |

Anything at or above `alert_threshold` (0.5) is surfaced with `reason_codes`
explaining why. Ties break on `trainee_id`, so the rank is stable across runs.

#### Some features are wrong for some exercises

Those five weights are the *default* profile. An exercise can name its own
vector in `[scoring.exercise_weights]`, because a movement can make a feature
meaningless or actively misleading. A correct plank is horizontal and
motionless — which is precisely what `fall` (bbox wider than tall) and
`stillness` (centroid not moving) are built to treat as an emergency.

Measured, on the defaults: a textbook plank scores **0.42** of the 0.5
threshold and displays `prolonged_stillness, off_task_orientation`, leaving
the actual form signal worth 0.12 against 0.42 of noise. On
`[scoring.exercise_weights.plank]` the same observation scores **0.0** and
displays nothing, while a sagging plank scores **0.68**.

A feature weighted 0 contributes nothing **and** emits no reason code — one
number does both, because a reason that explains no part of the score is worse
than no reason at all. An exercise with no profile scores on the defaults,
unchanged.

**These weights are unvalidated.** They are the prototype author's priors; no
incident has ever been scored with them. They now live in
[`configs/argus.default.toml`](configs/argus.default.toml), so retuning is a
config edit plus a `config_version` bump rather than a code change — see
[VALIDATION.md §1](docs/VALIDATION.md).

### Nursing

A nursing station names a `procedure`, and the laptop dispatches again on it:
a procedure this build has no scorer for scores a flat `0.0` rather than being
refused, so a ward running something unimplemented still sees its station.
`cpr` is the one implemented today, and it takes the **worst single fault**
rather than a weighted sum — each of these is independently a reason to send
someone, and averaging would let a dangerously slow rate be diluted by
well-locked elbows:

| Reason code | Fires when |
|---|---|
| `cpr_rate_slow` / `cpr_rate_fast` | Compression rate outside `cpr_rate_min_bpm`–`cpr_rate_max_bpm` (100–120, the AHA's published adult band) |
| `cpr_cadence_erratic` | Cycle-to-cycle variation past `cpr_cadence_cv_threshold` — alternating 80s and 140s average to a perfect 110 |
| `cpr_arms_bent` | Elbow angle under `cpr_min_elbow_angle_deg` |

Two things it deliberately does not do. It **does not report compression depth
or hand placement** — depth from an uncalibrated monocular camera needs a scale
reference the frame does not contain, and a published feasibility study against
an instrumented manikin found frequency agreed closely while depth was "overall
not accurate". And it **refuses a rate it cannot resolve**: below ~6 Hz of
observations the true period and twice the true period become
indistinguishable, so a correct 120/min reads back as 60/min — and an
instructor told "too slow" would coach someone into making it worse. A nursing
station must stream at **≥ 10 Hz**; below the resolvable limit the scorer
reports no rate instead of a plausible wrong one. See
[VALIDATION.md §2b](docs/VALIDATION.md).

### Welding

`compute_triage_welding` returns `0.0` with no reason codes and does not read
`payload` at all. That is the point: a welding station can connect, stream,
appear on the console and be evicted like any other, without anything asserting
a claim about technique that nothing has validated.

---

## Configuration

One versioned TOML holds every tunable: the fitness weights and thresholds, the
form-error vocabulary, the CPR band, the session's use case, and the WebSocket
ingest settings. Nothing in `src/argus/` hardcodes a tuning constant.

`config_version` is validated on load, unknown keys are rejected rather than
ignored, and the fitness weights must sum to 1.0. A typo'd `alert_threshhold`
fails loudly instead of silently leaving the default in place — an operator who
thinks they retuned the system and did not is the failure mode that matters.
`[session] use_case` is validated the same way, against `known_use_cases()`, so
a floor configured for a use case this build cannot score fails at load rather
than after a session of accepting phones.

`[scoring.form_error_vocab]` is not just server-side tuning — it is the
closed vocabulary a fitness phone's `form_reason_codes` must be drawn from.
Phone and laptop are expected to deploy from copies of the same config file;
see [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

---

## Layout

| Path | What it is |
|---|---|
| [`src/argus/triage.py`](src/argus/triage.py) | The deterministic scorers and the `use_case` -> scorer registry. Pure functions, stdlib only. |
| [`src/argus/config.py`](src/argus/config.py) | Versioned config loading and validation |
| [`src/argus/ingest/`](src/argus/ingest/) | The WebSocket boundary: per-use-case wire parsers, per-station sessions, the server |
| [`src/argus/synthetic.py`](src/argus/synthetic.py) | The synthetic fitness scene `argus demo` and the tests replay |
| [`src/argus/outputs.py`](src/argus/outputs.py), [`alerts.py`](src/argus/alerts.py) | The alert boundary: stderr alerts, JSON lines, HTTP, and the console's snapshot. Import no image library. |
| [`src/argus/console.py`](src/argus/console.py) | The console page served at `GET /`, and the per-use-case renderers |
| [`src/argus/discovery.py`](src/argus/discovery.py) | The LAN beacon that lets a phone find this laptop |
| [`scripts/build_binary.py`](scripts/build_binary.py) | Freezes the laptop side into a standalone executable |
| [`android/`](android/) | The phone app: detection + pose on the NPU, the fitness and nursing stations, the dashboard shell — see `android/README.md` |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | The wire contract the phone app implements, per use case |
| [`docs/CONSOLE.md`](docs/CONSOLE.md) | What the console shows, and what it is allowed to see |
| [`docs/ADDING_A_USE_CASE.md`](docs/ADDING_A_USE_CASE.md) | Runbook for a whole new domain: parser, scorer, renderer, phone station |
| [`docs/ADDING_AN_EXERCISE.md`](docs/ADDING_AN_EXERCISE.md) | Runbook for the next fitness form classifier, and the traps plank, bicep, and lunge each hit |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | What Argus has *not* been shown to do |
| [`SETUP.md`](SETUP.md) | Installing everything from scratch — laptop, phone, models, every dependency |
| [`USAGE.md`](USAGE.md) | Running it: with no phone, with fitness, with CPR; the console, the CLI, troubleshooting |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design and the reasoning behind it |

---

## Tests

```bash
pytest tests/ -q                                    # 433 passed
cd android && ./gradlew testDebugUnitTest           # the Kotlin host tests
```

Covers the scorers, config validation, the wire protocol for every registered
use case, session reconnect and TTL eviction, the ingest server over a real
WebSocket, both output sinks, and the CLI end to end. Four are the gates that
matter:

* **Determinism** — the same observation history replayed twice through the
  scoring path produces byte-identical output, with a companion test
  asserting the run was not simply empty.
* **Privacy** — no boundary module may import an image library, and no
  public callable on the boundary may accept an image-capable type. Checked
  by inspecting imports and type annotations.
* **Protocol** — an unrecognised protocol version, an unregistered `use_case`,
  a use case switched mid-connection, a malformed message, or a
  `form_reason_codes` entry outside the configured vocabulary is rejected,
  not silently ignored.
* **Reconnect** — a station reconnecting within `ingest.track_ttl_s` resumes
  its rolling history rather than starting over; a second live connection
  claiming the same `trainee_id` is rejected as a collision.

---

## Status

Runnable and tested today: the full ingest -> triage -> alert path for all
three registered use cases, replayed over a real WebSocket against a canned
fixture with no phone involved, and driven end to end by a real Android phone
(detection and pose on its own NPU) over both USB and LAN Wi-Fi; the console
and JSON/HTTP sinks; the CLI.

**The phone app is built and runs — see [`android/`](android/).** Fitness has
working on-device form classifiers for plank, bicep, and lunge, plus geometric
fault checks and rep counting for bicep/lunge. Nursing has its own station
screen streaming CPR pose at the rate the measurement requires. Lab and welding
are named placeholders on the dashboard that explain what shipping them would
take rather than pretending they exist.

What is still missing is validation against real subjects: every accuracy
figure any fitness classifier reports is a held-out-frame number from its
training dataset's own recordings, and no real CPR has been measured — the
rate estimator is tested against synthetic waveforms at known rates, which is a
test of the arithmetic, not of a camera. [docs/VALIDATION.md](docs/VALIDATION.md)
is the honest list, with what each gap would take to close.

## Team
1. Kavinder Roghit Kanthen - kkanthen@qti.qualcomm.com
2. Penghai Wei - pengwei@qti.qualcomm.com
3. Viraj Shah - virashah@qti.qualcomm.com
4. Aakarsh Gupta - aakagupt@qti.qualcomm.com

---

## Licence

MIT — see [LICENSE](LICENSE).
