# Argus — many eyes, one mind

Argus watches a HIIT floor through one phone per trainee and produces a
**deterministic, explainable triage rank** of who needs a human instructor
right now.

Each phone runs its own on-device pose model and form/exercise classifier and
streams the numeric result to a laptop over WebSocket — no video ever leaves
the phone. The laptop ranks every connected trainee with a pure, auditable
scorer and gives the trainer a live view.

> **Scope: this is a hackathon build.** It is a working system, not a shipped
> product — the engineering is real and the measurements are real, but three
> things would need revisiting before it went anywhere near a gym floor. The
> scoring weights have never been fitted to an incident
> ([VALIDATION.md](docs/VALIDATION.md) §2). No pose or detection accuracy claim
> is supported, because the app has been run on real devices but has never
> watched a real trainee whose reps were labelled and checked against ([§1](docs/VALIDATION.md)).
> And the phone's pose model is AGPL-3.0, which is fine to demo and develop
> against but is a licensing decision before distributing an application
> built on it — see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), where
> the permissively licensed alternatives are already scouted.

**Privacy is a property of the wiring.** No frame ever exists past a phone's
own camera pipeline, and every sink's signature accepts only
`TriageRecord{trainee_id, score, reason_codes, ts}` — the modules on that
boundary import no image library at all. This is enforced structurally and
checked by [`tests/test_privacy.py`](tests/test_privacy.py), not by a runtime
redaction filter the next contributor could forget.

The Android phone app lives in [`android/`](android/) in this same
repository — a dashboard front door (Fitness is real; Nursing/Lab/Welding are
named placeholders for the same engine applied elsewhere) opens a screen that
runs detection, pose, and form classification on the phone's own NPU, and
streams the result to this server. See [`android/README.md`](android/README.md)
for what it runs and [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the exact wire
contract it implements. [`demo/replay_client.py`](demo/replay_client.py) is a
reference client that stands in for a real phone when developing the laptop
side without one — an iOS app does not exist and is not started.

---

## Quick start

### As a binary

The laptop side builds to a single executable with no Python install
required. Launched with no arguments it starts the ingest server and opens
the trainer console, so it can be handed to someone who runs a gym rather
than a terminal.

```bash
pip install -e ".[build]"
python scripts/build_binary.py     # writes dist/argus (dist\argus.exe on Windows)

./dist/argus                       # server + console, no arguments needed
```

The build self-checks that the binary runs from a directory that is not the
repo, because a binary that only works beside its own source tree is exactly
the failure worth catching before shipping one.

### From a checkout

```powershell
.\run.ps1                     # .venv + deps + editable install

# No phone needed: generate a canned multi-trainee observation fixture and
# replay it over a real WebSocket connection against a running server.
.venv\Scripts\python.exe -m argus.cli demo --ticks 900
.venv\Scripts\python.exe -m argus.cli run --http-port 8080 &
.venv\Scripts\python.exe -m argus.cli replay --speed 1.0

# Watch the ranked, redacted output live:
#   http://127.0.0.1:8080/         (trainer console)
#   http://127.0.0.1:8080/triage   (raw JSON)
```

`--speed 1.0` streams at the fixture's own pace, which is what makes the
console behave like a real floor rather than filling instantly.

```powershell
# Diagnose the config, the ingest port, and whether phones can discover this
# laptop; also prints the LAN address a phone would connect to:
.venv\Scripts\python.exe -m argus.cli doctor

# What a phone does at setup: listen for laptops advertising themselves.
.venv\Scripts\python.exe -m argus.cli discover
```

### Connecting a phone

Name the session so phones can find it:

```toml
[session]
name = "Coach Riley — 6pm HIIT"
approval = "auto"      # or "manual" to approve each phone yourself
```

The laptop broadcasts where it is, so nobody types an IP: press **Find server
on this network** in the app's connect dialog, pick the session by name, and
the address fills itself in. A human still presses Connect — a beacon is an
unauthenticated datagram, so it gets to make a suggestion, not a decision.
Typing the address by hand still works, which is what a broadcast-filtered
guest network falls back to. See [Discovery](docs/PROTOCOL.md#discovery).

With `approval = "manual"`, a phone that asks to join waits while the request
sits at the top of the trainer console with **Approve** and **Decline**. Auto
is the default because the failure modes are not symmetric: an unwanted phone
on the console is a nuisance you can see and disconnect, whereas a trainee
standing at a rack unmonitored because nobody noticed a prompt is the thing
Argus exists to prevent. See [Admission](docs/PROTOCOL.md#admission).

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
| `argus doctor` | Check config validity, port availability, and LAN reachability |
| `argus config` | Print the effective configuration after flag overrides |
| `argus demo` | Generate a canned multi-station observation fixture |

Useful `run` flags: `--ws-host`, `--ws-port`, `--json-log`, `--http-port`,
`--max-ticks` (bound the run — mainly for tests/CI), `--quiet`.

Flags override config; config never overrides flags. `argus config` shows
what a run is actually tuned with.

---

## How the score works

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

### Some features are wrong for some exercises

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

---

## Configuration

One versioned TOML holds every tunable: scoring weights, thresholds, the
form-error vocabulary, and the WebSocket ingest settings. Nothing in
`src/argus/` hardcodes a tuning constant.

`config_version` is validated on load, unknown keys are rejected rather than
ignored, and the weights must sum to 1.0. A typo'd `alert_threshhold` fails
loudly instead of silently leaving the default in place — an operator who
thinks they retuned the system and did not is the failure mode that matters.

`[scoring.form_error_vocab]` is not just server-side tuning — it is the
closed vocabulary a phone's `form_reason_codes` must be drawn from. Phone and
laptop are expected to deploy from copies of the same config file; see
[`docs/PROTOCOL.md`](docs/PROTOCOL.md).

---

## Layout

| Path | What it is |
|---|---|
| [`src/argus/triage.py`](src/argus/triage.py) | The deterministic scorer. Pure functions, stdlib only. |
| [`src/argus/config.py`](src/argus/config.py) | Versioned config loading and validation |
| [`src/argus/ingest/`](src/argus/ingest/) | The WebSocket boundary: wire protocol, per-trainee sessions, the server |
| [`src/argus/synthetic.py`](src/argus/synthetic.py) | The synthetic trainee scene `argus demo` and the tests replay |
| [`src/argus/outputs.py`](src/argus/outputs.py), [`alerts.py`](src/argus/alerts.py) | The alert boundary: stderr alerts, JSON lines, HTTP, and the console's snapshot. Import no image library. |
| [`src/argus/console.py`](src/argus/console.py) | The trainer console page served at `GET /` |
| [`src/argus/discovery.py`](src/argus/discovery.py) | The LAN beacon that lets a phone find this laptop |
| [`scripts/build_binary.py`](scripts/build_binary.py) | Freezes the laptop side into a standalone executable |
| [`android/`](android/) | The phone app: detection + pose on the NPU, form classifiers, the dashboard shell — see `android/README.md` |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | The wire contract the phone app implements |
| [`docs/CONSOLE.md`](docs/CONSOLE.md) | What the trainer console shows, and what it is allowed to see |
| [`docs/ADDING_AN_EXERCISE.md`](docs/ADDING_AN_EXERCISE.md) | Runbook for the next form classifier, and the traps plank, bicep, and lunge each hit |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | What Argus has *not* been shown to do |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design and the reasoning behind it |

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

Covers the scorer, config validation, the wire protocol, session reconnect
and TTL eviction, the ingest server over a real WebSocket, both output sinks,
and the CLI end to end. Four are the gates that matter:

* **Determinism** — the same observation history replayed twice through the
  scoring path produces byte-identical output, with a companion test
  asserting the run was not simply empty.
* **Privacy** — no boundary module may import an image library, and no
  public callable on the boundary may accept an image-capable type. Checked
  by inspecting imports and type annotations.
* **Protocol** — an unrecognised protocol version, a malformed message, or a
  `form_reason_codes` entry outside the configured vocabulary is rejected,
  not silently ignored.
* **Reconnect** — a trainee reconnecting within `ingest.track_ttl_s` resumes
  their rolling history rather than starting over; a second live connection
  claiming the same `trainee_id` is rejected as a collision.

---

## Status

Runnable and tested today: the full ingest -> triage -> alert path, both
replayed over a real WebSocket against a canned multi-trainee fixture with no
phone involved, and driven end to end by a real Android phone (detection,
pose, and form classification on its own NPU) over both USB and LAN Wi-Fi;
the trainer dashboard and JSON/HTTP sinks; the CLI.

**The phone app is built and runs — see [`android/`](android/).** Plank,
bicep, and lunge all have working on-device form classifiers, plus geometric
fault checks and rep counting for bicep/lunge, and a dashboard front door
naming where this is headed next (Nursing, Lab, Welding). What is still
missing is validation against a real trainee: every accuracy figure any
classifier reports is a held-out-frame number from its training dataset's own
recordings, not a measurement against anyone this app has watched.
[docs/VALIDATION.md](docs/VALIDATION.md) is the honest list, with what each
gap would take to close.

---

## Licence

MIT — see [LICENSE](LICENSE).
