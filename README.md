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
> is supported, because no real trainee footage exists (§1). And the phone's
> pose model is AGPL-3.0, which is fine to demo and develop against but is a
> licensing decision before distributing an application built on it — see
> [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), where the permissively
> licensed alternatives are already scouted.

**Privacy is a property of the wiring.** No frame ever exists past a phone's
own camera pipeline, and every sink's signature accepts only
`TriageRecord{trainee_id, score, reason_codes, ts}` — the modules on that
boundary import no image library at all. This is enforced structurally and
checked by [`tests/test_privacy.py`](tests/test_privacy.py), not by a runtime
redaction filter the next contributor could forget.

The Android/iOS phone app is **not part of this repository** — see
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the exact wire contract it needs to
implement, and [`demo/replay_client.py`](demo/replay_client.py) for a
reference client that stands in for a real phone during development.

---

## Quick start

```powershell
.\run.ps1                     # .venv + deps + editable install

# No phone needed: generate a canned multi-trainee observation fixture and
# replay it over a real WebSocket connection against a running server.
.venv\Scripts\python.exe -m argus.cli demo --ticks 60
.venv\Scripts\python.exe -m argus.cli run --http-port 8080 &
.venv\Scripts\python.exe demo\replay_client.py

# Watch the ranked, redacted output live:
#   http://127.0.0.1:8080/         (trainer dashboard)
#   http://127.0.0.1:8080/triage   (raw JSON)
```

```powershell
# Diagnose the config and the ingest port; also prints the LAN address a
# phone should connect to:
.venv\Scripts\python.exe -m argus.cli doctor
```

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
| [`src/argus/outputs.py`](src/argus/outputs.py), [`alerts.py`](src/argus/alerts.py) | The alert boundary: console, JSON lines, HTTP, and the trainer dashboard. Import no image library. |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | The wire contract a phone app must implement |
| [`docs/ADDING_AN_EXERCISE.md`](docs/ADDING_AN_EXERCISE.md) | Runbook for the next form classifier, and the four traps the plank hit |
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

Runnable and tested today: the full ingest -> triage -> alert path, replayed
over a real WebSocket against a canned multi-trainee fixture with no phone
involved; the trainer dashboard and JSON/HTTP sinks; the CLI.

**Not yet built: the phone app.** Pose estimation, form/exercise
classification, and the client half of `docs/PROTOCOL.md` are a separate,
future project. No accuracy figure exists for any on-device model, because
none has been built yet. [docs/VALIDATION.md](docs/VALIDATION.md) is the
honest list, with what each gap would take to close.

---

## Licence

MIT — see [LICENSE](LICENSE).
