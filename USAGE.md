# Usage — running Argus

How to actually run the system, in four paths: with no phone at all, with a
phone doing fitness, with a phone doing CPR, and with a use case that has no
classifier yet. Then the console, the outputs, the full command reference, and
what to do when something does not work.

Install first: [SETUP.md](SETUP.md).

Every command below assumes the virtual environment is **activated**. Without
activating it, replace `argus` with `.venv\Scripts\python.exe -m argus.cli`
(Windows) or `.venv/bin/python -m argus.cli` (macOS/Linux) — identical in
effect.

---

## The 60-second version

```bash
argus demo --ticks 300                       # canned 5-station fixture
argus run --http-port 8080                   # terminal 1: server + console
argus replay --speed 1.0                     # terminal 2: pretend to be phones
```

Open **http://127.0.0.1:8080/**. Five stations appear and are ranked live; one
falls, one goes still, one holds a good plank and one a sagging one. `Ctrl-C`
in terminal 1 stops the server.

---

## Path A — no phone, fixture replay

The whole ingest → triage → alert path over a **real** WebSocket connection,
with no phone, no model, and no camera anywhere in the loop. This is what CI
runs on every commit.

### 1. Generate a fixture

```bash
argus demo --ticks 300 --out demo/synthetic_stream.json
```

Writes a synthetic **fitness** scene: five stations with deliberately different
behaviours. `--ticks 300` is about 20 seconds of floor time at the fixture's
own pace.

### 2. Start the server

```bash
argus run --http-port 8080
```

```
argus 0.1.0 | ws://0.0.0.0:8765 | protocol_version=1
triage endpoint:  http://127.0.0.1:8080/triage
trainer console:  http://127.0.0.1:8080/
```

`--http-port` matters: the console and `/triage` are **off by default**
(`outputs.http_port = 0`). `argus run --open` turns them on at port 8080 and
opens a browser, which is also what launching the packaged binary with no
arguments does.

### 3. Replay the fixture into it

In a second terminal:

```bash
argus replay --speed 1.0
```

```
[walker] sent 300 observations
[still] sent 300 observations
[faller] sent 300 observations
[good_plank] sent 300 observations
[sagging_plank] sent 300 observations
```

`--speed 1.0` honours the fixture's own timestamps, so the console behaves like
a real floor. The default, `--speed 0`, sends as fast as possible — right for a
smoke test, wrong for watching.

### 4. Watch

| Where | What |
|---|---|
| http://127.0.0.1:8080/ | the console — ranked station cards, skeletons, reason codes |
| http://127.0.0.1:8080/triage | the ranked records as raw JSON |
| the server's terminal | stderr alerts as stations cross the threshold |

`faller` should rise to the top with `fall_detected`; `sagging_plank` should
show a form code while `good_plank` stays silent — that contrast is the point
of the per-exercise weight profiles.

### Bounding a run

`--max-ticks N` stops after N rank ticks (0.5 s each by default), which is how
CI and the tests run it without a background process to kill.

---

## Path B — a phone, fitness

### On the laptop

```toml
# configs/argus.default.toml
[session]
name = "Coach Riley — 6pm class"   # how the phone shows this laptop in the picker
approval = "auto"                  # or "manual" to approve each phone yourself
use_case = "fitness"
```

```bash
argus doctor          # confirms the port, and prints the address a phone should use
argus run --http-port 8080
```

### On the phone

1. Open **Argus Edge** → tap the **Fitness** tile.
2. Tap **Server**. Either press **Find server on this network** and pick the
   session by name, or type the address:
   * over USB, after `adb reverse tcp:8765 tcp:8765` on the laptop:
     `ws://localhost:8765`
   * over Wi-Fi: `ws://<laptop-ip>:8765`
3. Fill in **Who is at this station?** and pick the **Exercise** — that choice
   is what selects the laptop's scoring profile, so a plank is not scored as if
   it were a fall.
4. Press **Connect**, then **Start**.

The status chip names the state and the active backend. Boxes appear around
everybody in frame; only the tracked subject's pose crosses the wire.

### What you should see

The station appears on the console within a rank tick or two, with a live
skeleton, its rep count, and any form reason codes the phone's on-device
classifier emitted. Nothing else ever leaves the phone: no frame exists past
its own camera pipeline, and the only fields on the wire are the numeric ones
[docs/PROTOCOL.md](docs/PROTOCOL.md) defines.

Three exercises have on-device form classifiers: **plank**, **bicep**,
**lunge**. Bicep and lunge also count reps and run a geometric fault check. Any
other exercise streams pose and reps with no form codes at all rather than
guessing.

### With `approval = "manual"`

A phone that asks to join waits in a `join_pending` state and *says so on its
own screen*, while the request sits at the top of the console with **Approve**
and **Decline**. Auto is the default because the failure modes are not
symmetric: an unwanted phone is a nuisance you can see and disconnect, whereas
a person standing unmonitored because nobody noticed a prompt is the thing
Argus exists to prevent.

Approve/Decline is accepted **only from the machine serving the console**
unless `outputs.allow_remote_join_control = true`, because approving a phone
decides who monitors a trainee.

---

## Path C — a phone, nursing / CPR

Same app, different station screen, and a genuinely different scoring path: the
phone runs **no classifier at all** and streams pose; the laptop derives the
fault from the movement itself.

### On the laptop

```bash
argus run --use-case nursing --http-port 8080
```

or set `[session] use_case = "nursing"` in the config. The console's header
dropdown also changes it at runtime — that affects every *future* handshake and
leaves connected phones on the use case they already agreed to.

### On the phone

**Argus Edge** → **Nursing** tile → **Connect** (same dialog) → **Start**, with
the phone placed so the whole torso and both arms of whoever is compressing are
in frame.

### What the laptop reports

| Reason code | Fires when |
|---|---|
| `cpr_rate_slow` / `cpr_rate_fast` | rate outside 100–120/min (the AHA adult band) |
| `cpr_cadence_erratic` | cycle-to-cycle variation past the configured threshold |
| `cpr_arms_bent` | elbow angle under 150° |

It takes the **worst single fault**, not a weighted sum — each of these is
independently a reason to send someone, and averaging would let a dangerously
slow rate be diluted by well-locked elbows.

Two things it deliberately will not do:

* **No depth or hand placement.** Depth from an uncalibrated monocular camera
  needs a scale reference the frame does not contain.
* **It refuses a rate it cannot resolve.** Below ~6 Hz of observations the true
  period and twice the true period are indistinguishable, so a correct 120/min
  would read back as 60/min — and an instructor told "too slow" would coach
  someone into making it worse. **A nursing station must stream at ≥ 10 Hz**;
  below the resolvable limit the scorer reports no rate rather than a plausible
  wrong one. The station screen carries an on-screen rate echo that is advisory
  only — where the two disagree, the laptop is the authority.

See [docs/VALIDATION.md](docs/VALIDATION.md) §2b.

---

## Path D — welding, the registered placeholder

`welding` is a use case the laptop accepts, carries, ranks and evicts like any
other — and scores a flat `0.0` with no reason codes, without reading its
payload at all. That is the point: it proves the non-scoring lifecycle without
asserting anything about technique that nothing has validated.

There is **no welding station in the phone app** (the dashboard tile says what
shipping one would take). To drive it, run the laptop with
`argus run --use-case welding` and connect a hand-written client that sends a
`welding` `hello` and observations carrying an opaque `payload` object — the
wire shape is in [docs/PROTOCOL.md](docs/PROTOCOL.md), and
[`demo/replay_client.py`](demo/replay_client.py) is the reference client to
start from.

Adding a real one is five registries and no branching in anyone else's code:
[docs/ADDING_A_USE_CASE.md](docs/ADDING_A_USE_CASE.md).

---

## The console

Served at `GET /` whenever `outputs.http_port` is set. Full description:
[docs/CONSOLE.md](docs/CONSOLE.md).

| Element | What it does |
|---|---|
| Station cards | one per connected phone, ordered by score, highest first |
| Skeleton | drawn from the live numeric keypoints — *not* from any image |
| Reason codes | why this station scores what it does; a feature weighted 0 emits none |
| Stale marking | a station silent past `console_stale_after_s` (2 s) is drawn stale, so a quiet station and a calm one can be told apart |
| Join requests | Approve / Decline, when `approval = "manual"` |
| Use-case dropdown | changes what future handshakes must declare |

Ties break on `trainee_id`, so the rank is stable across runs rather than
shuffling between equal scores.

---

## Outputs

| Sink | Enable with | Carries |
|---|---|---|
| stderr alerts | `outputs.console = true` (default) | `TriageRecord` — id, score, reason codes, ts |
| JSON lines log | `--json-log out/run.jsonl` | one JSON object per rank tick, same four fields per record |
| `GET /triage` | `--http-port 8080` | the current ranked records |
| `GET /console` | same | the above **plus** live numeric observations, so the page can draw a skeleton |
| `GET /healthz` | same | liveness |

Every sink except `/console` is typed to accept only the four-field
`TriageRecord`, and the modules on that boundary import no image library at
all. That is checked structurally by
[`tests/test_privacy.py`](tests/test_privacy.py), not by a redaction filter the
next contributor could forget.

---

## Command reference

| Command | Purpose |
|---|---|
| `argus run` | Start the WebSocket ingest server + triage ranking |
| `argus replay` | Replay a fixture into a running server, standing in for phones |
| `argus demo` | Generate a canned multi-station observation fixture |
| `argus discover` | Listen for Argus beacons on the LAN — what a phone does at setup |
| `argus doctor` | Check config validity, port availability, LAN reachability |
| `argus config` | Print the effective configuration after flag overrides |

Common flags, on `run`, `doctor`, `config` and `discover`:

| Flag | Effect |
|---|---|
| `--config PATH` | use a different TOML (default `configs/argus.default.toml`) |
| `--ws-host`, `--ws-port` | where the ingest server listens (default `0.0.0.0:8765`) |
| `--http-port N` | serve the console and `/triage` (0 = off, the default) |
| `--json-log PATH` | append one JSON object per rank tick |
| `--use-case NAME` | what this floor is running; a phone that disagrees is refused |
| `--quiet` | suppress the stderr alert sink |

`run` also takes `--max-ticks N` (stop after N ticks — mainly for tests and CI)
and `--open` (enable the console on 8080 and open a browser).

`replay` takes `--fixture`, `--ws-host`, `--ws-port`, and `--speed`.
`discover` takes `--timeout`.

**Flags override config; config never overrides flags.** `argus config` prints
what a run is actually tuned with — use it when a setting does not seem to have
taken.

---

## Configuration

One versioned TOML, [`configs/argus.default.toml`](configs/argus.default.toml),
holds every tunable: the fitness weights and thresholds, the form-error
vocabulary, the CPR band, the session's use case, discovery, and the ingest
settings. Nothing in `src/argus/` hardcodes a tuning constant.

The edits you are most likely to make:

```toml
[session]
name = "Coach Riley — 6pm class"   # what phones show in the server picker
approval = "auto"                  # "manual" to approve each phone
use_case = "fitness"               # "nursing", "welding"

[outputs]
http_port = 8080                   # turn the console on permanently
json_log = "out/session.jsonl"     # keep a record of the run

[ingest]
ws_port = 8765                     # move it if something else owns 8765
```

`config_version` is validated on load, unknown keys are **rejected rather than
ignored**, and the fitness weights must sum to 1.0. A typo'd
`alert_threshhold` fails loudly instead of silently leaving the default in
place — an operator who thinks they retuned the system and did not is the
failure mode that matters.

To keep local settings separate from the committed defaults, copy the file and
pass `--config my-floor.toml`.

**The fitness weights are unvalidated priors.** No incident has ever been
scored with them; see [docs/VALIDATION.md](docs/VALIDATION.md) before treating
any number here as tuned.

---

## Tests

```bash
pytest tests/ -q                  # 433 passed, ~35 s
cd android && ./gradlew testDebugUnitTest         # Kotlin host tests
cd android && ./gradlew connectedDebugAndroidTest # NPU + parity, needs a device
```

The four gates that matter: **determinism** (the same history replayed twice
produces byte-identical output), **privacy** (no boundary module may import an
image library), **protocol** (a bad version, an unregistered use case, a use
case switched mid-stream, or an out-of-vocabulary form code is rejected, not
ignored), and **reconnect** (a station returning within the TTL resumes its
history; a second connection claiming the same id is refused).

> `connectedDebugAndroidTest` uninstalls the app and wipes the staged models
> when it finishes. Re-run `android/stage.sh` afterwards.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Nothing at `http://127.0.0.1:8080/` | the console is off by default | `argus run --http-port 8080`, or `--open` |
| `could not bind …:8765` | something else owns the port | `argus run --ws-port 8766` and point the phone at 8766 |
| `argus replay` cannot connect | the server is not up yet, or is on another port | start `argus run` first; pass a matching `--ws-port` |
| `replay: fixture not found` | `argus demo` not run | `argus demo --ticks 300` first |
| Console is empty while the phone says Connected | the phone is streaming a different use case | the console header dropdown, or `--use-case`; a mismatch is refused at handshake and reported |
| Phone refused with a protocol-version error | app and laptop built from different commits | rebuild the app; `protocol_version` must match on both |
| Phone refused as a collision | another live connection claims that `trainee_id` | disconnect the other station, or use a different name |
| A station vanishes from the console | silent past `ingest.track_ttl_s` (10 s) | expected — it is presumed to have left; it resumes its history if it returns within the TTL |
| A station is drawn stale | silent past 2 s but not evicted | that distinction is deliberate: a quiet station and a calm one must not look the same |
| Nursing shows no rate | the stream is below the resolvable rate | stations must run at **≥ 10 Hz**; the scorer refuses a rate it cannot resolve rather than reporting a wrong one |
| A correct plank scores high | running on the default weights instead of the plank profile | pick **plank** as the exercise on the phone — that is what selects the profile |
| No form codes for an exercise | only plank, bicep and lunge ship a classifier | expected; see [docs/ADDING_AN_EXERCISE.md](docs/ADDING_AN_EXERCISE.md) |
| A phantom trainee appears | the detector fires on any depiction — a mirror, a poster, a screen | move the phone; a gym full of mirrors is a real placement problem, noted in [`android/README.md`](android/README.md) |
| **Find server** finds nothing | the network drops broadcast | type `ws://<laptop-ip>:8765`; `argus doctor` prints the address |
| Phone reports it cannot connect over USB | the `adb reverse` tunnel is gone | re-run `adb reverse tcp:8765 tcp:8765` — it does not survive a reconnect or reboot |

Install-time problems are in [SETUP.md § Appendix B](SETUP.md#appendix-b--install-troubleshooting).

---

## Where to read next

| Document | What it answers |
|---|---|
| [SETUP.md](SETUP.md) | installing everything from scratch, laptop and phone |
| [ARCHITECTURE.md](ARCHITECTURE.md) | why the system is shaped this way |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | the exact wire contract, per use case |
| [docs/CONSOLE.md](docs/CONSOLE.md) | what the console shows and what it is allowed to see |
| [docs/VALIDATION.md](docs/VALIDATION.md) | what Argus has **not** been shown to do |
| [docs/ADDING_A_USE_CASE.md](docs/ADDING_A_USE_CASE.md) | adding a whole domain |
| [docs/ADDING_AN_EXERCISE.md](docs/ADDING_AN_EXERCISE.md) | adding a fitness form classifier |
| [`android/README.md`](android/README.md) | how the phone half works internally |
