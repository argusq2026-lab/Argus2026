# Argus — many eyes, one mind

Argus watches many trainees at once on a Snapdragon X Elite AI PC and produces
a **deterministic, explainable triage rank** of who needs a human instructor
right now.

Detection, pose, and super-resolution run on the Hexagon NPU. A small VLM
samples only trainees the numeric scorer has already flagged, at ~1 Hz. The
ranking itself is a pure function of numeric history — the same input always
produces the same rank, even though a VLM's decoding is not deterministic.

**Privacy is a property of the wiring.** Raw frames and raw VLM captions have
no path to a sink: every sink's signature accepts only
`TriageRecord{trainee_id, score, reason_codes, ts}`, and the modules on that
boundary import no image library at all. This is enforced structurally and
checked by [`tests/test_privacy.py`](tests/test_privacy.py), not by a runtime
redaction filter that the next contributor could forget.

---

## Quick start

```powershell
.\run.ps1                     # x86-64 .venv + deps + editable install
.venv\Scripts\python.exe -m argus.cli run --engine mock --max-ticks 60
```

That runs the whole pipeline — capture, detection decode, two-stage pose, the
25→COCO landmark remap, tracking, triage, alerts — against synthetic tensors
emitted at the real artifact contracts. No models, no camera, no NPU needed.

```powershell
# Provision the real artifacts (models/ is gitignored, so a clone has none):
.venv\Scripts\python.exe scripts\fetch_models.py

# Diagnose the environment; every WARN/FAIL names its remedy:
.venv\Scripts\python.exe -m argus.cli doctor

# Real Hexagon NPU inference (needs the ARM64 venv and a matching QAIRT):
.\run.ps1 -Npu
.venv-npu\Scripts\python.exe -m argus.cli run --engine qnn-npu
```

### The two-venv split

`opencv-python` has **no win-arm64 wheel** — pip falls back to a from-source
numpy/meson build that fails without a C toolchain. `onnxruntime-qnn` and
`onnxruntime-genai` are **win-arm64 only**. One venv cannot host both, so
`run.ps1` creates an x86-64 `.venv` for the app and tests (it runs fine under
emulation on the X-Elite) and `-Npu` adds a native-ARM64 `.venv-npu` for the
NPU runtime.

---

## Commands

| Command | Purpose |
|---|---|
| `argus run` | Run the multi-camera triage pipeline |
| `argus doctor` | Check host, packages, artifacts, QAIRT, and engine availability |
| `argus config` | Print the effective configuration after flag overrides |
| `argus bootstrap` | Provision `models/` and the demo clip |
| `argus demo` | Regenerate the synthetic demo clip |

Useful `run` flags: `--engine {mock,onnx-cpu,qnn-npu}`, `--camera` (repeatable
— index or path), `--json-log`, `--http-port`, `--overlay-out`, `--window`,
`--max-ticks`, `--clock {auto,frame,wall}`.

Flags override config; config never overrides flags. `argus config` shows what
a run is actually tuned with.

---

## How the score works

Five weighted features, computed from numeric keypoints and boxes only:

| Feature | Weight | Signal |
|---|---:|---|
| `fall` | 0.40 | Sudden torso-centroid drop + bbox aspect flip (wider than tall) |
| `stillness` | 0.20 | Fraction of the ~2 s window with near-zero centroid motion |
| `occlusion` | 0.15 | Both hands *and* face below the keypoint-confidence threshold |
| `vlm_anomaly` | 0.15 | Caption matched against a closed vocabulary — free text never scores directly |
| `off_task` | 0.10 | Shoulder-line deviation from the station-facing angle |

Anything at or above `alert_threshold` (0.5) is surfaced with `reason_codes`
explaining why. Ties break on `trainee_id`, so the rank is stable across runs.

**These weights are unvalidated.** They are the prototype author's priors; no
incident has ever been scored with them. They now live in
[`configs/argus.default.toml`](configs/argus.default.toml), so retuning is a
config edit plus a `config_version` bump rather than a code change — see
[VALIDATION.md §2](docs/VALIDATION.md).

---

## Configuration

One versioned TOML holds every tunable: scoring weights and thresholds, model
paths, engine selection, camera sources, VLM cadence and top-K, tracker
parameters, and output sinks. Nothing in `src/argus/` hardcodes a tuning
constant.

`config_version` is validated on load, unknown keys are rejected rather than
ignored, and the weights must sum to 1.0. A typo'd `alert_threshhold` fails
loudly instead of silently leaving the default in place — an operator who
thinks they retuned the system and did not is the failure mode that matters.

---

## Layout

| Path | What it is |
|---|---|
| [`src/argus/triage.py`](src/argus/triage.py) | The deterministic scorer. Pure functions, stdlib only, no model runtime. |
| [`src/argus/config.py`](src/argus/config.py) | Versioned config loading and validation |
| [`src/argus/engines/`](src/argus/engines/) | One interface, three backends: `mock`, `onnx-cpu`, `qnn-npu` |
| [`src/argus/vision/`](src/argus/vision/) | Pre/post-processing per the real tensor contracts |
| [`src/argus/tracking/`](src/argus/tracking/) | Kalman motion + appearance re-identification |
| [`src/argus/pipeline/`](src/argus/pipeline/) | Multi-camera loop, VLM prefilter gate, overlay |
| [`src/argus/outputs.py`](src/argus/outputs.py), [`alerts.py`](src/argus/alerts.py) | The alert boundary. Import no image library. |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | Models, contracts, and **measured** NPU performance |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | What Argus has *not* been shown to do |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design and the reasoning behind it |

---

## Engines

| Kind | Detector | Pose | Super-res | Use |
|---|---|---|---|---|
| `mock` | synthetic tensors | synthetic tensors | synthetic tensors | Pipeline work with no artifacts |
| `onnx-cpu` | real | **unavailable** | real | Reference oracle, CI |
| `qnn-npu` | real | real | real | Production |

The mock does not short-circuit anything: it emits quantized tensors at the
real shapes, dtypes, and quantization ranges, so the real letterbox, YOLO-X box
decode, BlazePose SSD anchor decode, and landmark remap all run against it.
A mock that returned finished `Detection` objects would exercise none of that —
which is how the prototype's tensor contracts stayed wrong.

`onnx-cpu` has no pose stage because the BlazePose artifacts are QNN context
binaries with no CPU path. It raises rather than running a pipeline with pose
silently missing.

**No engine ever falls back quietly.** If `qnn-npu` cannot place a graph on the
Hexagon NPU it raises; `engine.allow_cpu_fallback` exists only for deliberate
A/B measurement and warns loudly when used.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

181 tests covering the scorer, config validation, the tensor contracts, the
CPU helpers, tracking and re-identification, the VLM gate, both output sinks,
and the pipeline end to end. Four of them are the gates that matter:

* **Real artifacts** — every shipped model is checked against its own
  `metadata.json`, and both ONNX artifacts are loaded and run with their output
  shapes and dtypes asserted. Skips cleanly when `models/` is unprovisioned;
  the `.bin` stages run under `-m npu` on hardware.
* **Determinism** — two independent process invocations over the same clip
  produce byte-identical output, with a companion test asserting the run was
  not simply empty.
* **Privacy** — no boundary module may import an image library, and no public
  callable on the boundary may accept an image-capable type. Checked by
  inspecting imports and type annotations.
* **Identity** — a track survives a 12-frame full occlusion with the same
  `trainee_id`, and two trainees crossing paths do not swap.

---

## Status

Runnable and tested today: the full pipeline in `mock` mode, the real YOLO-X
detector on `onnx-cpu` (contracts verified against the artifact), multi-camera
merged ranking, re-identification, all output sinks, and the CLI.

Not yet demonstrated: anything on real hardware or real footage. The NPU path
is written and contract-checked but has never executed — the host's QAIRT is
2.32.6 and the artifacts are 2.45.0. The VLM is a mock. No accuracy figure
exists for any model, because INT8 calibration used a placeholder set.

[docs/VALIDATION.md](docs/VALIDATION.md) is the honest list, with what each gap
would take to close.
