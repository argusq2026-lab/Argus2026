# Argus — architecture

## Overview

Argus is an on-device perception pipeline that watches many trainees at once
and produces a deterministic triage rank of who needs a human instructor. The
vision head runs on the Snapdragon X Elite Hexagon NPU; a small VLM samples
only trainees a numeric prefilter has already flagged.

Three properties drive the design, and each one is enforced by structure rather
than by convention:

1. **The rank is reproducible.** The scorer is a pure function of numeric
   history. Nothing non-deterministic upstream — including a VLM's decoding —
   can change that, because only a closed-vocabulary match ever reaches the
   score. Verified by two independent runs producing byte-identical output.
2. **Frames and captions cannot leak.** Not because they are filtered out, but
   because no sink has a parameter that could hold one, and the modules on the
   boundary import no image library. Verified by inspecting imports and type
   annotations.
3. **Nothing degrades quietly.** An engine that cannot reach the NPU raises. A
   backend that cannot load a role raises. A tensor that does not match its
   declared contract raises. A config with an unknown key raises.

## Target platform

**Windows AI PC — Snapdragon X Elite (X1E80100)**, Hexagon NPU v73, Adreno
X1-85, QAIRT 2.x. Artifacts are compiled for `sc8380xp` / HTP v73.

---

## System diagram

```
   camera 0            camera 1            camera N
      │                   │                   │
      ▼                   ▼                   ▼
  ┌────────────────────────────────────────────────┐
  │  Capture (OpenCV)                          CPU │
  └───────────────────────┬────────────────────────┘
                          │ BGR HWC frame
                          ▼
  ┌────────────────────────────────────────────────┐
  │  Letterbox -> RGB -> NCHW uint8             CPU │
  └───────────────────────┬────────────────────────┘
                          ▼
  ┌────────────────────────────────────────────────┐
  │  YOLO-X w8a8            2,748 us     NPU (100%) │
  └───────────────────────┬────────────────────────┘
                          │ boxes/scores/class_idx, 8400 anchors
                          ▼
  ┌────────────────────────────────────────────────┐
  │  De-letterbox + person filter + NMS         CPU │  (not in the graph)
  └───────────────────────┬────────────────────────┘
                          ▼
  ┌────────────────────────────────────────────────┐
  │  Kalman + appearance tracker -> trainee_id  CPU │
  └───────────────────────┬────────────────────────┘
                          │  per tracked trainee
       ┌──────────────────┴───────────────────┐
       ▼                                      │
  ┌─────────────────────────────┐             │
  │ QuickSRNet-Medium   632 us  │  NPU        │  only if crop < 2% of frame
  │ 128x128 -> 512x512          │             │
  └──────────────┬──────────────┘             │
                 ▼                            │
  ┌────────────────────────────────────────────────┐
  │  BlazePose detector      421 us    NPU (100%)   │  NHWC 128x128
  └───────────────────────┬────────────────────────┘
                          │ SSD: 512 + 384 anchors -> ROI
                          ▼
  ┌────────────────────────────────────────────────┐
  │  BlazePose landmarks     425 us    NPU (100%)   │  NHWC 256x256
  └───────────────────────┬────────────────────────┘
                          │ 25 landmarks (x, y, z, visibility)
                          ▼
  ┌────────────────────────────────────────────────┐
  │  BlazePose(25) -> COCO-17 remap             CPU │
  └───────────────────────┬────────────────────────┘
                          ▼
  ┌────────────────────────────────────────────────┐
  │  triage.py — deterministic scorer     CPU, pure │
  └───────────────────────┬────────────────────────┘
                          │  per-camera rank
                          ▼
  ┌────────────────────────────────────────────────┐
  │  VLM prefilter gate: score >= threshold,        │
  │  cadence due, top-K bounded                 CPU │
  └───────────────────────┬────────────────────────┘
                          ▼  only flagged trainees, ~1 Hz
  ┌────────────────────────────────────────────────┐
  │  VLM caption -> closed-vocabulary match    NPU  │  caption is a local;
  └───────────────────────┬────────────────────────┘  only the number survives
                          ▼
  ┌────────────────────────────────────────────────┐
  │  Merged rank across all cameras                 │
  └───────────────────────┬────────────────────────┘
                          │
        ══════════════════╪══════════════════  ALERT BOUNDARY
                          │  only {trainee_id, score, reason_codes, ts}
       ┌──────────┬───────┴────────┬──────────────┐
       ▼          ▼                ▼              ▼
   console    JSON lines     HTTP /triage    (overlay: local only,
                                              inside the boundary)
```

---

## Components

| Component | Module | Responsibility |
|---|---|---|
| Config | `argus.config` | Versioned TOML; every tunable, validated on load |
| Scorer | `argus.triage` | Pure functions over numeric history. stdlib only. |
| Engine interface | `argus.engines.base` | `ModelRunner` protocol + contract checking |
| Contracts | `argus.engines.metadata`, `.reference_specs` | Tensor contracts read from the artifact |
| Backends | `argus.engines.{mock,onnx_cpu,qnn_npu}` | Three implementations of one interface |
| Context binaries | `argus.engines.qnn_context` | EPContext wrapper + `qnn-net-run` |
| NPU sessions | `argus.engines.ort_qnn` | Vendored from QUAD-Client; no silent CPU fallback |
| Vision stages | `argus.vision.*` | Pre/post per the real contracts |
| Identity | `argus.tracking.*` | Kalman motion + HSV appearance re-association |
| Pipeline | `argus.pipeline.runner` | N cameras, per-source state, merged rank |
| VLM gate | `argus.pipeline.prefilter` | Pure function: threshold, cadence, top-K |
| Sinks | `argus.outputs`, `argus.alerts` | The alert boundary |

---

## Design decisions

### Contracts are read from the artifact, not written in the code

The prototype hardcoded tensor shapes, and all three were wrong: NHWC where the
detector wants NCHW, one fused `(1,N,6)` output where there are three separate
tensors, a heatmap decode for a model that emits no heatmaps, and a 128×128
input for a landmark stage that takes 256×256. None of it was caught, because
the real inference path had never executed.

So `argus.engines.metadata` parses each artifact's `metadata.json` into
`TensorSpec`s, `ValidatingRunner` checks every tensor in both directions on
every call, and `reference_specs.py` mirrors the contracts for mock mode with a
test asserting the two agree. A re-export that changes a shape now fails at
load, with a message naming the tensor.

### The mock emits tensors, not results

`MockBackend` produces quantized tensors at the real shapes, dtypes, and
quantization ranges, so the real decode paths run against it. That is why mock
mode is a legitimate development and CI target rather than a parallel
implementation that can drift.

The mock's synthetic scene and the demo clip's rendering share one definition
(`argus.synthetic`), so the boxes the tracker follows land on the pixels the
fixture actually contains — which matters because re-identification reads a
colour histogram of the crop.

### Identity is a correctness concern

`trainee_id` is what an alert points at, so a swapped ID sends an instructor to
the wrong person and resets the real one's triage history mid-incident. The
prototype's centroid tracker matched only against the last observed centroid,
kept no velocity, and deleted tracks after two seconds unseen.

The replacement keeps a Kalman estimate advancing through occlusions and
re-associates on combined motion + appearance cost, with greedy
globally-cheapest matching so assignment is deterministic. Its limits are
recorded in [VALIDATION.md §5](docs/VALIDATION.md): it will confuse two
trainees in identical PPE, and a learned re-ID embedding is the drop-in fix.

### The VLM gate is a pure function

The prototype's gate summed a hardcoded literal tuple, making it constant and
therefore always true — the VLM would have run for every tracked trainee, and
the top-K bound was applied only to the alert print. Since the whole latency
argument rests on that gate, it is now
`select_for_vlm(records, last_sample_ts, now, cfg, eligible_ids) -> list[str]`:
no I/O, no state, unit-testable without a camera or a model.

Ordering inside a tick follows from it — observations are pushed, the rank is
computed, and only then does the gate decide. A gate that reads a trainee's
real score cannot be evaluated before that score exists.

### Multi-camera from the start

"Many eyes" is the premise, so N sources is the base case. Each camera owns its
tracker (ids are namespaced `cam0-t3`, so two floors cannot merge into one
identity) and its own VLM budget (so a busy floor cannot starve a quiet one).
The rank is computed once over the union, because what an instructor needs is
one ordered list, not one per camera.

### Privacy by wiring

`TriageRecord` is frozen and has exactly four scalar fields. `argus.outputs`
and `argus.alerts` import no image library, so they cannot name a frame type
even by accident. VLM captions are locals inside one function: scored into a
number and dropped, never stored on a track.

`tests/test_privacy.py` asserts this by parsing the module ASTs for forbidden
imports and inspecting every public callable's type hints, so widening the
boundary fails CI rather than depending on review.

One honest caveat: `outputs.overlay_out` writes annotated frames to a video
file, which does persist imagery. That is an operator decision, so it is off by
default and stated in the config, the README, and the overlay module itself.

### Deterministic clock

Timestamps come from the tick index when every source is a file, and from a
monotonic clock when any source is live. That is what makes a replay
byte-identical on a fast machine and a slow one, and what makes the determinism
test meaningful rather than a coincidence of timing.

---

## Deployment

Local execution on the AI PC. `run.ps1` creates an x86-64 `.venv` for the app
and tests, and with `-Npu` a native-ARM64 `.venv-npu` for the NPU runtime — the
split is forced by `opencv-python` having no win-arm64 wheel while
`onnxruntime-qnn` is win-arm64 only.

`models/` is gitignored, so `argus bootstrap` reproduces the artifact tree from
the AI Hub job IDs in `models/argus_jobs.json` (or seeds it from a local copy)
and then verifies what it produced by loading it.

## Measured performance

See [docs/PIPELINE.md](docs/PIPELINE.md). Summary: 100% of all 744 profiled
layers run on the NPU across the four graphs; YOLO-X is 2,748 µs, the two pose
stages 421 + 425 µs, super-resolution 632 µs of which 74.3% is the single final
`DepthToSpace`.

End-to-end wall-clock, power, and thermals under sustained multi-camera load
are **not** measured — see [docs/VALIDATION.md](docs/VALIDATION.md).
