# Validation gaps

What Argus has **not** been shown to do. Each entry says what is unverified,
why it matters, and what closing it would take. Nothing here is a known bug —
these are claims the product cannot currently make.

Ordered by how much a wrong assumption would cost.

---

## 1. No real trainee-floor footage exists

**Status:** blocking for any accuracy claim.

Nothing in this repository has ever seen a real training floor.

* **INT8 calibration used a flagged placeholder set.** Quantization scales were
  fitted to data that is not the deployment distribution, so every accuracy
  figure that would follow from these artifacts is unverified.
* The consequence is already measurable, not hypothetical. `pose_detector.bin`'s
  `box_scores_*` were calibrated to an entirely non-positive logit range (scale
  5.5528, zero_point 255), leaving the detector with effectively two confidence
  values: 0.5 at the top quantization step and 0.004 at the next. See
  [PIPELINE.md §5](PIPELINE.md#5-what-the-placeholder-calibration-actually-did).
* `demo/trainees_demo.mp4` is procedurally drawn rectangles. It is a **pipeline
  fixture**: it exercises capture → track → triage → emit deterministically and
  offline, and a real YOLO-X will not fire on it. It cannot support an accuracy
  number of any kind.

**To close:** capture representative footage across the shifts, lighting, PPE,
and camera placements the product will actually run under; re-quantize against
it; re-run the AI Hub profile jobs; then measure detection and pose accuracy on
a held-out split.

**Owner decision needed:** how much footage, from which sites, and under what
consent and retention terms. Argus's privacy design keeps frames out of the
alert path at runtime, but a calibration corpus is stored imagery and needs its
own answer.

---

## 2. The five scoring weights have never been fitted to an incident

**Status:** blocking for operational trust in the rank.

`fall 0.40 / stillness 0.20 / occlusion 0.15 / vlm_anomaly 0.15 / off_task 0.10`
are the prototype author's priors. No incident has ever been scored with them.
`alert_threshold = 0.5` is equally unvalidated: nobody has measured how many
alerts an instructor would receive per hour, or what fraction would be real.

The weights now live in `configs/argus.default.toml` rather than in code, so
retuning is a config edit and a `config_version` bump — the mechanism is ready,
the evidence is not.

**To close:** build the labelled-clip harness. Concretely:

1. A corpus of clips, each labelled with what happened and whether an
   instructor was in fact needed.
2. A replay path that runs a clip through the pipeline and emits the full
   per-feature breakdown, not just the final score.
3. A fitting step that searches the weight simplex against a chosen operating
   point, and reports precision/recall at each threshold.

The pieces this needs already exist: the scorer is pure, its inputs are numeric
and serialisable, and `FrameClock` makes replay reproducible. What is missing
is the corpus and the labels.

**Owner decision needed:** what counts as "needed an instructor" — the
label definition is the experiment, and it cannot be inferred from the code.

---

## 3. Latency, power, and thermals under sustained multi-camera load

**Status:** blocking for a capacity claim ("how many cameras per box").

Measured today: single-graph NPU latency from AI Hub — 2,748 µs for YOLO-X,
421 + 425 µs for the two pose stages, 632 µs for super-resolution, all at 100%
NPU placement. Those are real ([PIPELINE.md §3](PIPELINE.md#3-measured-npu-performance)).

Never measured:

* End-to-end wall-clock frame time including CPU post-processing, tracking, and
  drawing.
* Behaviour with N > 1 cameras contending for one HTP.
* Sustained thermal behaviour over a shift, and where the SoC begins to throttle.
* Power draw in any mode. The prototype's `performance / balanced / efficiency`
  table (9.8 ms / 2,150 mW etc.) was invented; it has been deleted rather than
  carried forward.

**To close:** run the pipeline on the X-Elite against N real streams for hours,
recording frame time percentiles, NPU utilisation, package power, and skin
temperature. `argus run` already records per-camera frame counts and VLM call
counts; a latency histogram per stage is the missing instrumentation.

---

## 4. The NPU path has not executed on hardware

**Status:** blocking for the product's core premise.

The tensor contracts are now correct and enforced — read from each artifact's
`metadata.json`, checked on every call, and verified against the real files by
`tests/test_artifacts.py`, which loads and runs both ONNX artifacts. But:

* The two BlazePose stages are **QNN context binaries** and cannot run without
  the NPU. Their tests are marked `npu` and skip on any non-ARM64 host.
* The development host's QAIRT is **2.32.6.250402**; the artifacts were built
  with **2.45.0**. A context binary is tied to its producing runtime, so these
  two are expected to be incompatible. `argus doctor` reports the skew and
  `QnnNpuBackend` refuses rather than emitting an opaque QNN error.
* The `EPContext` wrapper path is written against ORT's documented mechanism
  and unit-tested for structure, but has not been executed against a real
  context binary on hardware.

**To close, in order:** install QAIRT 2.45.x (or recompile the pose artifacts
against 2.32.6); create the ARM64 venv with `run.ps1 -Npu`; run
`pytest -m npu`; then `argus run --engine qnn-npu` over real footage.

---

## 5. Re-identification is colour-histogram based, not learned

**Status:** known limitation, quantified only by construction.

`trainee_id` is a triage key — an instructor is dispatched to a *specific*
person — so an ID swap is a correctness failure. The tracker combines a Kalman
motion model with an HSV torso histogram, which is enough to hold identity
through occlusion and to separate trainees of different colours (both tested in
`tests/test_tracking.py`).

It will confuse **two trainees in identical PPE** who cross paths, and no
measurement of how often that happens on a real floor exists.

**To close:** either measure the swap rate on labelled real footage and accept
it, or replace `argus.tracking.appearance.signature` with a learned re-ID
embedding (OSNet-class, exported via AI Hub as a fourth NPU graph). The
interface is a drop-in; the cost is one more per-trainee inference.

---

## 6. The VLM is a mock

**Status:** the `vlm_anomaly` feature contributes 0.0 today.

`MockVLMCaptioner` returns a phrase containing no vocabulary term, so the
15%-weighted `vlm_anomaly` feature is inert and the rank is driven entirely by
pose and motion. Everything around it is real and tested: the prefilter gate,
the cadence, the top-K bound, the closed-vocabulary matcher, and
`GenieVLMCaptioner` itself.

**To close:** build a Genie / `onnxruntime-genai` bundle (Moondream2 or an
alternative) and point `vlm.bundle_dir` at it. This needs a **non-arm64 build
host** — `qai-hub-models` pulls torch, which has no win-arm64 wheel — so the
bundle is built elsewhere and staged onto the X-Elite.

Note that the closed vocabulary stays authoritative either way. A real VLM's
decoding is not deterministic, but only whether its caption contains one of
nine fixed phrases ever reaches the score, which is what keeps the rank
reproducible.

**Owner decision needed:** the vocabulary itself. Nine phrases were chosen by
the prototype author; which anomalies actually matter on this floor, and what
an instructor should be told about each, is a domain question.

---

## 7. The C++ runner was dropped, not ported

**Status:** deliberate omission.

The prototype's `cpp/argus_infer.cpp` was an unbuilt skeleton — real QNN C-API
wiring in shape, but never compiled, and its `#include <opencv2/opencv.hpp>`
plus QNN headers were never resolved against an SDK.

It is not carried into this repo. Neither `cmake` nor an MSVC toolchain is
present on the development host, so it could not be compiled, and shipping a
second unbuildable skeleton would repeat the prototype's central problem:
plausible-looking code that has never run.

**To close, if it is wanted:** the QAIRT SDK is installed at
`C:\Qualcomm\AIStack\QAIRT\qairt\2.32.6.250402` with headers under `include/QNN`
and `QnnHtp.dll` under `lib/arm64x-windows-msvc`, so the build is viable once a
toolchain is installed. The Python path is the reference; a C++ runner should
be written against the same `metadata.json` contracts and gated on the same
real-artifact tests.

**Owner decision needed:** whether a C++ runner is a product requirement at
all, or whether the Python path plus the NPU-resident graphs is sufficient.
