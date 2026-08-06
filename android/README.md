# Argus Edge — Android station

The phone half of the system described in `docs/PROTOCOL.md`: one phone watches
one trainee, runs perception on its own NPU, and streams structured numeric
observations to the laptop's WebSocket ingest server. The laptop scores and
ranks; no frame, and nothing derived from a frame except the observation
fields, ever leaves the device.

> **Scope: hackathon build.** Everything documented here was measured on a real
> Galaxy S25 Ultra and the tests are honest about what they do and do not cover
> — but no accuracy claim is supported, because no real trainee footage exists.
> Demoing and developing against the AGPL-3.0 pose model triggers nothing;
> distributing an application built on it is the decision to revisit.

## Status

| Piece | State |
|---|---|
| Gradle build → installable APK | `./gradlew assembleDebug` |
| **Detection + COCO-17 pose (YOLO26-pose)** | **the pose path** — single-stage on the Hexagon NPU, ~22 ms/frame at 640², all people landmarked, CPU fallback disabled. AGPL-3.0, see below |
| YOLO-X w8a8 detector | fallback path, used when `yolo26_pose_fp32.onnx` is absent — 9.6 ms/frame |
| Contract sidecar (`Sidecar.kt`) | required per model; generated from the real ONNX by `scripts/gen_yolox_fixture.py` |
| Decode (`decodeDetections`) | pure function, pinned to the deleted PC reference by fixture cases on host and device |
| NPU↔CPU parity | measured: worst 11 LSB score / 6 LSB box vs CPU reference; bounded at 16/9 in the fixture |
| Protocol client (`Protocol.kt`, `IngestClient.kt`) | encodes `hello`/`observation` per PROTOCOL.md, verified against the server's own parser via a shared fixture |
| Orientation | rotates freely; the activity survives the config change, so the NPU sessions and the WebSocket are not torn down |
| User controls | one-line status chip (state + colour), large Start/Stop, server dialog, camera flip; diagnostics, threshold slider and model import behind **Debug** |
| Connection resilience | auto-reconnect with capped backoff on transport drops; protocol refusals stay terminal and say so |
| Server discovery (`Discovery.kt`) | **written, not yet built or run on a device** — "Find server on this network" listens for the laptop's UDP beacon, lists sessions by the instructor's name, and fills the address in; a human still presses Connect. Parsing is covered by `DiscoveryTest` (host); the receive path needs a device on real Wi-Fi. See PROTOCOL.md, "Discovery" |
| Join approval | **written, not yet built or run on a device** — `hello` carries an optional display name and the chosen session; `join_pending` is its own client state, so a station awaiting an instructor's approval says so instead of looking hung. Refusals stay terminal, including a declined or unanswered join. `JoinPendingTest` (host) covers the parsing; the server side is verified in `tests/test_admission.py`. See PROTOCOL.md, "Admission" |
| BlazePose landmarks | fallback path — fp16, upper-body only (no knees/ankles), ROI-cropped per person |
| Subject selection | `SubjectTracker` — largest box with hysteresis, switches counted on screen |
| End-to-end to the laptop | **working** — station appears in `GET /triage` over `adb reverse` or LAN |
| **Form classifiers** | **working** — `FormClassifier.kt`, a logistic regression evaluated in Kotlin arithmetic, one instance per exercise (plank 3x26, bicep 2x16, lunge 2x18). No ML runtime, no ONNX, no second NPU dispatch. Refit from an MIT dataset; see below |
| Geometric form checks | **working** — `GeometricFormChecks.kt`, a second fault per exercise upstream never collected ML labels for: `loose_upper_arm` (bicep, elbow-shoulder angle vs. vertical) and `knee_angle_out_of_range` (lunge, hip-knee-ankle angle, gated behind the same `depth_gate` as `knee_over_toe`). Both thresholds are upstream's own stated cutoffs, unvalidated here — see `docs/VALIDATION.md` §1c/§1d |
| Rep counting | **working, bicep and lunge only** — `RepCounter.kt`, a debounced peak/valley crossing on the same joint angle the geometric checks use (elbow flex for bicep, knee angle for lunge). Counts a rep regardless of form quality — a flagged rep still counts — and is display-only on the wire (`rep_count`, `docs/PROTOCOL.md`), never scored. Plank has no counter: it is a hold, not a rep. The hysteresis thresholds are a hand-picked heuristic, not fit or measured against real reps — see the class docstring |
| Other exercises | not started (`form_reason_codes` empty for any exercise without a shipped `<exercise>_lr.json`; see `docs/ADDING_AN_EXERCISE.md`) |

## How detection stays honest

Three fixtures, all generated from the real artifact by scripts in this repo,
all shared verbatim between the Python and Kotlin test suites (`tests/data` is
mounted into both — never copied):

- **`<model>.json` sidecar** — I/O names, shapes, and output quantization
  parameters read from the graph's own QuantizeLinear initializers. The app
  refuses a model without a sidecar and refuses a session whose live I/O
  disagrees with it. Nothing about the tensor contract is hardcoded in Kotlin;
  scale `4.4157/51` lives in the artifact's sidecar, not the code.
- **`tests/data/yolox_parity.json`** — CPU-reference raw outputs on a
  procedural input (`(x*7 + y*13 + c*31) % 256`), plus crafted decode cases
  with known person anchors. `DecodeTest` (host) pins the Kotlin decode to the
  Python reference exactly; `NpuParityTest` (device) re-runs the frozen input
  on the Hexagon and holds it to the measured tolerance.
- **`tests/data/protocol_vectors.json`** — canonical protocol messages passed
  through `argus.ingest.protocol` *at generation time*. `ProtocolTest` (host)
  proves the Kotlin encoder reproduces them; `tests/test_protocol_vectors.py`
  proves the server parses them. One file, both ends of the wire.
- **`tests/data/<exercise>_vectors.json`** (one per shipped classifier) — real
  held-out feature vectors with the probabilities the *fitted* scikit-learn
  model produced for them. `FormClassifierTest` (host) requires the Kotlin
  reimplementation to reproduce them to 1e-6, and `tests/test_form_artifacts.py`
  re-derives them from the shipped coefficients in stdlib Python, for every
  `*_lr.json` artifact this build ships. A transposed feature order, a skipped
  standardization, or one-vs-rest sigmoids instead of a softmax all still
  yield plausible probabilities — only comparison against the fitted model
  catches them.

The measured NPU↔CPU divergence (11 LSB ≈ 0.042 in score units) is stated, not
hidden: a borderline detection near the 0.35 threshold can legitimately differ
between the phone and a CPU replay of the same frame.

## Staging a model

Models are gitignored artifacts. Generate the sidecar and fixture once:

```bash
python scripts/gen_yolox_fixture.py path/to/yolox.onnx   # writes yolox.json next to it
```

Then either use the **Model** button in the app (multi-select `yolox.onnx`,
`yolox.data`, `yolox.json` in the system picker) or stage over adb:

```bash
adb push yolox.onnx yolox.data yolox.json /data/local/tmp/
adb shell "run-as com.argus.edge mkdir -p files/models && \
  run-as com.argus.edge cp /data/local/tmp/yolox.onnx /data/local/tmp/yolox.data \
  /data/local/tmp/yolox.json files/models/"
```

The current artifact is the source model of AI Hub job `jgo8m0l1p`
(`yolox-onnx-w8a8-clean`, sha `1d9ae4a4…`): QDQ ONNX, so the QNN EP compiles it
for whatever Hexagon it finds at session init — the same file serves v73 and
v79. (The *compiled* target of that job is an `sc8380xp` context binary and
will not load on a phone; use the source, not the target.)

## How the NPU became reachable (kept for the next device bring-up)

A stock retail phone can run QNN on the Hexagon with no root, no Qualcomm
account, and no QAIRT SDK. Two non-obvious requirements, both found the hard
way on this S25 Ultra (`SM-S938U1`, Android 16, locked bootloader, SELinux
enforcing):

1. **`<uses-native-library>` for `libcdsprpc.so`/`libadsprpc.so`** in the
   manifest. QNN dlopens the vendor cDSP client; app linker namespaces cannot
   see `/vendor` unless the library is declared (API 31+). Undeclared, the
   failure surfaces as `QNN_DEVICE_ERROR_INVALID_CONFIG` — nowhere near the
   cause. (Direct `open("/dev/fastrpc-cdsp")` still returns EACCES; that is
   irrelevant, the vendor client opens it. `DspAccessTest` and `QnnSessionTest`
   are kept side by side precisely because they disagree.)
2. **`useLegacyPackaging = true`** so the `.so` files exist on disk for QNN's
   filesystem `backend_path`. Unextracted, the EP silently fails to register —
   visible only because CPU fallback is disabled.

Also ruled out: NNAPI on Android 16 binds to `nnapi-reference` (CPU) — the
HIDL neural-networks HAL is gone. Measured dispatch overhead: ~500 µs per NPU
call on this device (see `NpuEvidenceTest`), ~5% of a YOLO-X frame but larger
than an entire BlazePose stage — batch or down-cadence pose when it lands.

## The pose path: YOLO26-pose (single-stage)

This is what the station runs. `stage.sh` stages it by default; the two-model
path (YOLO-X + BlazePose) remains as a fallback and takes over automatically if
`yolo26_pose_fp32.onnx` is absent. The status strip names the active backend, so
which one is running is never a guess.

The fallback is kept deliberately rather than deleted. If the licence below
forces a move to `hrnet_pose` (MIT) or `rtmpose_body2d` (Apache-2.0), both are
**two-stage** top-down models — they would reuse the ROI derivation, the crop,
and the per-person dispatch that path already implements, including the measured
2.2x ROI scale that took a sweep to find.

Why it was adopted: the 25-point BlazePose export is upper-body only, and four of the
seven `form_reason_codes` the protocol defines — `insufficient_depth`,
`knee_valgus`, `heels_rising`, partly `incomplete_lockout` — need knees or
ankles it structurally cannot produce. Measured against BlazePose on four images:

| | BlazePose | YOLO26-pose |
|---|---|---|
| keypoints | 13/17 | 16/17 |
| legs (COCO 13-16) | **0/4, structurally** | **4/4** |

It is also much simpler. One pass, one coordinate space, confidences already
activated: no ROI to derive (the bug that made pose look broken), no crop, no
25-to-COCO remap, no visibility-logit convention, no quantization parameters,
and one NPU dispatch rather than two.

**Licence.** The weights are Ultralytics YOLO26, **AGPL-3.0**, against this
repository's MIT. Shipping it commercially means releasing the app under AGPL or
buying an Ultralytics licence — a business decision, and the reason this is a
flag rather than a replacement. `rtmpose_body2d` is the Apache-2.0 alternative
(COCO-17 with legs) if that is unacceptable; it needs `mmcv`, which is a
difficult build, and it is two-stage so the ROI question returns.

Export it with:

```python
from qai_hub_models.models.yolo26_pose.model import Yolo26PoseDetector
import torch
m = Yolo26PoseDetector.from_pretrained().eval()
torch.onnx.export(m, torch.zeros(1,3,640,640), "yolo26_pose_fp32.onnx",
                  opset_version=17, dynamo=False,
                  input_names=["image"], output_names=["boxes","scores","keypoints"])
```

Note the input is float32 in [0, 1], not the w8a8 detector's raw uint8.

## The form classifiers

Three exercises ship a classifier today: plank, bicep, lunge. Each is
`assets/<exercise>_lr.json`, a multinomial logistic regression (26/16/18
features respectively), refit by `scripts/train_form_model.py --exercise
<name>` from
[NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)
(MIT). `FormClassifier.kt` is exercise-agnostic — it reads landmarks, feature
kinds, classes, and the code mapping from whichever artifact it is given, so
what follows about plank (the first one built, and the one the traps below
were found on) applies structurally to all three; see
`docs/ADDING_AN_EXERCISE.md` for what differs per exercise and
`docs/VALIDATION.md` §1b–§1d for each one's accuracy figure and caveats —
bicep's in particular (0.63) is materially weaker than plank's or lunge's
(0.99 each) and should not be trusted operationally yet.

Their pickled models could not be used directly — many of their features do
not exist on this wire:

- `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index` — COCO-17
  stops at the ankles.
- a MediaPipe `z` depth estimate per landmark — YOLO26-pose emits none.
- a MediaPipe **visibility** per landmark, dropped deliberately. See below.

It runs as arithmetic, not a model. Standardize, multiply by a 3x26 matrix,
add an intercept, softmax: no ONNX session, no third artifact to stage, and no
second NPU dispatch on a path where dispatch alone costs ~500 us. Below the
0.6 probability threshold it reports no codes at all rather than a best guess,
which is the source project's own "unknown" behaviour.

Features are normalized to the box the landmarks span — deliberately *not*
the detector's `bbox_xyxy`, since MediaPipe's person box and YOLO26's are not
the same convention, and the model was fitted against the landmark extent.
`FormClassifier.normalizeToLandmarkBox` and
`train_form_model.bbox_normalize` must stay identical.

Lunge carries a second gate beyond landmark visibility: its `knee_over_toe`
label was only ever collected — and, per upstream's own detection code, only
ever evaluated — at the bottom of a lunge. The artifact's `depth_gate` bounds
`mean(ankle_y) - mean(hip_y)` to the training data's own observed range;
outside it `FormClassifier` declines rather than naming a class the label was
never fit against. See docs/VALIDATION.md §1d.

### Visibility is not confidence, and it cost a rebuild

The first revision kept the visibility columns, on the assumption that
MediaPipe visibility and YOLO26 keypoint confidence are the same kind of
number. They are not. MediaPipe visibility saturates: on this dataset
`nose_v` has mean 0.9993 and a standard deviation of **0.0013**. Standardizing
an entirely ordinary YOLO26 confidence of 0.85 against that gives **-116
sigma**, and 13 such features saturate the softmax before the pose is
consulted.

The symptom on a real device was a model that reported 100% confidence on
whatever the camera happened to see, and reported a form error for a correct
plank. It was reproduced offline by taking a known-correct plank from the
fixture and substituting realistic confidences for the MediaPipe ones: the
prediction flips from `C` at 100% to `L` at 100% with the pose untouched.

The rule: a feature transfers between two pose models only if both mean the
same thing by it. Coordinates do. Confidence does not. Confidence now gates
the verdict instead — `MIN_VISIBLE_LANDMARKS` requires 10 of the 13 landmarks
above 0.3 before the classifier will judge at all, which also covers the
separate problem that a three-class softmax has no way to say "that is not a
plank".

**None of the reported accuracies are a claim about a trainee.** They are
held-out *frames* from each source dataset's own recordings — plank's
frame-level leakage was ruled out, but none of the three is a per-subject
estimate, and bicep's 0.63 is weak even on those terms. docs/VALIDATION.md
§1b–§1d.

Retrain with:

```bash
pip install -r requirements-train.txt
python scripts/train_form_model.py --exercise plank   # or bicep, lunge, all
```

Each exercise's artifact and fixture come out of one run; committing only one
is what `test_artifact_and_fixture_agree_on_encoding_and_threshold` notices.

## Open

- **Resampler parity**: Android's bilinear vs OpenCV's `INTER_LINEAR` are not
  bit-identical; with a w8a8 detector this can move borderline anchors. The
  parity fixture bypasses the resampler deliberately; quantifying its
  detection-level effect needs real footage on both platforms.
- **Pose accuracy is unvalidated.** The path runs and the invariants hold
  (confidences in [0,1], keypoints inside their ROI, unmapped joints exactly
  zero), but nothing here shows the landmarks are anatomically right on a real
  person — `PosePipelineTest` uses a synthetic figure precisely because it
  tests wiring, not accuracy. This is docs/VALIDATION.md §1 restated for the
  phone: no real footage, no accuracy claim.
- **A depicted person is a person.** The detector fires on any depiction —
  someone on a monitor, a poster, a photograph, or a reflection. Gyms are full
  of mirrors, so a badly placed station can acquire a phantom trainee and report
  it with full confidence; nothing downstream can tell, because the observation
  is well-formed. `SubjectTracker`'s largest-box rule helps by accident (a real
  trainee in front of the phone is much larger than a person on a screen across
  the room) but was not designed for this and does not solve it. Mitigations if
  it bites, none needing a new model: a minimum box-area floor, centre-weighted
  selection, or an operator-set region of interest. Observed during bring-up,
  not yet a problem in any real placement.
- **One trainee is reported, by design.** A `PROTOCOL.md` observation has one
  `bbox_xyxy` and one 17-keypoint set, and one connection carries one
  `trainee_id`. Multiple people are detected and (up to a budget) landmarked so
  the operator can see who is in frame while placing the phone, but only the
  subject's pose crosses the wire. Reporting several people would need a
  protocol change, not an app change.
- **Only 13 of 17 keypoints can ever light up.** The 25-point BlazePose
  export has no knees or ankles, so COCO 13-16 stay at exactly zero confidence
  and no leg bones are drawn. Their absence on screen is accurate, not a
  rendering bug — and the status line reads `n/13` for that reason.
- **No ROI rotation.** MediaPipe rotates the landmark ROI to the hip→shoulder
  axis; this takes an axis-aligned square, the same simplification the PC
  pipeline made and recorded. A trainee lying down is fed upright-boxed, which
  is exactly the case fall detection cares about.
- **Form classifier**: not started; `form_reason_codes` always empty.
- The AI Hub profile job on `Snapdragon 8 Elite QRD` (`jp2e44lxp`) failed with
  an infra-side "unexpected device error"; on-device measurement supersedes it
  for now.

## Build & test

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export ANDROID_HOME=~/Library/Android/sdk
cd android
./gradlew assembleDebug testDebugUnitTest        # host: build + decode/protocol tests
./gradlew connectedDebugAndroidTest              # device: NPU + parity (needs staged model)
./stage.sh --models <dir-with-yolox.onnx>        # build, install, stage, launch
```

> **`connectedDebugAndroidTest` uninstalls the app when it finishes** — both the
> app and the test APK, which also destroys `files/models/`. On the phone the
> symptom is simply that nothing happens: no app, no model, no explanation.
> Always re-run `./stage.sh` after a device-test run. That is what the script is
> for.
