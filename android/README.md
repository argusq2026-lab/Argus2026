# Argus Edge — Android station

The phone half of the multi-edge design: a capture-and-compute node that scores
locally and reports only redacted records to the PC aggregator.

This module is **not** finished. What is here is the contract, the identity, the
NPU seam, and the capture loop — all of it building, tested, and installable.
What is not here is inference itself, for one reason stated plainly below.

## Status

| Piece | State |
|---|---|
| Gradle build → installable APK | done, `./gradlew assembleDebug` |
| Wire contract (`Wire.kt`) | done, checked against Python via a shared fixture |
| Station identity (`DeviceIdentity.kt`) | done |
| Camera capture (`MainActivity.kt`) | done, CameraX 640² analysis stream |
| NPU session bring-up (`Detector.kt`) | done — opens a QNN session or refuses |
| Detector post-processing | **not written** — see below |
| Tracking, pose, triage scoring | not started; pending the split-point decision |
| Transport to the PC | interface only (`EdgeTransport`) |

## Why post-processing is deliberately absent

`QnnDetector.detect` throws `NotImplementedError`. That is a decision, not an
oversight.

The PC path reads every tensor contract from the artifact's own `metadata.json`
(`argus.engines.metadata`) rather than hardcoding shapes, because the prototype
hardcoded three and all three were wrong — NHWC where the detector wants NCHW,
one fused output tensor where there are three, a heatmap decode for a model that
emits no heatmaps. None of it was caught, because the real inference path had
never executed. `ARCHITECTURE.md` records this and `docs/VALIDATION.md` §7
records the related habit of shipping plausible code that has never run.

No SM8750 artifact exists yet, so there is no `metadata.json` here to read a
contract *from*. Writing a decode against a guessed shape would repeat the exact
mistake this repository was built to correct. It gets written when there is a
real artifact to write it against.

## What the phone needs staged, and what it does not

**Does not need staging:** the QNN backend. `onnxruntime-android-qnn` depends on
`com.qualcomm.qti:qnn-runtime`, published by Qualcomm to Maven Central, so
Gradle packages `libQnnHtp.so`, `libQnnSystem.so`, and the per-Hexagon skels —
including `libQnnHtpV79Skel.so`, which is the Snapdragon 8 Elite's — straight
into the APK. No Qualcomm account and no QAIRT SDK download is involved.

**Does need staging:** the model, at
`/data/data/com.argus.edge/files/models/yolox_sm8750.onnx`.

```bash
adb push yolox_sm8750.onnx /data/local/tmp/
adb shell run-as com.argus.edge mkdir -p files/models
adb shell "run-as com.argus.edge cp /data/local/tmp/yolox_sm8750.onnx files/models/"
```

## Target and versions

| | |
|---|---|
| Device | Snapdragon 8 Elite — `sm8750` (QRD) / `sm8750-ac` (Galaxy S25) |
| Hexagon | **v79** — the PC artifacts are v73 and will not load here |
| Bundled QAIRT | 2.33.0, via `qnn-runtime` |
| minSdk / compileSdk | 31 / 35, `arm64-v8a` only |

A QNN *context binary* is tied to its producing runtime, so one compiled against
a QAIRT other than 2.33.0 reproduces the version skew `docs/VALIDATION.md` §4
describes. Compiling to plain QDQ ONNX avoids the pinning — the execution
provider builds the graph for whatever HTP it finds at session init — at the
cost of a slower first load. That is why the staged artifact above is `.onnx`.

## The two contracts this module is held to

Both are shared files, checked by both languages, so the platforms cannot drift:

- `tests/data/wire_vectors.json` — what a station may say. Kotlin's encoder must
  produce it (`WireVectorsTest`); Python's decoder must read it
  (`tests/test_wire_vectors.py`). Regenerate: `python scripts/gen_wire_vectors.py`.
- `tests/data/scorer_vectors.json` — what a score means. Binding once scoring
  moves on-device. Regenerate: `python scripts/gen_scorer_vectors.py`.

Equality is on decoded values, not bytes: Kotlin and Python format doubles
differently, and the PC parses the payload rather than comparing it.

## Open question this module cannot settle

`Preprocess.kt` letterboxes with a textbook bilinear filter; the PC uses
`cv2.resize(INTER_LINEAR)`, a fixed-point kernel. These are not guaranteed to
agree bit for bit, and the detector is w8a8, so a one-LSB input difference can
move an anchor across a quantisation step. Since phone-vs-PC comparison is how
this port gets validated, that has to be settled — either mandate OpenCV-Android
here for exactness, or accept a tolerance and measure its detection-level impact
on real footage. Until then, treat any phone/PC detection difference as
unexplained rather than blaming the model.

## Build

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export ANDROID_HOME=~/Library/Android/sdk
cd android && ./gradlew assembleDebug testDebugUnitTest
```
