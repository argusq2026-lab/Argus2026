# Argus Edge — Android station

The phone half of the multi-edge design: a capture-and-compute node that scores
locally and reports only redacted records to the PC aggregator.

This module is **not** finished. What is here is the contract, the identity, the
capture loop, and a working NPU path — all of it building, tested, and running on
a stock Galaxy S25 Ultra. What is not here is the detector decode, for one reason
stated plainly below, and everything an end user would actually need.

## Status

| Piece | State |
|---|---|
| Gradle build → installable APK | done, `./gradlew assembleDebug` |
| Wire contract (`Wire.kt`) | done, checked against Python via a shared fixture |
| Station identity (`DeviceIdentity.kt`) | done |
| Camera capture (`MainActivity.kt`) | done, CameraX 640² analysis stream |
| NPU session bring-up (`Detector.kt`) | **works on real hardware** — graph runs on Hexagon v79, no CPU fallback |
| Detector post-processing | **not written** — see below |
| Tracking, pose, triage scoring | not started; pending the split-point decision |
| Transport to the PC | interface only (`EdgeTransport`) |

## Resolved: the NPU runs, and what it took

`QnnSessionTest` passes on a stock **Galaxy S25 Ultra** (`SM-S938U1`,
`ro.soc.model=SM8750`, Hexagon v79, Android 16, locked bootloader, SELinux
enforcing). A graph executes on the NPU with **CPU fallback explicitly
disabled**, so the pass cannot be an accidental CPU run.

No root, no unlock, no platform signing, no Qualcomm account, and no QAIRT SDK
download. Two things were required, and neither is obvious:

### 1. Declare the vendor libraries in the manifest

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false"/>
<uses-native-library android:name="libadsprpc.so" android:required="false"/>
```

This is the whole fix for the `QNN_DEVICE_ERROR_INVALID_CONFIG` failure. QNN
`dlopen`s the vendor cDSP client at runtime. Bundled `.so` files live in the
app's linker namespace, which cannot see `/vendor` by default; `libcdsprpc.so`
is listed in `/vendor/etc/public.libraries.txt`, and since API 31 an app must
*declare* such a library to be granted namespace access to it. Without the
declaration the load fails deep inside QNN and surfaces only as an invalid
device config, which points nowhere near the real cause.

### 2. Extract native libraries to disk

```kotlin
packaging { jniLibs { useLegacyPackaging = true } }
```

QNN takes a filesystem `backend_path`. Android has not unpacked `.so` files by
default since API 23, so without this there is no path to hand it, the execution
provider silently fails to register, and every node lands on CPU. The only
reason that surfaced is that CPU fallback is disabled — with fallback on it
would have presented as a working app quietly running at CPU speed.

### A red herring worth recording

`DspAccessTest` still reports `EACCES` opening `/dev/fastrpc-cdsp` directly, and
that is *fine*. The app never opens the node itself; the vendor client does,
from the vendor namespace. Reasoning from the mode bits on that node led to the
confident and wrong conclusion that retail hardware was closed to third-party
NPU use. It is not. Both tests are kept precisely because they disagree, and the
disagreement is the point: node permissions are not the access mechanism.

### Also ruled out along the way

NNAPI is not a route on Android 16. It runs, but:

```
Cannot list manifest for android.hardware.neuralnetworks@1.1::IDevice
compilation finished successfully on nnapi-reference
```

`nnapi-reference` is the CPU fallback driver; the HIDL neural-networks HAL is
gone, so there is no vendor driver to dispatch to. Irrelevant now that QNN
works, but worth not re-testing.

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
