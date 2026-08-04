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
| NPU session bring-up (`Detector.kt`) | written; **does not yet work on hardware** — see below |
| Detector post-processing | **not written** — see below |
| Tracking, pose, triage scoring | not started; pending the split-point decision |
| Transport to the PC | interface only (`EdgeTransport`) |

## Open: the NPU has not executed a graph on hardware

`QnnSessionTest` runs on a connected device and currently **fails**. That is the
accurate status, not a broken build: it is the Android counterpart of
`docs/VALIDATION.md` §4, and it will keep failing until the Hexagon is actually
reachable.

Measured on a Galaxy S25 Ultra (`SM-S938U1`, `ro.soc.model=SM8750`, Android 16):

```
QNN SetupBackend failed Failed to create device.
Error: QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values
```

Eight provider configurations were tried — bare `backend_path`, three
`htp_performance_mode` values, explicit `htp_arch=79`, `htp_arch` with
`device_id`, a deliberate `htp_arch=75` mismatch, and fp16 precision. All fail
identically. Bumping the runtime from QAIRT 2.33.0 to 2.42.0 changed nothing.

Two things this *has* established:

- The backend library loads. The QNN GPU backend, used as a control through the
  same wiring, fails differently (`QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED`), so
  HTP is initialising and it is specifically device creation that is rejected.
- Native libraries must be extracted to disk. QNN takes a filesystem
  `backend_path`, and Android has not unpacked `.so` files by default since API
  23, so without `useLegacyPackaging = true` the EP silently fails to register
  and every node lands on CPU. The only reason that surfaced is that CPU
  fallback is explicitly disabled — with fallback on, this would have looked
  like a working app running at CPU speed.

### Root cause: a retail handset does not let an app touch the DSP

Measured on the device:

```
$ ls -lZ /dev/fastrpc-cdsp
crw-rw-r-- 1 system system u:object_r:vendor_qdsp_device:s0  /dev/fastrpc-cdsp
$ getenforce
Enforcing
```

QNN reaches the Hexagon through `/dev/fastrpc-cdsp`. It is owned by `system:system`
with no write bit for others, labelled `vendor_qdsp_device`, and SELinux is
enforcing. An app in `untrusted_app` cannot open it — nor can `shell` (uid 2000),
which was verified directly. Setting `ADSP_LIBRARY_PATH` does not help, because
the problem is access to the device node rather than skel resolution.

This is a property of the retail build, not of our code. Samsung ships its own
`/vendor/lib64/libsnap_qnn.so` for system-level use.

### NNAPI is not a way round it on Android 16

NNAPI exists to let apps reach vendor accelerators through a system HAL. Tested
here with the standard `onnxruntime-android` build (the `-qnn` build has no
NNAPI support compiled in), it runs — but:

```
Cannot list manifest for android.hardware.neuralnetworks@1.1::IDevice
ExecutionPlan::SimpleBody::finish: compilation finished successfully on nnapi-reference
```

`nnapi-reference` is NNAPI's CPU fallback driver. Android 16 removed the HIDL
neural-networks HAL, so there is no vendor NN driver left to dispatch to. NNAPI
on this device is a slower path to the same CPU.

### What this leaves

Nothing in this module can unblock it; the options are hardware or partnership
decisions:

| Route | Reaches NPU | Cost |
|---|---|---|
| Snapdragon 8 Elite QRD / dev device | yes | procurement; not a retail phone |
| Platform-signed or `userdebug` build | yes | OEM relationship or unlocked device |
| Samsung ENN SDK | yes | Samsung partner programme |
| LiteRT via Play Services | untested | worth a spike before anything else |
| Adreno GPU (Vulkan/OpenCL delegate) | no, but is a real accelerator | app-accessible today |
| AI Hub device farm | yes, off-device | already working; profiling only |

The last row is the one to lean on now: AI Hub runs jobs on real 8 Elite
hardware, so the model can be compiled, profiled, and numerically validated on
v79 without solving on-device access at all. That decouples the model question
from the deployment question, and the deployment question is the one that needs
a decision from outside this repository.

Until then, no phone-side inference number means anything.

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
