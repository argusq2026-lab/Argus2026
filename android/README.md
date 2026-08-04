# Argus Edge — Android station

The phone half of the system described in `docs/PROTOCOL.md`: one phone watches
one trainee, runs perception on its own NPU, and streams structured numeric
observations to the laptop's WebSocket ingest server. The laptop scores and
ranks; no frame, and nothing derived from a frame except the observation
fields, ever leaves the device.

## Status

| Piece | State |
|---|---|
| Gradle build → installable APK | `./gradlew assembleDebug` |
| Person detection (YOLO-X w8a8) on the Hexagon NPU | **working on a stock Galaxy S25 Ultra** — 9.6 ms/frame wall-clock at 640², CPU fallback disabled |
| Contract sidecar (`Sidecar.kt`) | required per model; generated from the real ONNX by `scripts/gen_yolox_fixture.py` |
| Decode (`decodeDetections`) | pure function, pinned to the deleted PC reference by fixture cases on host and device |
| NPU↔CPU parity | measured: worst 11 LSB score / 6 LSB box vs CPU reference; bounded at 16/9 in the fixture |
| Protocol client (`Protocol.kt`, `IngestClient.kt`) | encodes `hello`/`observation` per PROTOCOL.md, verified against the server's own parser via a shared fixture |
| User controls | live box overlay, start/stop, threshold slider, model import (file picker or adb), server connect dialog, camera flip, keep-screen-on |
| Pose model → real keypoints | **not started** — observations carry the protocol's zero-confidence keypoints, as PROTOCOL.md specifies for a phone with no pose estimate |
| Form/exercise classifier | not started (`form_reason_codes` always empty) |

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

## Open

- **Resampler parity**: Android's bilinear vs OpenCV's `INTER_LINEAR` are not
  bit-identical; with a w8a8 detector this can move borderline anchors. The
  parity fixture bypasses the resampler deliberately; quantifying its
  detection-level effect needs real footage on both platforms.
- **Pose + form models**: the next on-device milestones; the protocol fields
  are already carried, zeroed.
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
```

`connectedDebugAndroidTest` reinstalls the APK, which wipes `files/models/` —
restage after it if you then want to run the app itself.
