# Third-party notices

Argus is released under the [MIT License](LICENSE).

This is a hackathon build. Nothing below is a reason not to demo it — the
distinctions drawn are about *distributing an application*, which a demo is
not. They are recorded so the decision exists in writing if this ever becomes
something shipped.

---

## Vendored third-party files

Two binaries are committed to this repository and are not Argus's work:

| File | Origin | Licence |
|---|---|---|
| `android/gradle/wrapper/gradle-wrapper.jar` | [Gradle](https://gradle.org) build wrapper | Apache-2.0 |
| `android/app/src/androidTest/assets/pose_person_256.png` | Sample input from Qualcomm [`qai-hub-models`](https://github.com/quic/ai-hub-models) (`mediapipe_pose` asset store) | BSD-3-Clause |
| `android/app/src/main/assets/plank_lr.json` | Coefficients fitted on labelled data from [NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction) | MIT |

The Gradle wrapper is committed so a fresh clone can build the Android module
without a local Gradle install — the standard practice for the wrapper.

The pose sample is committed because `PoseRealPersonTest` needs a photograph of
a real, correctly-framed person to assert against. A synthetic figure cannot
serve: the landmark model does not respond to drawn people, which is the whole
reason that test exists alongside `PosePipelineTest`.

The plank coefficients are committed because they *are* the classifier — 26
floats per class, evaluated as arithmetic in `PlankClassifier.kt` rather than
loaded into a model runtime. Unlike the pose models below there is no artifact
to stage: this ships inside the APK. The upstream project is MIT, the same
licence as this repository, so no obligation beyond attribution attaches. Only
their labelled data was used; none of their code is present here, and their
own fitted model could not have been used regardless — its feature set needs
MediaPipe landmarks and a depth estimate COCO-17 and YOLO26-pose do not
provide. See `scripts/train_plank_model.py`.

(An earlier iteration of this project vendored an Apache-2.0 QNN Execution
Provider helper for on-device NPU inference; that entire local-inference
subsystem — and the vendored file with it — was removed when pose estimation and
form/exercise classification moved to each trainee's phone. See
[ARCHITECTURE.md](ARCHITECTURE.md).)

---

## Runtime dependencies — laptop (Python)

Installed from PyPI at their own licences, not redistributed here:

| Package | Licence |
|---|---|
| `websockets` | BSD-3-Clause |

`requirements-train.txt` (`scikit-learn` BSD-3-Clause, `numpy` BSD-3-Clause) is
offline tooling for `scripts/train_plank_model.py`, not a runtime dependency of
anything. Neither the server nor the phone imports either: the fitted model
ships as coefficients and is evaluated as arithmetic.

---

## Runtime dependencies — phone (Android)

Resolved by Gradle at their own licences, not redistributed in this repository.
They *are* packaged into a built APK, so this is the list that matters for
shipping.

| Artifact | Licence |
|---|---|
| `com.microsoft.onnxruntime:onnxruntime-android-qnn` | MIT |
| `com.qualcomm.qti:qnn-runtime` (transitive) | Qualcomm proprietary — see the AAR's own terms |
| `com.squareup.okhttp3:okhttp` | Apache-2.0 |
| `androidx.camera:camera-*`, `androidx.core`, `androidx.appcompat`, `androidx.activity`, `androidx.lifecycle` | Apache-2.0 |
| `junit:junit` (test only) | EPL-1.0 |
| `org.json:json` (test only) | Public Domain |
| `androidx.test:*` (test only) | Apache-2.0 |

---

## Model artifacts

Models are **not** in this repository — `models/` is gitignored and the Android
module's artifacts are staged onto the device by hand. They are provisioned from
Qualcomm AI Hub or exported locally, and each carries its **own** licence, which
is not this repository's. Anyone building a shippable product is responsible for
the terms of whichever model they stage.

| Model | Source | Licence |
|---|---|---|
| YOLO-X w8a8 detector | AI Hub job `jgo8m0l1p` | Apache-2.0 (YOLOX upstream) |
| BlazePose landmark | `qai_hub_models.models.mediapipe_pose`, weights via zmurez/MediaPipePyTorch | Apache-2.0 |
| **YOLO26-pose** (single-stage — **the pose path the station runs**) | `qai_hub_models.models.yolo26_pose`, weights from Ultralytics | **AGPL-3.0** |

### On the AGPL model specifically

`android/.../Yolo26Pose.kt` is Argus's own code under this repository's MIT
licence; it loads an ONNX file at runtime and contains no Ultralytics code. The
weights are not committed here. Nothing AGPL-licensed is therefore redistributed
by this repository.

That is a statement about **this repository**, not about a shipped product.
AGPL obligations attach to conveying a work to users, so an application
distributed with those weights is a different question, and the export step
itself is performed by the AGPL-licensed `ultralytics` package. Whether exported
weights are a derivative work of the training code is genuinely contested;
Ultralytics asserts that they are and sells commercial licences accordingly.
Development, evaluation and internal benchmarking are not conveying, so using
it today triggers none of this. **Distributing an application built on it
does**, and this model is now the station's chosen pose path rather than an
experiment — so that decision is live rather than hypothetical, and wants legal
review before any release.

Permissively-licensed alternatives with the same COCO-17 coverage are recorded
in `android/README.md` — `hrnet_pose` is MIT, `rtmpose_body2d` and `litehrnet`
are Apache-2.0. The two-model fallback path is deliberately retained in the app
partly so that switching remains cheap: those alternatives are two-stage and
would reuse its ROI and crop machinery.
