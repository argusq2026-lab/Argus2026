# Setup — installing Argus from scratch

Everything needed to go from a bare machine to a running system, on both
halves: the **laptop** (ingest server, triage, console) and the **phone**
(Android station that does the looking). Every dependency is named, with where
it comes from and which step installs it.

Once installed, [USAGE.md](USAGE.md) is how you run it.

> **You do not need a phone to run Argus.** The laptop half is complete on its
> own and ships a fixture replayer that stands in for phones over a real
> WebSocket. If you only want to see the system work, do [Part 1](#part-1--the-laptop)
> and stop — that is a five-minute install with one dependency.

---

## What you are installing

| Half | Runs on | Language / runtime | Third-party runtime deps | Needed for a demo? |
|---|---|---|---|---|
| **Server + console** | Laptop (Windows, macOS, Linux) | Python 3.11+ | **one** — `websockets` | Yes |
| **Station app** | Android phone with a Qualcomm Hexagon NPU | Kotlin / Android SDK 35 | CameraX, ONNX Runtime QNN, OkHttp (all fetched by Gradle) | No — optional |
| **Perception models** | Staged onto the phone | ONNX artifacts | exported locally, not committed | Only with a phone |

The laptop loads **no model and runs no inference** — that is a design property,
not an omission (see [ARCHITECTURE.md](ARCHITECTURE.md)). There is no CUDA, no
OpenCV, no camera library and no ML runtime anywhere in the laptop install.

---

# Part 1 — The laptop

**Verified by CI** on Windows and Ubuntu, Python 3.11 and 3.12
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## 1.1 Prerequisites

| Need | Version | Why |
|---|---|---|
| Python | **3.11 or newer** | `tomllib` and the typing syntax the config loader uses |
| git | any | to clone the repository |
| A browser | any | the console is a web page served by the server |

No compiler, no GPU, no admin rights. Disk footprint under 50 MB including the
virtual environment.

**Install the prerequisites** if the machine does not have them:

```powershell
# Windows (PowerShell)
winget install Python.Python.3.11
winget install Git.Git
```

```bash
# macOS (Homebrew)
brew install python@3.11 git
```

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
```

Confirm before continuing — the version must be 3.11 or higher:

```bash
python --version        # or python3 --version
```

## 1.2 Get the code

```bash
git clone https://github.com/argusq2026-lab/Argus2026.git
cd Argus2026
```

## 1.3 Install

Pick **one** of the two paths. They produce the same `.venv`.

### Path A — Windows, one command

```powershell
.\run.ps1
```

Installs [uv](https://docs.astral.sh/uv/) if it is missing, creates `.venv`
with Python 3.11, installs `requirements.txt`, then installs Argus itself as an
editable package. Use `.\run.ps1 -Python 3.12` to pick a different interpreter.

> If PowerShell refuses to run the script (`running scripts is disabled on this
> system`), either use Path B or allow it for this session only:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Path B — any OS, stdlib only

No extra tooling — `venv` and `pip` ship with Python.

```bash
python -m venv .venv

# activate it:
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
.venv\Scripts\activate.bat         # Windows cmd.exe

python -m pip install --upgrade pip
pip install -r requirements.txt    # websockets, pytest, pytest-timeout
pip install -e .                   # argus itself + the `argus` command
```

Every command in this repository's docs is written two ways: `argus …` assumes
the virtual environment is **activated**, and
`.venv\Scripts\python.exe -m argus.cli …` works without activating it. They are
identical in effect.

## 1.4 Verify the install

Three checks, in increasing order of thoroughness:

```bash
argus --version
# argus 0.1.0

argus doctor
# 10 checks covering config validity, whether the ingest port is bindable,
# which LAN address a phone should be pointed at, and whether discovery is
# advertising. WARN lines are advice; FAIL lines are things to fix.

pytest tests/ -q
# 433 passed   (count as of this writing; it only grows)
```

Then prove the whole pipeline end to end, with no phone involved — this is
[USAGE.md § Path A](USAGE.md#path-a--no-phone-fixture-replay) in short form:

```bash
argus demo --ticks 200                       # write a 5-station fixture
argus run --http-port 8080 --max-ticks 120   # in one terminal
argus replay --speed 1.0                     # in another
# then open http://127.0.0.1:8080/
```

## 1.5 Optional — the standalone binary

Freezes the laptop half into one executable so the machine that runs a floor
needs no Python at all. Launched with no arguments it starts the server and
opens the console.

```bash
pip install -e ".[build]"       # adds pyinstaller, build-time only
python scripts/build_binary.py  # writes dist/argus  (dist\argus.exe on Windows)

./dist/argus                    # or double-click it
```

The build self-checks that the binary runs from a directory that is **not** the
repository, because a binary that only works beside its own source tree is the
failure worth catching before handing one to somebody.

## 1.6 Optional — retraining the form classifiers

Only needed to refit an exercise model; the fitted coefficients are already
committed as `android/app/src/main/assets/<exercise>_lr.json`.

```bash
pip install -r requirements-train.txt        # scikit-learn, numpy
python scripts/train_form_model.py --exercise plank      # or bicep, lunge, all
```

Deliberately kept out of `requirements.txt`: the CI import-hygiene job asserts
that importing the scorer pulls in no ML runtime at all. See
[docs/ADDING_AN_EXERCISE.md](docs/ADDING_AN_EXERCISE.md).

## 1.7 Firewall — only if a phone will connect over Wi-Fi

Not needed for the fixture replay, and not needed over USB.

| Port | Protocol | Direction | What it is |
|---|---|---|---|
| **8765** | TCP | inbound to the laptop | WebSocket ingest — every phone connects here |
| **8766** | UDP | outbound broadcast from the laptop | The discovery beacon phones listen for |

Windows prompts on the first `argus run`; allow it on **Private** networks. To
add the rule ahead of time, in an elevated PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Argus ingest" -Direction Inbound `
    -Protocol TCP -LocalPort 8765 -Profile Private -Action Allow
```

macOS prompts the same way on first run. On Linux with `ufw`:
`sudo ufw allow from 192.168.0.0/16 to any port 8765 proto tcp`.

Guest and enterprise Wi-Fi often block both client-to-client traffic and
broadcast. If discovery finds nothing, the address can always be typed by hand
— and if the network isolates clients entirely, use the USB path in
[§3.1](#31-usb-recommended-for-development).

---

# Part 2 — The phone

**The app requires a Qualcomm phone.** Perception runs on the Hexagon NPU with
CPU fallback *disabled on purpose*, so a device without one reports that it
cannot run rather than quietly running slowly and wrongly.

| Requirement | Value | Note |
|---|---|---|
| SoC | Qualcomm Snapdragon with a Hexagon NPU | Developed and measured on a **Galaxy S25 Ultra** (`SM-S938U1`, SM8750, Hexagon v79) |
| ABI | `arm64-v8a` only | Shipping a 32-bit slice would install and then fail to load the QNN backend |
| Android | **12 or newer** (API 31) | `<uses-native-library>`, which is what makes the NPU reachable at all |
| Camera | any rear camera | |
| Free space | ~150 MB | The debug APK is ~90 MB (it bundles the QNN native libraries), plus staged models |
| Root / unlocked bootloader | **not needed** | Stock retail phone, locked bootloader, SELinux enforcing |
| Qualcomm account or QAIRT SDK | **not needed** | ONNX Runtime QNN comes from Maven Central |

## 2.1 Prepare the phone

1. **Settings → About phone → Software information → tap "Build number" seven
   times** to unlock Developer options.
2. **Settings → Developer options → USB debugging: on.**
3. Connect the USB cable and set the phone's USB mode to **File transfer**, not
   charge-only.
4. Accept the *Allow USB debugging?* RSA fingerprint prompt on the phone.

Confirm from the laptop — the device must be listed as `device`, not
`unauthorized`:

```bash
adb devices
```

On Windows, if nothing is listed, install the manufacturer's USB driver
(Samsung: "Samsung USB Driver for Mobile Phones"); `adb` itself is installed in
the next step.

## 2.2 Install the build toolchain

| Need | Version | Where from |
|---|---|---|
| JDK | **17** | Temurin, Zulu, or Android Studio's bundled JBR |
| Android SDK Platform | **35** | `sdkmanager` or Android Studio |
| Android SDK Build-Tools | 35.x | same |
| Android SDK Platform-Tools | any current | provides `adb` |
| Gradle | 8.13 | **not installed by you** — `./gradlew` downloads it |

Everything else — AGP 8.9.2, Kotlin 2.0.21, CameraX 1.4.1, OkHttp 4.12.0, and
`onnxruntime-android-qnn` 1.28.0 (which transitively brings
`com.qualcomm.qti:qnn-runtime`, i.e. `libQnnHtp.so` and the per-Hexagon skels)
— is declared in [`android/app/build.gradle.kts`](android/app/build.gradle.kts)
and fetched from Maven Central by the first build. Expect that first build to
pull a few hundred MB and take several minutes; later builds are incremental.

### Easiest — Android Studio

Install [Android Studio](https://developer.android.com/studio), open the
`android/` folder, and accept the SDK components it offers. It installs the
JDK, the SDK, and platform-tools, and writes `android/local.properties` itself.

### Headless — command-line tools only

```bash
# 1. JDK 17
#    macOS:   brew install --cask temurin@17
#    Ubuntu:  sudo apt install -y openjdk-17-jdk
#    Windows: winget install EclipseAdoptium.Temurin.17.JDK

# 2. Android command-line tools, then the SDK packages
#    Download "Command line tools only" from developer.android.com/studio
#    and unzip to $ANDROID_HOME/cmdline-tools/latest
export ANDROID_HOME="$HOME/Android/sdk"          # macOS: ~/Library/Android/sdk
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"

sdkmanager --install "platform-tools" "platforms;android-35" "build-tools;35.0.0"
sdkmanager --licenses                            # accept them all
```

Set the two environment variables the build reads, in every shell that builds:

```bash
export JAVA_HOME=/path/to/jdk-17                 # macOS brew: /opt/homebrew/opt/openjdk@17
export ANDROID_HOME=$HOME/Android/sdk
```

```powershell
$env:JAVA_HOME  = "C:\Program Files\Eclipse Adoptium\jdk-17"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
```

If Gradle cannot find the SDK it will say so; the alternative to `ANDROID_HOME`
is a one-line `android/local.properties` containing
`sdk.dir=C\:\\Users\\you\\AppData\\Local\\Android\\Sdk` (git-ignored).

## 2.3 Build and install the app

```bash
cd android
./gradlew assembleDebug          # build the APK  (CI runs exactly this)
./gradlew installDebug           # build + install onto the connected phone
```

```powershell
cd android
.\gradlew.bat assembleDebug
.\gradlew.bat installDebug
```

The APK lands at `android/app/build/outputs/apk/debug/app-debug.apk`. To keep
or share a specific build:

```powershell
New-Item -ItemType Directory -Force ..\dist | Out-Null
Copy-Item app\build\outputs\apk\debug\app-debug.apk ..\dist\argus-edge-debug.apk
```

**Installing a prebuilt APK instead of building** — the only step that needs no
JDK or SDK, just platform-tools:

```bash
adb install -r dist/argus-edge-debug.apk
```

or copy the file to the phone and open it, with *install from unknown sources*
allowed. You still have to stage the models (§2.4).

Run the host-side unit tests while you are here — decode, protocol encoding,
form-classifier arithmetic, rep counting, discovery parsing:

```bash
./gradlew testDebugUnitTest
```

## 2.4 Stage the perception models — **required**

**The models are not in this repository.** Each carries its own licence, one of
them is AGPL-3.0, and this repository is public and MIT — committing them would
mean redistributing under terms it cannot offer. What is committed is the
recipe, and it is a complete one: the exports are byte-identical across runs at
the versions pinned in [`android/models.json`](android/models.json), which
`fetch_edge_models.py` verifies by sha256.

Without a staged model the app installs, launches, and reports **"No model
staged"** — it does not pretend to work.

### Export them (on the laptop)

```bash
pip install -r requirements-models.txt      # torch, qai-hub-models, onnx, ultralytics
python scripts/fetch_edge_models.py --out models/edge
```

This writes:

| Artifact | Role | Licence |
|---|---|---|
| `yolo26_pose_fp32.onnx` | **The path the station runs** — single-stage detector + COCO-17 pose | **AGPL-3.0** (Ultralytics) |
| `pose_landmark_fp32.onnx` | BlazePose landmarks, fallback path only | Apache-2.0 |

> **Read [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) before distributing
> anything built on the pose model.** Demoing and developing against AGPL-3.0
> weights triggers nothing; shipping an application built on them is a
> licensing decision, and the permissively licensed alternatives are already
> scouted there.

No Qualcomm AI Hub account is required. The `--aihub-job` flag fetches an
optional device-optimised variant and only works for the account that owns the
job; the local export is the primary path.

The two-model fallback detector (`yolox`, Apache-2.0) is optional and is
documented rather than automated, because its pip extra does not install
cleanly. `fetch_edge_models.py` prints the exact export command; afterwards
generate its required contract sidecar with
`python scripts/gen_yolox_fixture.py models/edge/yolox.onnx`.

### Push them onto the phone

Three ways in, same destination — the app's private `files/models/`:

```bash
# POSIX — one command: build, install, stage, grant camera, adb reverse, launch
cd android && ./stage.sh --models ../models/edge
```

```powershell
# Windows PowerShell — the same steps directly
$adb    = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$pkg    = "com.argus.edge"
$models = "..\models\edge"

foreach ($f in @("yolo26_pose_fp32.onnx","pose_landmark_fp32.onnx")) {
    & $adb push "$models\$f" /data/local/tmp/
}
& $adb shell "run-as $pkg mkdir -p files/models"
foreach ($f in @("yolo26_pose_fp32.onnx","pose_landmark_fp32.onnx")) {
    & $adb shell "run-as $pkg cp /data/local/tmp/$f files/models/"
}
```

**Or with no cable at all:** copy the files to the phone, open the app, tap
**Debug → Model**, and multi-select them in the system picker. Nothing is ever
fetched from a network by the app.

Verify:

```bash
adb shell "run-as com.argus.edge ls files/models/"
```

> **`./gradlew connectedDebugAndroidTest` uninstalls the app when it finishes**
> — both APKs, which also destroys `files/models/`. On the phone the symptom is
> simply that nothing happens. Always re-run `./stage.sh` afterwards; that is
> what the script is for.

## 2.5 Grant the camera permission

Tap *Allow* on first launch, or pre-grant it:

```bash
adb shell pm grant com.argus.edge android.permission.CAMERA
```

## 2.6 Verify the phone install

Open **Argus Edge**. You should see the dashboard — Fitness, Nursing, Lab,
Welding. Tap **Fitness**, then **Start**: the status chip names the active
backend (`yolo26-pose (single-stage)`), not "No model staged", and boxes appear
around people in frame.

The device-side instrumented tests, which prove the graph really executes on
the Hexagon with CPU fallback disabled, need a connected phone and a staged
model. CI cannot run them and deliberately does not skip them into a green tick
that would mean nothing:

```bash
cd android
./gradlew connectedDebugAndroidTest    # then re-run ./stage.sh — see the warning above
```

---

# Part 3 — Connecting the phone to the laptop

## 3.1 USB (recommended for development)

No Wi-Fi, and no ambiguity about which network the phone is on:

```bash
adb reverse tcp:8765 tcp:8765
```

Then, in the app's **Server** dialog, type `ws://localhost:8765`. The tunnel
does not survive an `adb` reconnect or a reboot — re-run it if the app reports
it cannot connect. (`stage.sh` sets it up for you.)

## 3.2 Wi-Fi (how a real floor runs)

Put the phone and the laptop on the same network, then either:

* press **Find server on this network** in the connect dialog, pick the session
  by the instructor's name, and let the address fill itself in — a human still
  presses Connect, because a beacon is an unauthenticated datagram and gets to
  make a suggestion, not a decision; or
* type `ws://<laptop-ip>:8765` by hand. `argus doctor` prints exactly which
  address a phone should use.

Name the session first so it is identifiable in the picker, in
`configs/argus.default.toml`:

```toml
[session]
name = "Coach Riley — 6pm class"
use_case = "fitness"
```

Both halves must agree on `protocol_version` (currently 1) and on `use_case`; a
mismatch is refused at handshake with a stated reason rather than half-working.
See [docs/PROTOCOL.md](docs/PROTOCOL.md).

---

# Appendix A — Complete dependency inventory

Nothing below is installed implicitly by something else in a way that is not
listed here.

### Laptop

| Dependency | Version | Installed by | Purpose |
|---|---|---|---|
| Python | ≥ 3.11 | you, §1.1 | runtime |
| `websockets` | ≥ 13 | `requirements.txt` | **the only runtime dependency** — the ingest server |
| `pytest`, `pytest-timeout` | ≥ 7.0, ≥ 2.1 | `requirements.txt` | test suite |
| `argus` | 0.1.0 (this repo) | `pip install -e .` | the package and the `argus` command |
| `pyinstaller` | ≥ 6.0 | `pip install -e ".[build]"` | optional — the standalone binary |
| `scikit-learn`, `numpy` | ≥ 1.3, ≥ 1.24 | `requirements-train.txt` | optional — retraining form classifiers |
| `torch`, `qai-hub-models`, `qai-hub`, `onnx`, `ultralytics` | pinned in `android/models.json` | `requirements-models.txt` | optional — exporting the phone's models |

The triage scorer itself imports **only the standard library**; CI asserts that
importing it pulls in no transport library and no model runtime.

### Phone (all fetched automatically by Gradle from Maven Central)

| Dependency | Version | Purpose |
|---|---|---|
| AGP / Kotlin | 8.9.2 / 2.0.21 | build |
| Gradle | 8.13 | downloaded by `./gradlew` |
| AndroidX core-ktx, appcompat, activity-ktx, lifecycle | see `build.gradle.kts` | app scaffolding |
| CameraX | 1.4.1 | camera frames |
| `onnxruntime-android-qnn` | 1.28.0 | inference + the QNN execution provider (brings `qnn-runtime`) |
| OkHttp | 4.12.0 | the WebSocket client |
| JUnit / AndroidX test | 4.13.2 / 1.2.1 | tests |

### Model artifacts (exported, never committed)

| Artifact | Licence | Required? |
|---|---|---|
| `yolo26_pose_fp32.onnx` | AGPL-3.0 | yes — this is the path the station runs |
| `pose_landmark_fp32.onnx` | Apache-2.0 | no — fallback path |
| `yolox.onnx` + `.data` + generated `.json` sidecar | Apache-2.0 | no — fallback detector |

---

# Appendix B — Install troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python: command not found` | 3.11 not on PATH | use `python3`, or reinstall with "Add to PATH" ticked |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use install Path B |
| `argus: command not found` after installing | virtualenv not activated | activate it, or use `.venv\Scripts\python.exe -m argus.cli …` |
| `[CONFIG ERROR] unknown key …` | edited config has a typo | unknown keys are rejected on purpose — an operator who thinks they retuned the system and did not is the failure that matters |
| `argus doctor` says the ingest port is not bindable | something else is on 8765 | `argus run --ws-port 8766`, and point the phone at the same port |
| `SDK location not found` from Gradle | `ANDROID_HOME` unset | export it, or write `android/local.properties` (§2.2) |
| Gradle fails with an "Unsupported class file major version" | wrong JDK | it must be **17**; check `java -version` and `JAVA_HOME` |
| `adb devices` shows `unauthorized` | RSA prompt not accepted | unlock the phone and accept it; `adb kill-server && adb devices` to re-prompt |
| `adb devices` shows nothing (Windows) | missing OEM USB driver | install it, and set USB mode to File transfer |
| App shows **"No model staged"** | §2.4 not done, or a device test wiped it | re-run `./stage.sh`, or import via **Debug → Model** |
| App installs but reports the NPU is unavailable | not a Qualcomm Hexagon device | there is no CPU fallback by design; this app needs the hardware in §2 |
| `run-as: package not debuggable` | a release build is installed | stage onto the debug build (`installDebug`) |
| Phone cannot reach the laptop over Wi-Fi | firewall, or client isolation on the network | open TCP 8765 (§1.7), or use the USB path (§3.1) |
| **Find server** finds nothing | the network drops broadcast | type `ws://<laptop-ip>:8765` by hand; `argus doctor` prints the address |
| Phone connects then is immediately refused | `protocol_version` or `use_case` mismatch | rebuild the app from the same commit, or run the laptop with `--use-case <what the phone sends>` |

Anything not covered here is likely a *usage* question rather than an install
one — see [USAGE.md](USAGE.md).
