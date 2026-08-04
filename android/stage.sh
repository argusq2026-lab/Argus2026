#!/usr/bin/env bash
# Build, install, stage models, and launch the station in one step.
#
# This exists because `./gradlew connectedDebugAndroidTest` UNINSTALLS the app
# when it finishes — both the app and the test APK — which also destroys
# files/models/. Running the device tests therefore leaves you with no app and
# no staged model, and the symptom on the phone is simply that nothing happens.
# Re-run this after any device-test run.
#
# Usage:
#   ./stage.sh                      build, install, stage, launch
#   ./stage.sh --models DIR         take yolox.* / pose_landmark_fp32.onnx from DIR
#   ./stage.sh --no-build           skip the Gradle build
set -euo pipefail

cd "$(dirname "$0")"
: "${JAVA_HOME:=/opt/homebrew/opt/openjdk@17}"
: "${ANDROID_HOME:=$HOME/Library/Android/sdk}"
export JAVA_HOME ANDROID_HOME
ADB="$ANDROID_HOME/platform-tools/adb"
PKG=com.argus.edge

MODELS_DIR=""
BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS_DIR="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! "$ADB" get-state >/dev/null 2>&1; then
  echo "no device: check the USB cable, that USB debugging is on, and that the" >&2
  echo "phone's USB mode is File transfer rather than charge-only." >&2
  exit 1
fi

if [ "$BUILD" = 1 ]; then
  echo "==> building"
  ./gradlew assembleDebug -q
fi

echo "==> installing"
"$ADB" install -r app/build/outputs/apk/debug/app-debug.apk | tail -1

# Models are gitignored artifacts. Prefer a directory given on the command
# line, else whatever was last pushed to /data/local/tmp.
if [ -n "$MODELS_DIR" ]; then
  echo "==> pushing models from $MODELS_DIR"
  for f in yolo26_pose_fp32.onnx yolox.onnx yolox.data yolox.json pose_landmark_fp32.onnx; do
    [ -f "$MODELS_DIR/$f" ] && "$ADB" push "$MODELS_DIR/$f" /data/local/tmp/ >/dev/null
  done
fi

echo "==> staging into the app sandbox"
"$ADB" shell "run-as $PKG mkdir -p files/models" || {
  echo "run-as failed: is this a debuggable build?" >&2; exit 1; }
staged=0
for f in yolo26_pose_fp32.onnx yolox.onnx yolox.data yolox.json pose_landmark_fp32.onnx; do
  if "$ADB" shell "[ -f /data/local/tmp/$f ]"; then
    "$ADB" shell "run-as $PKG cp /data/local/tmp/$f files/models/"
    staged=$((staged + 1))
  else
    echo "    missing /data/local/tmp/$f — push it or pass --models DIR"
  fi
done
echo "    staged $staged file(s):"
"$ADB" shell "run-as $PKG ls files/models/" | sed 's/^/      /'

"$ADB" shell pm grant $PKG android.permission.CAMERA || true
# The laptop's ingest server, reachable from the phone as ws://localhost:8765.
# Harmless when no server is running; the app will just report the refusal.
"$ADB" reverse tcp:8765 tcp:"${ARGUS_WS_PORT:-8765}" >/dev/null 2>&1 || true

echo "==> launching"
"$ADB" shell am start -n $PKG/.MainActivity >/dev/null
echo "done — press Start on the phone."
