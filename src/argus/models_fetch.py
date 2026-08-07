"""`argus fetch-models` — reproduce the phone's model weights on this machine.

The weights are not in the repository, not in the APK, and not in the GitHub
release, and that is a licensing decision rather than an oversight: one of
them (yolo26-pose) is AGPL-3.0, and this project is public and MIT, so
*distributing* the weights would mean offering terms the repo is not entitled
to offer. What is distributed instead is the recipe — `android/models.json`
pins the toolchain versions that make the export byte-identical, and records
the sha256 the result must match.

For a checkout, `scripts/fetch_edge_models.py` runs that recipe. This module
is the same recipe for someone who has only the released binary: it finds a
Python on the machine, builds a private venv with the pinned toolchain, runs
the export there, and verifies the hashes. The binary itself stays 9 MB —
torch is multiple GB and is downloaded once, by the user's machine, on the
user's decision, which is exactly the line the licence note draws: Argus
automates *reproduction*, it does not *redistribute*.

Nothing here runs implicitly. The subcommand states what it is about to
download and how big it is, and `--print-only` emits the manual recipe for
whoever would rather read it first.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: The manifest travels with the code the same way the default config does:
#: bundled into `argus/_data/` by the wheel and the PyInstaller build, read
#: from the repo when running from a checkout.
_PACKAGED_MANIFEST = Path(__file__).resolve().parent / "_data" / "models.json"
_REPO_MANIFEST = Path(__file__).resolve().parents[2] / "android" / "models.json"

#: What the venv needs for the automated exports. Read from the manifest's
#: pinned_versions at runtime — stated here only as the fallback order.
_PINNED_ORDER = ("torch", "onnx", "ultralytics", "qai-hub-models", "qai-hub")

#: The export itself, run inside the venv's Python — not this process, which
#: in the frozen binary has no torch and never will. Kept as one small script
#: whose only inputs are argv, so what runs is inspectable in full.
_EXPORT_CHILD = r'''
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

def export(module_path: str, class_name: str, shape, names, target: Path) -> None:
    module = __import__(module_path, fromlist=[class_name])
    model = getattr(module, class_name).from_pretrained().eval()
    torch.onnx.export(
        model, torch.zeros(*shape), str(target),
        opset_version=17, dynamo=False,
        input_names=["image"], output_names=list(names),
    )
    print(f"exported {target.name}")

export("qai_hub_models.models.yolo26_pose.model", "Yolo26PoseDetector",
       (1, 3, 640, 640), ["boxes", "scores", "keypoints"],
       out_dir / "yolo26_pose_fp32.onnx")
export("qai_hub_models.models.mediapipe_pose.model", "PoseLandmarkDetector",
       (1, 3, 256, 256), ["scores", "landmarks"],
       out_dir / "pose_landmark_fp32.onnx")
'''


def load_manifest() -> dict:
    for candidate in (_PACKAGED_MANIFEST, _REPO_MANIFEST):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "models.json not found; expected it bundled beside the package or in "
        "a repo checkout at android/models.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _automated_models(manifest: dict) -> dict[str, dict]:
    return {
        name: spec
        for name, spec in manifest.get("models", {}).items()
        if spec.get("automated")
    }


def verify(out_dir: Path, manifest: dict) -> tuple[list[str], list[str]]:
    """(present-and-matching files, problems). Hash mismatches are problems:
    a wrong model that loads is worse than a missing one that fails loudly."""
    good: list[str] = []
    problems: list[str] = []
    for name, spec in _automated_models(manifest).items():
        expected = spec.get("sha256")
        for filename in spec.get("files", []):
            path = out_dir / filename
            if not path.is_file():
                problems.append(f"{filename}: missing")
            elif expected and _sha256(path) != expected:
                problems.append(
                    f"{filename}: sha256 mismatch — not the export the pinned "
                    "toolchain produces; delete it and re-run"
                )
            else:
                good.append(filename)
    return good, problems


def print_recipe(manifest: dict) -> None:
    """The manual path, for whoever wants to read before running."""
    pins = manifest.get("pinned_versions", {})
    pin_args = " ".join(
        f"{pkg}=={pins[pkg]}" for pkg in _PINNED_ORDER if pkg in pins
    )
    print("Manual recipe (what `argus fetch-models` automates):")
    print(f"  python3 -m venv argus-model-toolchain")
    print(f"  argus-model-toolchain/bin/pip install {pin_args}")
    print("  # then run the export — from a checkout:")
    print("  python scripts/fetch_edge_models.py --out models/edge")
    print()
    print("Licences (why none of this ships pre-built):")
    for name, spec in manifest.get("models", {}).items():
        print(f"  {name}: {spec.get('licence', 'unknown')}")


def find_python() -> str | None:
    """A real Python on this machine — the frozen binary is not one."""
    for candidate in ("python3", "python"):
        found = shutil.which(candidate)
        if not found:
            continue
        try:
            probe = subprocess.run(
                [found, "-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"],
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return found
    return None


def fetch(out_dir: Path, manifest: dict, assume_yes: bool = False) -> int:
    """Build the pinned toolchain in a private venv and run the export."""
    good, problems = verify(out_dir, manifest)
    if good and not problems:
        print(f"all model files already present and verified in {out_dir}")
        _print_phone_instructions(out_dir, good)
        return 0

    python = find_python()
    if python is None:
        print(
            "No Python 3.10+ found on this machine. The export toolchain "
            "(torch) cannot be bundled into this binary, so a Python install "
            "is the one prerequisite. Install one, or run the recipe on any "
            "machine that has one:",
            file=sys.stderr,
        )
        print_recipe(manifest)
        return 1

    pins = manifest.get("pinned_versions", {})
    pin_args = [f"{pkg}=={pins[pkg]}" for pkg in _PINNED_ORDER if pkg in pins]
    print("This will download the pinned export toolchain into a private venv:")
    print(f"  {' '.join(pin_args)}")
    print("torch alone is on the order of gigabytes; the weights themselves are")
    print("fetched by the upstream packages from their own sources.")
    if not assume_yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted; --print-only shows the manual recipe")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = out_dir / ".toolchain"
    venv_python = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python"
    )
    if not venv_python.exists():
        print(f"creating venv at {venv_dir} …")
        subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)
    print("installing the pinned toolchain (this is the long step) …")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--quiet", *pin_args], check=True
    )

    print("running the export …")
    subprocess.run([str(venv_python), "-c", _EXPORT_CHILD, str(out_dir)], check=True)

    good, problems = verify(out_dir, manifest)
    if problems:
        for problem in problems:
            print(f"  FAILED {problem}", file=sys.stderr)
        return 1
    print(f"\nverified against android/models.json: {', '.join(good)}")
    _print_phone_instructions(out_dir, good)
    print(
        "\nNote: the yolox detector artifact is documented rather than "
        "automated (see android/models.json) — the single-stage yolo26 model "
        "above is the one the app prefers and is sufficient on its own."
    )
    return 0


def _print_phone_instructions(out_dir: Path, files: list[str]) -> None:
    print("\nTo put them on a phone:")
    print(f"  1. Copy {', '.join(files)} from {out_dir} to the phone's Downloads")
    print("     (cable, cloud drive, or `adb push <file> /sdcard/Download/`).")
    print("  2. In the Argus Edge station screen, long-press Debug and")
    print("     multi-select the files in the picker. The status strip reads")
    print("     Ready when the model is loaded.")
