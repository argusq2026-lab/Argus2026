"""Reproduce the Android station's model artifacts from a fresh clone.

The models are not in this repository. Each carries its own licence — one of
them is AGPL-3.0 — and this repository is public and MIT, so committing them
would mean redistributing terms it is not entitled to offer. What is committed
instead is the recipe, which is enough: the exports below are **byte-identical**
across runs given the pinned versions in `android/models.json`, verified by
sha256 here.

    python scripts/fetch_edge_models.py --out models/edge
    cd android && ./stage.sh --models ../models/edge

No Qualcomm AI Hub account is needed for any of this. An account only buys the
optional device-optimised variant (see `--aihub-job`), and AI Hub job IDs are
scoped to the account that created them, so a recorded job ID is provenance for
*us* rather than something a stranger can fetch — hence the local export being
the primary path rather than a fallback.

`yolox` is the exception and is documented rather than automated: its artifact
came from an AI Hub export run whose upstream `qai_hub_models` entry needs the
`yolox` pip extra, which does not install cleanly (its build requires torch to
already be present). The command is printed so it can be run deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "android" / "models.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_versions(manifest: dict) -> list[str]:
    """Pinned versions are what make the exports byte-identical; warn on drift."""
    from importlib.metadata import PackageNotFoundError, version

    problems = []
    for package, expected in manifest.get("pinned_versions", {}).items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed (expected {expected})")
            continue
        if actual != expected:
            problems.append(f"{package} is {actual}, manifest pins {expected}")
    return problems


def export_yolo26_pose(out_dir: Path) -> Path:
    import torch
    from qai_hub_models.models.yolo26_pose.model import Yolo26PoseDetector

    target = out_dir / "yolo26_pose_fp32.onnx"
    model = Yolo26PoseDetector.from_pretrained().eval()
    torch.onnx.export(
        model, torch.zeros(1, 3, 640, 640), str(target),
        opset_version=17, dynamo=False,
        input_names=["image"], output_names=["boxes", "scores", "keypoints"],
    )
    return target


def export_pose_landmark(out_dir: Path) -> Path:
    import torch
    from qai_hub_models.models.mediapipe_pose.model import PoseLandmarkDetector

    target = out_dir / "pose_landmark_fp32.onnx"
    model = PoseLandmarkDetector.from_pretrained().eval()
    torch.onnx.export(
        model, torch.zeros(1, 3, 256, 256), str(target),
        opset_version=17, dynamo=False,
        input_names=["image"], output_names=["scores", "landmarks"],
    )
    return target


def fetch_aihub_optimised(job_id: str, out_dir: Path) -> Path | None:
    """The device-optimised build, for whoever owns the job. Optional."""
    import zipfile

    import onnx
    import qai_hub as hub

    job = hub.get_job(job_id)
    if not job.get_status().success:
        print(f"    job {job_id} is not successful; skipping")
        return None
    archive = Path(str(job.get_target_model().download(str(out_dir / "_aihub.zip"))))
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir / "_aihub")
    found = sorted((out_dir / "_aihub").rglob("*.onnx"))
    if not found:
        return None

    # AI Hub renames graph outputs to output_0/1/2. The app checks tensor names
    # against a fixed contract, deliberately, so restore the canonical ones
    # rather than loosening the check -- a contract that accepts any name is not
    # a contract.
    model = onnx.load(str(found[0]))
    canonical = ["boxes", "scores", "keypoints"]
    for index, output in enumerate(model.graph.output):
        if index >= len(canonical):
            break
        old = output.name
        new = canonical[index]
        if old == new:
            continue
        output.name = new
        for node in model.graph.node:
            node.output[:] = [new if o == old else o for o in node.output]
    target = out_dir / "yolo26_pose_sm8750.onnx"
    onnx.save(model, str(target))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/edge", help="where to write artifacts")
    parser.add_argument(
        "--aihub-job", action="store_true",
        help="also fetch the device-optimised build (needs the AI Hub account that owns the job)",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    drift = check_versions(manifest)
    if drift:
        print("version drift — exports may not match the recorded hashes:")
        for problem in drift:
            print(f"  ! {problem}")
        print()

    models = manifest["models"]
    ok = True

    for name, exporter in (
        ("yolo26_pose", export_yolo26_pose),
        ("pose_landmark", export_pose_landmark),
    ):
        entry = models[name]
        print(f"==> {name}  ({entry['licence']})")
        path = exporter(out_dir)
        digest = sha256_of(path)
        expected = entry.get("sha256")
        if expected and digest != expected:
            print(f"    MISMATCH  got {digest[:16]}…  manifest says {expected[:16]}…")
            print("    the artifact differs from the one this repository was built against")
            ok = False
        else:
            print(f"    {path.name}  {path.stat().st_size // 1024} KB  sha256 {digest[:16]}… ok")

    if args.aihub_job:
        entry = models["yolo26_pose"]
        print(f"==> yolo26_pose optimised (AI Hub job {entry['aihub_job']})")
        try:
            path = fetch_aihub_optimised(entry["aihub_job"], out_dir)
            if path:
                print(f"    {path.name}  {path.stat().st_size // 1024} KB")
        except Exception as exc:  # noqa: BLE001 — provenance, not control flow
            print(f"    unavailable: {type(exc).__name__}: {str(exc)[:120]}")
            print("    AI Hub jobs are scoped to the account that created them.")

    entry = models["yolox"]
    print(f"==> yolox  ({entry['licence']}) — not automated")
    print(f"    {entry['export_command']}")
    print("    then place yolox.onnx / yolox.data in the output directory, and run")
    print("    python scripts/gen_yolox_fixture.py <out>/yolox.onnx")

    print(f"\nartifacts in {out_dir}")
    print("stage them with:  cd android && ./stage.sh --models", out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
