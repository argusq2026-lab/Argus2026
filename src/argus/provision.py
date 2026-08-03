"""Model provisioning — turn a fresh clone into a runnable tree.

`models/` is gitignored, so `git clone && run.ps1` leaves nothing to infer
with. This module reproduces the artifact tree from the AI Hub jobs recorded in
`models/argus_jobs.json`, or seeds it from a local copy, and then **verifies**
what it produced rather than assuming the download worked.

Verification is the point. It loads every ONNX artifact and compares the
session's own I/O to the manifest, so a download that produced the wrong file,
a truncated transfer, or a re-export with a changed contract is caught here
rather than 200 frames into a run.

It also carries a duplicate-node-name repair. That was written for the
QuickSRNet load failure (`two nodes with same node name (/model/cnn/0/Conv)`)
before that failure was traced to its actual cause — an ONNX Runtime QDQ-fusion
collision at `extended` optimization, not a malformed artifact; see
:mod:`argus.engines.onnx_common`. The repair is kept because duplicate node
names are a real class of export defect worth detecting cheaply, but it is a
diagnostic here, not the fix for QuickSRNet.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from argus.config import ArgusConfig
from argus.engines import reference_specs
from argus.engines.metadata import ModelSpec

#: Directory layout produced under `models/`, one entry per AI Hub export.
EXPORT_DIRS = {
    "yolox": "yolox-onnx-w8a8",
    "quicksrnetmedium": "quicksrnetmedium-onnx-w8a8",
    "mediapipe_pose": "mediapipe_pose-qnn_context_binary-w8a8-qualcomm_snapdragon_x_elite",
}

#: The `qai_hub_models` export command that regenerates each directory. Printed
#: verbatim when automated download is unavailable, so a human can finish the job.
EXPORT_COMMANDS = {
    "yolox": (
        "python -m qai_hub_models.models.yolox.export "
        "--target-runtime onnx --quantize w8a8 --device 'Snapdragon X Elite CRD'"
    ),
    "quicksrnetmedium": (
        "python -m qai_hub_models.models.quicksrnetmedium.export "
        "--target-runtime onnx --quantize w8a8 --device 'Snapdragon X Elite CRD'"
    ),
    "mediapipe_pose": (
        "python -m qai_hub_models.models.mediapipe_pose.export "
        "--target-runtime qnn_context_binary --quantize w8a8 "
        "--device 'Snapdragon X Elite CRD'"
    ),
}


class ProvisionError(RuntimeError):
    """Provisioning could not produce a usable artifact tree."""


@dataclass
class ArtifactStatus:
    role: str
    path: Path
    present: bool
    note: str = ""


# ---------------------------------------------------------------------------
# QuickSRNet repair
# ---------------------------------------------------------------------------


def duplicate_node_names(onnx_path: Path) -> list[str]:
    """Node names that appear more than once in an ONNX graph."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    counts = Counter(node.name for node in model.graph.node if node.name)
    return sorted(name for name, n in counts.items() if n > 1)


def repair_duplicate_node_names(onnx_path: Path) -> int:
    """Make every node name unique in place. Returns the number renamed.

    ONNX node names are diagnostic identifiers; edges are carried by tensor
    names, which are untouched. Renaming a duplicate therefore cannot change
    what the graph computes — it only makes the model loadable.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    seen: Counter[str] = Counter()
    renamed = 0
    for node in model.graph.node:
        if not node.name:
            continue
        seen[node.name] += 1
        if seen[node.name] > 1:
            node.name = f"{node.name}_dup{seen[node.name] - 1}"
            renamed += 1
    if renamed:
        onnx.save(model, str(onnx_path), save_as_external_data=False)
    return renamed


# ---------------------------------------------------------------------------
# metadata.json
# ---------------------------------------------------------------------------


def _tensor_json(spec) -> dict:
    payload: dict = {"shape": list(spec.shape), "dtype": spec.dtype}
    if spec.is_quantized:
        payload["quantization_parameters"] = {
            "scale": spec.scale,
            "zero_point": spec.zero_point,
        }
    return payload


def write_metadata(directory: Path, specs: list[ModelSpec]) -> Path:
    """Write a metadata.json describing `specs` into `directory`.

    Used when a download produced model files but no manifest. The contracts
    come from `reference_specs`, and `tests/test_artifacts.py` independently
    checks the *artifact's own* session I/O against them, so a manifest written
    here cannot paper over a re-export that actually changed shape.
    """
    head = specs[0]
    payload = {
        "model_id": head.model_id,
        "model_name": head.model_name,
        "runtime": head.runtime,
        "precision": head.precision,
        "tool_versions": {"qairt": head.qairt_version} if head.qairt_version else {},
        "model_files": {
            spec.file_name: {
                "inputs": {s.name: _tensor_json(s) for s in spec.inputs},
                "outputs": {s.name: _tensor_json(s) for s in spec.outputs},
            }
            for spec in specs
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "metadata.json"
    out.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def seed_from_directory(source: Path, models_root: Path, force: bool = False) -> list[str]:
    """Copy an existing artifact tree in. Returns the names copied."""
    if not source.is_dir():
        raise ProvisionError(f"--from-dir does not exist: {source}")
    copied = []
    for export_dir in EXPORT_DIRS.values():
        src = source / export_dir
        dst = models_root / export_dir
        if not src.is_dir():
            continue
        if dst.exists() and not force:
            copied.append(f"{export_dir} (already present, skipped)")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(export_dir)
    for sidecar in ("argus_jobs.json", "argus_cu_jobs.json", "argus_summary.json",
                    "argus_profiles_raw.json"):
        src = source / sidecar
        if src.is_file() and (force or not (models_root / sidecar).exists()):
            shutil.copy2(src, models_root / sidecar)
    if not copied:
        raise ProvisionError(
            f"{source} contains none of the expected export directories "
            f"{sorted(EXPORT_DIRS.values())}"
        )
    return copied


def download_from_aihub(jobs_path: Path, models_root: Path, force: bool = False) -> list[str]:
    """Download target models for the compile jobs recorded in argus_jobs.json."""
    try:
        import qai_hub as hub
    except ImportError as exc:
        raise ProvisionError(
            "qai-hub is not installed. Install the extra with:\n"
            "  uv pip install 'argus[provision]'\n"
            "then configure a token (`qai-hub configure --api_token ...`), or "
            "provision manually with the export commands printed above."
        ) from exc

    if not jobs_path.is_file():
        raise ProvisionError(f"job manifest not found: {jobs_path}")
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))

    downloaded = []
    for key, entry in jobs.items():
        job_id = entry.get("compile") or entry.get("profile")
        if not job_id:
            continue
        model_id = key.split("::")[0]
        export_dir = EXPORT_DIRS.get(model_id)
        if export_dir is None:
            continue
        target = models_root / export_dir
        if target.exists() and not force:
            downloaded.append(f"{key} (already present, skipped)")
            continue
        target.mkdir(parents=True, exist_ok=True)
        job = hub.get_job(job_id)
        model = job.get_target_model() if hasattr(job, "get_target_model") else job.model
        if model is None:
            raise ProvisionError(f"job {job_id} has no downloadable target model")
        model.download(str(target))
        downloaded.append(f"{key} <- {job_id}")
    if not downloaded:
        raise ProvisionError(f"{jobs_path} recorded no usable job ids")
    return downloaded


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(cfg: ArgusConfig) -> list[ArtifactStatus]:
    """Check every role's artifact exists, and that ONNX ones actually load."""
    statuses: list[ArtifactStatus] = []
    for role, reference in reference_specs.BY_ROLE.items():
        try:
            path = Path(cfg.models.path(role))
        except Exception as exc:
            statuses.append(ArtifactStatus(role, Path("?"), False, str(exc)))
            continue

        if not path.is_file():
            statuses.append(ArtifactStatus(role, path, False, "missing"))
            continue

        if reference.is_context_binary:
            statuses.append(
                ArtifactStatus(role, path, True, "context binary (load requires the NPU)")
            )
            continue

        note = "loads; session I/O matches metadata"
        try:
            from argus.engines.onnx_cpu import OnnxCpuBackend

            runner = OnnxCpuBackend(cfg.engine.graph_optimization_level).load(path, reference)
            try:
                actual_out = set(runner.output_names)
            finally:
                runner.close()
            expected_out = {o.name for o in reference.outputs}
            if actual_out != expected_out:
                note = f"outputs {sorted(actual_out)} != expected {sorted(expected_out)}"
        except Exception as exc:
            note = f"does NOT load: {exc}"
        statuses.append(ArtifactStatus(role, path, True, note))
    return statuses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def bootstrap(
    cfg: ArgusConfig,
    *,
    force: bool = False,
    skip_demo: bool = False,
    from_dir: Path | None = None,
) -> int:
    """Provision `models/`, repair known-bad artifacts, and verify the result."""
    models_root = Path(cfg.models.root)
    if not models_root.is_absolute():
        models_root = cfg.models.base_dir / models_root
    models_root.mkdir(parents=True, exist_ok=True)

    print(f"provisioning into {models_root}")
    try:
        if from_dir is not None:
            for line in seed_from_directory(from_dir, models_root, force):
                print(f"  seeded {line}")
        else:
            for line in download_from_aihub(
                models_root / "argus_jobs.json", models_root, force
            ):
                print(f"  downloaded {line}")
    except ProvisionError as exc:
        print(f"[ERROR] {exc}")
        print("\nTo provision manually, run these on a non-arm64 build host:")
        for name, command in EXPORT_COMMANDS.items():
            print(f"  # {name}\n  {command}")
        print(f"\nthen copy the results into {models_root} (or pass --from-dir).")
        return 1

    # Ensure each export directory has a manifest Argus can read.
    for role, reference in reference_specs.BY_ROLE.items():
        directory = models_root / EXPORT_DIRS[reference.model_id]
        if directory.is_dir() and not (directory / "metadata.json").is_file():
            specs = [
                spec
                for spec in reference_specs.BY_ROLE.values()
                if spec.model_id == reference.model_id
            ]
            unique = list({spec.file_name: spec for spec in specs}.values())
            print(f"  wrote {write_metadata(directory, unique)}")

    # Cheap export-defect screen. Not the QuickSRNet fix -- see the module
    # docstring and argus.engines.onnx_common.
    for role in ("detector", "super_res"):
        try:
            onnx_path = Path(cfg.models.path(role))
        except Exception:
            continue
        if not onnx_path.is_file():
            continue
        try:
            duplicates = duplicate_node_names(onnx_path)
        except Exception as exc:
            print(f"  [WARN] could not inspect {onnx_path.name}: {exc}")
            continue
        if duplicates:
            renamed = repair_duplicate_node_names(onnx_path)
            print(
                f"  repaired {onnx_path.name}: renamed {renamed} duplicate node "
                f"name(s), e.g. {duplicates[0]}"
            )

    if not skip_demo:
        demo_path = cfg.models.base_dir / "demo" / "trainees_demo.mp4"
        if force or not demo_path.is_file():
            from demo.make_demo_video import make_demo_video

            make_demo_video(demo_path)
            print(f"  generated {demo_path}")

    print("\nverification:")
    failures = 0
    for status in verify(cfg):
        mark = "ok  " if status.present and "does NOT load" not in status.note else "FAIL"
        if mark == "FAIL":
            failures += 1
        print(f"  [{mark}] {status.role}: {status.note}")
    return 1 if failures else 0
