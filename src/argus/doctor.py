"""`argus doctor` — say exactly what is missing and exactly how to fix it.

Every check reports PASS / WARN / FAIL with a one-line remedy. A FAIL means the
configured engine cannot run; a WARN means it can run but something about the
result will be less than it appears (CPU instead of NPU, a placeholder-
calibrated model, an unprovisioned artifact tree).
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from argus.config import ArgusConfig
from argus.engines import reference_specs
from argus.engines.metadata import MetadataError, load_model_specs
from argus.engines.qnn_context import find_qairt_sdk, sdk_version

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def _check_host() -> list[Check]:
    machine = platform.machine()
    native_arm = machine.lower() in ("arm64", "aarch64")
    return [
        Check(
            "host architecture",
            PASS if native_arm else WARN,
            f"{machine} on {platform.system()} {platform.release()}",
            ""
            if native_arm
            else "This interpreter is emulated x86-64. It runs the core app and "
            "onnx-cpu fine, but cannot load the win-arm64 NPU wheels. Use "
            "`run.ps1 -Npu` and .venv-npu for engine.kind = 'qnn-npu'.",
        ),
        Check(
            "python",
            PASS if sys.version_info >= (3, 11) else FAIL,
            platform.python_version(),
            "" if sys.version_info >= (3, 11) else "Argus requires Python 3.11+.",
        ),
    ]


def _check_packages() -> list[Check]:
    checks = []
    for module, needed_for in (
        ("cv2", "capture, tracking appearance, overlay"),
        ("numpy", "everything"),
        ("onnx", "EPContext wrapper generation"),
        ("onnxruntime", "engine.kind = 'onnx-cpu'"),
    ):
        present = importlib.util.find_spec(module) is not None
        checks.append(
            Check(
                f"package {module}",
                PASS if present else FAIL,
                "installed" if present else "missing",
                "" if present else f"Required for {needed_for}. Run .\\run.ps1",
            )
        )
    qnn = importlib.util.find_spec("onnxruntime_qnn") is not None
    checks.append(
        Check(
            "package onnxruntime-qnn",
            PASS if qnn else WARN,
            "installed" if qnn else "missing",
            ""
            if qnn
            else "Only needed for engine.kind = 'qnn-npu'. Install with "
            "`run.ps1 -Npu` into the native-ARM64 .venv-npu.",
        )
    )
    return checks


def _check_artifacts(cfg: ArgusConfig) -> list[Check]:
    checks: list[Check] = []
    for role, reference in reference_specs.BY_ROLE.items():
        try:
            path = Path(cfg.models.path(role))
        except Exception as exc:
            checks.append(Check(f"artifact {role}", FAIL, str(exc), "Fix [models] in the config."))
            continue

        if not path.is_file():
            checks.append(
                Check(
                    f"artifact {role}",
                    WARN,
                    f"missing: {path}",
                    "Run `argus bootstrap`. models/ is gitignored, so a fresh "
                    "clone has none. Not needed for engine.kind = 'mock'.",
                )
            )
            continue

        size_mb = path.stat().st_size / 1e6
        checks.append(Check(f"artifact {role}", PASS, f"{path.name} ({size_mb:.1f} MB)"))

        metadata_key = reference_specs.METADATA_KEY_BY_ROLE[role]
        try:
            specs = load_model_specs(cfg.models.path(metadata_key))
        except (MetadataError, Exception) as exc:
            checks.append(
                Check(f"contract {role}", WARN, f"metadata unreadable: {exc}",
                      "Re-run `argus bootstrap` to restore metadata.json.")
            )
            continue

        actual = specs.get(reference.file_name)
        if actual is None:
            checks.append(
                Check(f"contract {role}", FAIL,
                      f"metadata.json has no entry for {reference.file_name}",
                      "The artifact and its manifest disagree; re-provision.")
            )
        elif (actual.inputs, actual.outputs) != (reference.inputs, reference.outputs):
            checks.append(
                Check(f"contract {role}", FAIL,
                      "metadata.json tensor contract differs from the built-in reference",
                      "The artifact was re-exported with a different contract. Update "
                      "src/argus/engines/reference_specs.py and re-run the tests.")
            )
        else:
            checks.append(Check(f"contract {role}", PASS, "matches reference_specs"))
    return checks


def _check_qairt(cfg: ArgusConfig) -> list[Check]:
    sdk = find_qairt_sdk(cfg.engine.qairt_sdk_root)
    if sdk is None:
        return [
            Check(
                "QAIRT SDK",
                WARN,
                "not found",
                "Needed only for engine.context_binary_mode = 'netrun' and for the "
                "C++ runner. Set QNN_SDK_ROOT or engine.qairt_sdk_root.",
            )
        ]

    local = sdk_version(sdk) or "unknown"
    checks = [Check("QAIRT SDK", PASS, f"{local} at {sdk}")]

    artifact_versions = {
        spec.qairt_version for spec in reference_specs.BY_ROLE.values() if spec.qairt_version
    }
    skewed = [v for v in artifact_versions if v.split(".")[:2] != local.split(".")[:2]]
    if skewed:
        checks.append(
            Check(
                "QAIRT version skew",
                WARN,
                f"artifacts built with {sorted(skewed)}, local SDK is {local}",
                "A QNN context binary is tied to the runtime that produced it. "
                "Install the matching QAIRT, or recompile the artifacts against "
                "the installed one, before trusting the pose stage on hardware.",
            )
        )
    else:
        checks.append(Check("QAIRT version skew", PASS, f"artifacts and SDK agree on {local}"))
    return checks


def _check_engine(cfg: ArgusConfig) -> list[Check]:
    if cfg.engine.kind == "mock":
        return [
            Check(
                "engine",
                WARN,
                "engine.kind = 'mock'",
                "Mock synthesises tensors at the real contract: good for pipeline "
                "work, meaningless for accuracy. Switch to onnx-cpu or qnn-npu for "
                "real inference.",
            )
        ]
    if cfg.engine.kind == "onnx-cpu":
        return [
            Check(
                "engine",
                WARN,
                "engine.kind = 'onnx-cpu'",
                "The CPU reference path. Pose is unavailable on it — the BlazePose "
                "artifacts are QNN context binaries with no CPU path.",
            )
        ]
    checks = []
    if cfg.engine.allow_cpu_fallback:
        checks.append(
            Check(
                "engine fallback",
                WARN,
                "engine.allow_cpu_fallback = true",
                "Results may come from the CPU. Any latency figure from such a run "
                "is not an NPU figure. Set it back to false outside A/B measurement.",
            )
        )
    try:
        from argus.engines import ort_qnn

        provider = ort_qnn.register_qnn_ep()
        checks.append(
            Check(
                "QNN execution provider",
                PASS if provider else FAIL,
                provider or "not available",
                ""
                if provider
                else "Install onnxruntime-qnn into a native ARM64 venv "
                "(`run.ps1 -Npu`). Note it is a plugin EP: it does not appear in "
                "get_available_providers() until registered.",
            )
        )
    except Exception as exc:
        checks.append(Check("QNN execution provider", FAIL, str(exc),
                            "See the quad-npu-prereqs skill."))
    return checks


def run_doctor(cfg: ArgusConfig) -> int:
    """Print every check. Returns 1 if anything FAILed, else 0."""
    checks: list[Check] = []
    checks += _check_host()
    checks += _check_packages()
    checks += _check_artifacts(cfg)
    checks += _check_qairt(cfg)
    checks += _check_engine(cfg)

    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{check.status}] {check.name.ljust(width)}  {check.detail}")
        if check.remedy:
            print(f"        -> {check.remedy}")

    failures = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)
    print(f"\n{len(checks)} checks: {failures} failed, {warnings} warned")
    return 1 if failures else 0
