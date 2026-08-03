"""Running a QNN context binary.

`onnxruntime-qnn` cannot open a `.bin` directly: a context binary is compiled
HTP code plus a serialised graph, not an ONNX model. The two BlazePose stages
ship only in that form, so Argus needs an explicit answer for them.

Two are provided, selected by ``engine.context_binary_mode``:

``epcontext`` (default, in-process)
    Generate a tiny ONNX model whose single ``EPContext`` node points at the
    ``.bin``, then hand that to ONNX Runtime with the QNN EP registered. This
    is ORT's own mechanism for pre-compiled contexts; the wrapper carries the
    I/O names, shapes and dtypes straight from ``metadata.json``, which is
    exactly the information a raw ``.bin`` does not self-describe to ORT.

``netrun`` (out-of-process, diagnostic)
    Shell out to the QAIRT SDK's ``qnn-net-run.exe``. One subprocess and a
    round-trip through raw files per inference, so it is far too slow for the
    per-frame path — it exists to prove a binary loads and produces the
    declared shapes on hardware, independently of ORT.

**Version skew is checked up front.** The shipped artifacts were compiled with
QAIRT 2.45.0; a locally-installed 2.32.6 will reject them with an opaque
backend error. :func:`check_qairt_compatibility` turns that into a sentence
naming both versions.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from argus.engines.base import EngineError
from argus.engines.metadata import ModelSpec, TensorSpec

#: Where a default QAIRT install puts its versioned SDK directories.
_DEFAULT_QAIRT_GLOB = Path("C:/Qualcomm/AIStack/QAIRT/qairt")

#: ONNX Runtime's op for a pre-compiled EP context blob.
_EPCONTEXT_DOMAIN = "com.microsoft"
_EPCONTEXT_OP = "EPContext"


class ContextBinaryError(EngineError):
    """Failure specific to loading or executing a `.bin`."""


def find_qairt_sdk(configured: str = "") -> Path | None:
    """Locate the QAIRT SDK root: config, then env, then the default install."""
    if configured:
        p = Path(configured)
        return p if p.is_dir() else None
    for var in ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT", "SNPE_ROOT"):
        value = os.environ.get(var)
        if value and Path(value).is_dir():
            return Path(value)
    if _DEFAULT_QAIRT_GLOB.is_dir():
        versions = sorted(
            (d for d in _DEFAULT_QAIRT_GLOB.iterdir() if d.is_dir()),
            key=lambda d: _version_key(d.name),
        )
        if versions:
            return versions[-1]
    return None


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", name)[:4]) or (0,)


def sdk_version(sdk_root: Path) -> str | None:
    """The SDK's version, taken from its directory name."""
    match = re.search(r"\d+(?:\.\d+)+", sdk_root.name)
    return match.group(0) if match else None


def check_qairt_compatibility(spec: ModelSpec, sdk_root: Path | None) -> str | None:
    """Return a human-readable warning when artifact and SDK majors differ.

    Returns None when they agree, or when either version is unknown.
    """
    if sdk_root is None or not spec.qairt_version:
        return None
    local = sdk_version(sdk_root)
    if not local:
        return None
    artifact_major = _version_key(spec.qairt_version)[:2]
    local_major = _version_key(local)[:2]
    if artifact_major == local_major:
        return None
    return (
        f"{spec.file_name} was compiled with QAIRT {spec.qairt_version} but the "
        f"local SDK at {sdk_root} is {local}. A context binary is tied to the "
        "runtime that produced it; expect a backend-initialisation failure. "
        "Install the matching QAIRT, or recompile the artifact against the "
        "installed one (see scripts/fetch_models.py)."
    )


# ---------------------------------------------------------------------------
# EPContext wrapper generation
# ---------------------------------------------------------------------------

_ONNX_DTYPE = {
    "uint8": 2,    # TensorProto.UINT8
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "float32": 1,
    "float16": 10,
}


def _value_info(spec: TensorSpec):
    from onnx import helper

    elem_type = _ONNX_DTYPE.get(spec.dtype)
    if elem_type is None:
        raise ContextBinaryError(f"unsupported tensor dtype for EPContext: {spec.dtype}")
    return helper.make_tensor_value_info(spec.name, elem_type, list(spec.shape))


def build_epcontext_model(bin_path: Path, spec: ModelSpec, out_path: Path) -> Path:
    """Write an ONNX wrapper around `bin_path` and return its path.

    ``embed_mode=0`` keeps the (multi-MB) binary on disk and references it by
    name; the wrapper is a few hundred bytes and is regenerated whenever the
    binary's mtime or size changes.
    """
    try:
        from onnx import checker, helper
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ContextBinaryError(
            "the `onnx` package is required to wrap a QNN context binary"
        ) from exc

    if not bin_path.is_file():
        raise ContextBinaryError(f"context binary not found: {bin_path}")

    node = helper.make_node(
        _EPCONTEXT_OP,
        inputs=[s.name for s in spec.inputs],
        outputs=[s.name for s in spec.outputs],
        name=f"{bin_path.stem}_ctx",
        domain=_EPCONTEXT_DOMAIN,
        embed_mode=0,
        ep_cache_context=bin_path.name,
        source="QNN",
        main_context=1,
        max_size=0,
        partition_name=bin_path.stem,
        onnx_model_filename=out_path.name,
    )
    graph = helper.make_graph(
        [node],
        f"{bin_path.stem}_epcontext",
        [_value_info(s) for s in spec.inputs],
        [_value_info(s) for s in spec.outputs],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid(_EPCONTEXT_DOMAIN, 1),
        ],
        producer_name="argus",
    )
    # The EPContext op is not in the standard schema registry, so full checking
    # would reject a valid wrapper. Structural checking still catches a
    # malformed graph.
    checker.check_model(model, full_check=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(model.SerializeToString())
    return out_path


def epcontext_wrapper_path(bin_path: Path, cache_dir: Path) -> Path:
    """Cache location for a binary's wrapper.

    The wrapper must sit beside the `.bin` because ``ep_cache_context`` is
    resolved relative to the ONNX file's directory.
    """
    return bin_path.with_name(f"{bin_path.stem}_epctx.onnx")


# ---------------------------------------------------------------------------
# qnn-net-run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetRunPaths:
    executable: Path
    backend: Path


def find_net_run(sdk_root: Path) -> NetRunPaths:
    """Locate `qnn-net-run.exe` and `QnnHtp.dll` for this host architecture."""
    import platform

    arch = platform.machine().lower()
    candidates = (
        ["arm64x-windows-msvc", "aarch64-windows-msvc"]
        if arch in ("arm64", "aarch64")
        else ["x86_64-windows-msvc"]
    )
    for target in candidates:
        exe = sdk_root / "bin" / target / "qnn-net-run.exe"
        lib = sdk_root / "lib" / target / "QnnHtp.dll"
        if exe.is_file() and lib.is_file():
            return NetRunPaths(exe, lib)
    raise ContextBinaryError(
        f"qnn-net-run.exe / QnnHtp.dll not found under {sdk_root} for any of "
        f"{candidates}. Check the QAIRT install."
    )


class NetRunContextRunner:
    """Executes a context binary by shelling out to `qnn-net-run`.

    Diagnostic-grade: a subprocess plus raw-file marshalling per call. Use it
    to prove a binary loads on hardware, not to serve frames.
    """

    def __init__(self, bin_path: Path, spec: ModelSpec, sdk_root: Path, timeout_s: float = 120.0):
        self.spec = spec
        self._bin = bin_path
        self._paths = find_net_run(sdk_root)
        self._timeout = timeout_s

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="argus-netrun-") as tmp:
            tmp_path = Path(tmp)
            entries = []
            for tensor_spec in self.spec.inputs:
                raw = tmp_path / f"{tensor_spec.name}.raw"
                np.ascontiguousarray(inputs[tensor_spec.name]).tofile(raw)
                entries.append(f"{tensor_spec.name}:={raw}")
            input_list = tmp_path / "input_list.txt"
            input_list.write_text(" ".join(entries) + "\n", encoding="utf-8")

            out_dir = tmp_path / "output"
            cmd = [
                str(self._paths.executable),
                "--backend", str(self._paths.backend),
                "--retrieve_context", str(self._bin),
                "--input_list", str(input_list),
                "--output_dir", str(out_dir),
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._timeout
            )
            if proc.returncode != 0:
                raise ContextBinaryError(
                    f"qnn-net-run failed for {self._bin.name} (exit {proc.returncode}):\n"
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
            return self._read_outputs(out_dir)

    def _read_outputs(self, out_dir: Path) -> dict[str, np.ndarray]:
        results: dict[str, np.ndarray] = {}
        for tensor_spec in self.spec.outputs:
            matches = sorted(out_dir.rglob(f"{tensor_spec.name}.raw"))
            if not matches:
                raise ContextBinaryError(
                    f"qnn-net-run produced no output named {tensor_spec.name!r} "
                    f"under {out_dir}"
                )
            data = np.fromfile(matches[0], dtype=tensor_spec.np_dtype)
            results[tensor_spec.name] = data.reshape(tensor_spec.shape)
        return results

    def close(self) -> None:
        pass
