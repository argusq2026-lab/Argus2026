"""Parse the AI Hub `metadata.json` that ships beside every exported artifact.

The metadata is the tensor contract: shapes, dtypes, and the quantization
(scale, zero_point) pairs needed to dequantize a w8a8 output. Argus reads it
rather than hardcoding shapes, because the prototype's hardcoded contracts
were wrong for all three models and nothing caught it — the pre/post code
never met the real artifacts.

Reading the contract from the artifact means a re-export that changes an
input layout or an output count fails at load with a specific message,
instead of producing plausible-looking garbage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class MetadataError(ValueError):
    """Raised when metadata.json is missing, malformed, or contradicts the artifact."""


@dataclass(frozen=True)
class TensorSpec:
    """One declared input or output tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # numpy dtype name, e.g. "uint8"
    scale: float | None = None
    zero_point: int | None = None

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def is_quantized(self) -> bool:
        return self.scale is not None

    def dequantize(self, array: np.ndarray) -> np.ndarray:
        """(q - zero_point) * scale, in float32. Identity if not quantized."""
        if not self.is_quantized:
            return array.astype(np.float32)
        zp = self.zero_point or 0
        return (array.astype(np.float32) - float(zp)) * float(self.scale)

    def quantize(self, array: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`dequantize`, clipped to the dtype's range."""
        if not self.is_quantized:
            return array.astype(self.np_dtype)
        zp = self.zero_point or 0
        info = np.iinfo(self.np_dtype)
        q = np.rint(array.astype(np.float32) / float(self.scale) + zp)
        return np.clip(q, info.min, info.max).astype(self.np_dtype)


@dataclass(frozen=True)
class ModelSpec:
    """The full contract for one artifact file."""

    model_id: str
    model_name: str
    runtime: str  # "onnx" | "qnn_context_binary"
    precision: str  # "w8a8"
    file_name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    qairt_version: str | None = None

    def input(self, name: str | None = None) -> TensorSpec:
        if name is None:
            if len(self.inputs) != 1:
                raise MetadataError(
                    f"{self.file_name} has {len(self.inputs)} inputs; name one explicitly"
                )
            return self.inputs[0]
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise MetadataError(f"{self.file_name}: no input named {name!r}")

    def output(self, name: str) -> TensorSpec:
        for spec in self.outputs:
            if spec.name == name:
                return spec
        raise MetadataError(
            f"{self.file_name}: no output named {name!r} "
            f"(has {[s.name for s in self.outputs]})"
        )

    @property
    def is_context_binary(self) -> bool:
        return self.runtime == "qnn_context_binary"


def _tensor_specs(block: dict[str, Any], kind: str, file_name: str) -> tuple[TensorSpec, ...]:
    specs = []
    for name, meta in (block or {}).items():
        if "shape" not in meta or "dtype" not in meta:
            raise MetadataError(f"{file_name}: {kind} {name!r} lacks shape/dtype")
        q = meta.get("quantization_parameters") or {}
        specs.append(
            TensorSpec(
                name=name,
                shape=tuple(int(d) for d in meta["shape"]),
                dtype=str(meta["dtype"]),
                scale=float(q["scale"]) if "scale" in q else None,
                zero_point=int(q["zero_point"]) if "zero_point" in q else None,
            )
        )
    if not specs:
        raise MetadataError(f"{file_name}: no {kind}s declared")
    return tuple(specs)


def load_model_specs(metadata_path: str | Path) -> dict[str, ModelSpec]:
    """Parse a metadata.json into {file_name: ModelSpec}.

    A single metadata.json can describe several files — MediaPipe-Pose ships
    a detector and a landmark binary under one manifest.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise MetadataError(
            f"metadata.json not found: {path}. Run `argus bootstrap` to provision "
            "models/ — a fresh clone has no artifacts (models/ is gitignored)."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"{path}: invalid JSON: {exc}") from exc

    files = raw.get("model_files")
    if not isinstance(files, dict) or not files:
        raise MetadataError(f"{path}: missing `model_files`")

    qairt = (raw.get("tool_versions") or {}).get("qairt")
    specs: dict[str, ModelSpec] = {}
    for file_name, block in files.items():
        specs[file_name] = ModelSpec(
            model_id=str(raw.get("model_id", "")),
            model_name=str(raw.get("model_name", "")),
            runtime=str(raw.get("runtime", "")),
            precision=str(raw.get("precision", "")),
            file_name=file_name,
            inputs=_tensor_specs(block.get("inputs"), "input", file_name),
            outputs=_tensor_specs(block.get("outputs"), "output", file_name),
            qairt_version=str(qairt) if qairt else None,
        )
    return specs


def load_model_spec(metadata_path: str | Path, file_name: str) -> ModelSpec:
    """Parse a metadata.json and return the contract for one named file."""
    specs = load_model_specs(metadata_path)
    if file_name not in specs:
        raise MetadataError(
            f"{metadata_path}: no entry for {file_name!r} (has {sorted(specs)})"
        )
    return specs[file_name]
