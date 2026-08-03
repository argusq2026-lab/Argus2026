"""The engine interface every backend implements.

One interface, three implementations — mock, CPU-ONNX, NPU-QNN — selected by
config. A backend is a factory for :class:`ModelRunner`s: thin objects that
take a dict of named input arrays and return a dict of named output arrays,
with the declared :class:`~argus.engines.metadata.ModelSpec` attached.

Pre/post-processing lives in :mod:`argus.vision`, never in a backend, so all
three backends are interchangeable and the CPU path is a genuine reference
implementation of the NPU path rather than a parallel one.

There is no automatic degradation anywhere in this module. If `qnn-npu` cannot
place a graph on the Hexagon NPU it raises; it does not return a CPU session
that produces correct numbers at a tenth of the speed and lets a latency
budget silently become fiction.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from argus.engines.metadata import ModelSpec, TensorSpec


class EngineError(RuntimeError):
    """Backend construction or execution failure."""


class ContractViolation(EngineError):
    """A tensor did not match the shape/dtype declared in metadata.json."""


@runtime_checkable
class ModelRunner(Protocol):
    """A loaded artifact ready to execute."""

    spec: ModelSpec

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute once. Keys are the names declared in metadata.json."""
        ...

    def close(self) -> None: ...


def check_tensor(spec: TensorSpec, array: np.ndarray, *, where: str) -> None:
    """Assert an array matches a declared contract, or raise with specifics.

    Batch (axis 0) is allowed to differ so a runner can be fed a single sample
    against a batched declaration; every other axis must match exactly.
    """
    if array.dtype != spec.np_dtype:
        raise ContractViolation(
            f"{where}: tensor {spec.name!r} dtype is {array.dtype}, "
            f"metadata.json declares {spec.dtype}"
        )
    if len(array.shape) != len(spec.shape):
        raise ContractViolation(
            f"{where}: tensor {spec.name!r} has rank {len(array.shape)} "
            f"{array.shape}, metadata.json declares rank {len(spec.shape)} {spec.shape}"
        )
    mismatched = [
        (i, got, want)
        for i, (got, want) in enumerate(zip(array.shape, spec.shape))
        if i != 0 and got != want
    ]
    if mismatched:
        raise ContractViolation(
            f"{where}: tensor {spec.name!r} shape {array.shape} does not match "
            f"metadata.json {spec.shape} (axis mismatches: {mismatched})"
        )


def check_inputs(spec: ModelSpec, inputs: dict[str, np.ndarray]) -> None:
    declared = {s.name for s in spec.inputs}
    got = set(inputs)
    if got != declared:
        raise ContractViolation(
            f"{spec.file_name}: input names {sorted(got)} != declared {sorted(declared)}"
        )
    for tensor_spec in spec.inputs:
        check_tensor(tensor_spec, inputs[tensor_spec.name], where=f"{spec.file_name} input")


def check_outputs(spec: ModelSpec, outputs: dict[str, np.ndarray]) -> None:
    declared = {s.name for s in spec.outputs}
    got = set(outputs)
    if got != declared:
        raise ContractViolation(
            f"{spec.file_name}: output names {sorted(got)} != declared {sorted(declared)}"
        )
    for tensor_spec in spec.outputs:
        check_tensor(tensor_spec, outputs[tensor_spec.name], where=f"{spec.file_name} output")


class EngineBackend(abc.ABC):
    """Creates runners for artifacts. One instance per process per config."""

    #: Config value that selects this backend.
    kind: str = ""

    @abc.abstractmethod
    def load(self, path: Path, spec: ModelSpec) -> ModelRunner:
        """Load one artifact. Raises :class:`EngineError` on any failure."""

    def close(self) -> None:  # pragma: no cover - most backends are stateless
        """Release process-wide backend resources."""


class ValidatingRunner:
    """Wraps a runner so every call is contract-checked in both directions.

    Applied to every backend, including mock: a mock that drifts from the real
    contract is worse than no mock, because it makes the pipeline look correct
    against tensors the hardware would never produce.
    """

    def __init__(self, inner: ModelRunner):
        self._inner = inner
        self.spec = inner.spec

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        check_inputs(self.spec, inputs)
        outputs = self._inner.run(inputs)
        check_outputs(self.spec, outputs)
        return outputs

    def close(self) -> None:
        self._inner.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ValidatingRunner {self.spec.file_name} via {self._inner!r}>"
