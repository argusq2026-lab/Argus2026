"""CPU ONNX backend — the reference implementation.

This backend exists to be *correct*, not fast. It runs the same artifacts and
the same pre/post code as the NPU path on any machine, which makes it the
oracle a QNN result can be diffed against and the only path CI can execute.

It cannot load a QNN context binary: a `.bin` is compiled HTP code, not a
graph. Asking for `pose_detector.bin` on this backend raises rather than
silently substituting something — a pipeline that quietly ran without pose
would still emit scores, just meaningless ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from argus.engines.base import EngineBackend, EngineError, ModelRunner
from argus.engines.metadata import ModelSpec
from argus.engines.onnx_common import create_session_with_fusion_retry


class OnnxCpuRunner:
    """One ONNX Runtime session pinned to the CPU execution provider."""

    def __init__(self, path: Path, spec: ModelSpec, optimization_level: str = "extended"):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise EngineError(
                "onnxruntime is required for engine.kind = 'onnx-cpu'"
            ) from exc

        self.spec = spec
        self._path = path
        try:
            self._session = create_session_with_fusion_retry(
                lambda options: ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                ),
                optimization_level,
                label=path.name,
            )
        except Exception as exc:
            raise EngineError(f"failed to load {path}: {exc}") from exc

        session_outputs = [o.name for o in self._session.get_outputs()]
        declared = [o.name for o in spec.outputs]
        if set(session_outputs) != set(declared):
            raise EngineError(
                f"{path.name}: session outputs {session_outputs} do not match "
                f"metadata.json {declared}; the artifact and its manifest disagree"
            )
        self._output_names = session_outputs

    @property
    def output_names(self) -> list[str]:
        """Output names as the *session* reports them, not as metadata declares.

        Used by provisioning to verify an artifact describes itself honestly.
        """
        return list(self._output_names)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        results = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, results))

    def close(self) -> None:
        self._session = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OnnxCpuRunner {self._path.name}>"


class OnnxCpuBackend(EngineBackend):
    kind = "onnx-cpu"

    def __init__(self, optimization_level: str = "extended"):
        self._optimization_level = optimization_level

    def load(self, path: Path, spec: ModelSpec) -> ModelRunner:
        if spec.is_context_binary:
            raise EngineError(
                f"{spec.file_name} is a QNN context binary and has no CPU path. "
                "Set engine.kind = 'qnn-npu' to run the pose stage, or "
                "engine.kind = 'mock' to develop without it. Argus will not "
                "silently run a pipeline with pose missing."
            )
        if not path.is_file():
            raise EngineError(
                f"model artifact not found: {path}. Run `argus bootstrap` — "
                "models/ is gitignored, so a fresh clone has none."
            )
        return OnnxCpuRunner(path, spec, self._optimization_level)
