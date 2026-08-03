"""NPU backend — Hexagon HTP via the QNN Execution Provider.

Handles both artifact forms Argus ships with:

* **QDQ ONNX** (YOLO-X, QuickSRNet) — loaded straight through
  :func:`argus.engines.ort_qnn.create_npu_session`, with the persistent HTP
  context cache enabled so graph compilation is a one-time cost.
* **QNN context binary** (both BlazePose stages) — wrapped in an EPContext
  ONNX model first, since ORT cannot open a `.bin`, then loaded with context
  *generation* disabled because the context already exists.

Nothing here degrades quietly. If the QNN EP is unavailable, or ends up not
being the active provider, or the artifact was built against a different QAIRT
than the one installed, construction raises with a message naming the cause.
`engine.allow_cpu_fallback` exists only for deliberate A/B measurement and is
loud when used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from argus.config import EngineConfig
from argus.engines import ort_qnn
from argus.engines.base import EngineBackend, EngineError, ModelRunner
from argus.engines.metadata import ModelSpec
from argus.engines.onnx_common import create_session_with_fusion_retry, make_session_options
from argus.engines.qnn_context import (
    ContextBinaryError,
    NetRunContextRunner,
    build_epcontext_model,
    check_qairt_compatibility,
    epcontext_wrapper_path,
    find_qairt_sdk,
)


class OrtSessionRunner:
    """An ORT session whose active provider has been verified to be the NPU."""

    def __init__(self, session, spec: ModelSpec, source: Path):
        self.spec = spec
        self._session = session
        self._source = source
        self._output_names = [o.name for o in session.get_outputs()]
        declared = {o.name for o in spec.outputs}
        if set(self._output_names) != declared:
            raise EngineError(
                f"{source.name}: session outputs {sorted(self._output_names)} do not "
                f"match metadata.json {sorted(declared)}"
            )

    @property
    def active_provider(self) -> str:
        return self._session.get_providers()[0]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        results = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, results))

    def close(self) -> None:
        self._session = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OrtSessionRunner {self._source.name} on {self.active_provider}>"


class QnnNpuBackend(EngineBackend):
    kind = "qnn-npu"

    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._sdk_root = find_qairt_sdk(cfg.qairt_sdk_root)
        if sys.platform == "win32" and not ort_qnn.native_arm64_python():
            raise EngineError(
                "engine.kind = 'qnn-npu' requires a native ARM64 interpreter. "
                f"This one reports {__import__('platform').machine()!r}, so the "
                "win-arm64 onnxruntime-qnn wheel cannot load here at all. Use the "
                "ARM64 venv (.venv-npu) created by `run.ps1 -Npu`."
            )

    def load(self, path: Path, spec: ModelSpec) -> ModelRunner:
        if not path.is_file():
            raise EngineError(
                f"model artifact not found: {path}. Run `argus bootstrap` — "
                "models/ is gitignored, so a fresh clone has none."
            )
        warning = check_qairt_compatibility(spec, self._sdk_root)
        if warning and not self._cfg.allow_cpu_fallback:
            raise EngineError(warning)
        if warning:
            print(f"[WARN] {warning}", file=sys.stderr)

        if spec.is_context_binary:
            return self._load_context_binary(path, spec)
        return self._load_onnx(path, spec)

    # -- artifact forms -----------------------------------------------------

    def _load_onnx(self, path: Path, spec: ModelSpec) -> ModelRunner:
        session = create_session_with_fusion_retry(
            lambda options: ort_qnn.create_npu_session(
                path,
                allow_fallback=self._cfg.allow_cpu_fallback,
                cache_dir=self._cfg.cache_dir,
                generate_context=True,
                session_options=options,
            ),
            self._cfg.graph_optimization_level,
            label=path.name,
        )
        runner = OrtSessionRunner(session, spec, path)
        if self._cfg.allow_cpu_fallback and runner.active_provider == ort_qnn.CPU_EP:
            print(
                f"[WARN] {path.name} is running on the CPU, not the NPU "
                "(engine.allow_cpu_fallback = true). Latency figures from this "
                "run are not NPU figures.",
                file=sys.stderr,
            )
        return runner

    def _load_context_binary(self, path: Path, spec: ModelSpec) -> ModelRunner:
        if self._cfg.context_binary_mode == "netrun":
            if self._sdk_root is None:
                raise ContextBinaryError(
                    "engine.context_binary_mode = 'netrun' needs the QAIRT SDK. "
                    "Set engine.qairt_sdk_root or QNN_SDK_ROOT."
                )
            return NetRunContextRunner(path, spec, self._sdk_root)

        wrapper = epcontext_wrapper_path(path, Path(self._cfg.cache_dir))
        if not wrapper.is_file() or wrapper.stat().st_mtime < path.stat().st_mtime:
            build_epcontext_model(path, spec, wrapper)

        session = ort_qnn.create_npu_session(
            wrapper,
            allow_fallback=self._cfg.allow_cpu_fallback,
            check_format=False,  # the wrapper has one opaque node, nothing to inspect
            cache_dir=self._cfg.cache_dir,
            generate_context=False,  # the context IS the .bin; do not regenerate
            session_options=make_session_options(self._cfg.graph_optimization_level),
        )
        return OrtSessionRunner(session, spec, path)
