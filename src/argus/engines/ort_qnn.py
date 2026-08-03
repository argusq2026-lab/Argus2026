# SPDX-License-Identifier: Apache-2.0
#
# This file alone is licensed Apache-2.0, not MIT like the rest of Argus.
# Derived from quad_mcp_client/ort_qnn.py, Copyright QUAD Contributors,
# github.com/CBN-AI-TEAM/QUAD-Client. See THIRD-PARTY-NOTICES.md.
"""QNN Execution Provider session helper — vendored from QUAD-Client.

Upstream: ``quad_mcp_client/ort_qnn.py`` in github.com/CBN-AI-TEAM/QUAD-Client
(Apache-2.0). Vendored rather than depended on because ``quad-mcp-client`` is
not published on PyPI — the prototype worked around that by inserting
``../../src`` into ``sys.path`` at a call site, which only worked while Argus
lived inside that repo's tree.

Trimmed to what Argus needs: session creation with **no silent CPU fallback**,
provider verification, the ConvInteger pre-check, the persistent HTP context
cache, and the double-registration guard. Upstream's profiling-JSON
``provider_split`` helper is not carried over.

The guard rail that matters: ONNX Runtime silently falls back to the CPU EP
when QNN initialisation fails. Inference still completes and the numbers still
look right — the *only* signal is ``session.get_providers()[0]``. Everything
here exists so that failure is an exception rather than a latency mystery.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"
# ORT 1.27+ registers the QNN plugin EP under the short name "QNN".
_QNN_EP_ALIASES = frozenset({"QNNExecutionProvider", "QNN"})

_registration_lock = threading.Lock()
_qnn_registered = False

#: Integer-compute / dynamic-quant ops that mark an ONNX model as ConvInteger
#: style. The HTP accepts only QDQ; a ConvInteger model loads without error and
#: is silently executed on the CPU.
_CONVINTEGER_OPS = frozenset(
    {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul"}
)


class NPUFallbackError(RuntimeError):
    """The QNN EP is not the active provider, and fallback was not opted into."""

    def __init__(self, active_provider: str, model_path: str = "") -> None:
        self.active_provider = active_provider
        self.model_path = model_path
        super().__init__(
            f"NPU (QNNExecutionProvider) is not active — ONNX Runtime is using "
            f"'{active_provider}' for {model_path or 'this model'}. Inference "
            "would run on the CPU, not the NPU. Common causes: plain "
            "'onnxruntime' is installed instead of 'onnxruntime-qnn'; the "
            "interpreter is not native ARM64; the model is not QNN-compatible; "
            "or the QNN backend libraries are not on PATH."
        )


class ConvIntegerFormatError(RuntimeError):
    """An INT8 model uses ConvInteger format, which the HTP silently rejects."""

    def __init__(self, ops: list[str], model_path: str = "") -> None:
        self.ops = ops
        self.model_path = model_path
        op_str = ", ".join(sorted(set(ops))) or "ConvInteger"
        super().__init__(
            f"{model_path or 'model'} uses ConvInteger INT8 format ({op_str}), "
            "which is not supported on the Hexagon HTP. The NPU requires QDQ "
            "(QuantizeLinear/DequantizeLinear) quantization; a ConvInteger model "
            "runs on CPU with no error at all. Re-quantize with QuantFormat.QDQ."
        )


def convinteger_ops(model_path: str | Path) -> list[str]:
    """ConvInteger-style op types present in an ONNX model.

    An empty list means the model is QDQ or float — or that it could not be
    inspected, in which case we never block: a probe failure must not break the
    flow.
    """
    if not str(model_path).lower().endswith(".onnx"):
        return []
    try:
        import onnx
    except ImportError:
        return []
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return []
    return [n.op_type for n in model.graph.node if n.op_type in _CONVINTEGER_OPS]


def qnn_backend_library() -> str:
    """The QNN HTP backend library for this OS. HTP = Hexagon Tensor Processor."""
    return "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"


def native_arm64_python() -> bool:
    """Whether this interpreter is native ARM64.

    The NPU wheels are win-arm64-only; an emulated x86-64 interpreter on a
    Snapdragon host cannot load them, which is the single most common reason
    the QNN EP is "unavailable" on hardware that plainly has an NPU.
    """
    import platform

    return platform.machine().lower() in ("arm64", "aarch64")


def qnn_provider_options(
    cache_dir: str | Path | None, *, generate_context: bool = True
) -> dict[str, str]:
    """Provider options for the QNN EP.

    `generate_context` must be False when loading a model that *is already* an
    EPContext wrapper around a pre-compiled `.bin` — asking ORT to generate a
    context for a model that is one is both wasteful and, on some versions, an
    error.
    """
    opts: dict[str, str] = {"backend_path": qnn_backend_library()}
    if cache_dir is not None and generate_context:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        # Persist the compiled HTP context so graph compilation is a one-time
        # cost rather than a 10-20x penalty on every process start.
        opts["ep.context_enable"] = "1"
        opts["ep.context_file_path"] = str(cache / "qnn_context.onnx")
    return opts


def register_qnn_ep() -> str | None:
    """Register the QNN plugin EP if present; return its provider name.

    ORT 1.27+ ships onnxruntime-qnn as a *plugin* EP: it does not appear in
    ``get_available_providers()`` until it is registered, so a check made
    before registration always reports the NPU as missing.
    """
    global _qnn_registered
    import onnxruntime as ort

    try:
        import onnxruntime_qnn as qnn_pkg
    except ImportError:
        qnn_pkg = None

    if qnn_pkg is not None and hasattr(ort, "register_execution_provider_library"):
        with _registration_lock:
            if not _qnn_registered:
                ort.register_execution_provider_library("QNN", qnn_pkg.get_library_path())
                _qnn_registered = True

    return next((ep for ep in ort.get_available_providers() if ep in _QNN_EP_ALIASES), None)


def create_npu_session(
    model_path: str | Path,
    *,
    allow_fallback: bool = False,
    check_format: bool = True,
    cache_dir: str | Path | None = None,
    generate_context: bool = True,
    session_options: Any = None,
):
    """Create an ONNX Runtime session pinned to the Hexagon NPU.

    Raises :class:`NPUFallbackError` when the QNN EP does not end up active and
    `allow_fallback` is False.
    """
    import onnxruntime as ort

    if check_format and not allow_fallback:
        conv_ops = convinteger_ops(model_path)
        if conv_ops:
            raise ConvIntegerFormatError(conv_ops, str(model_path))

    qnn_ep = register_qnn_ep()
    if qnn_ep is None and not allow_fallback:
        arch_hint = (
            ""
            if native_arm64_python()
            else " This interpreter is not native ARM64, so the win-arm64 QNN "
            "wheel cannot be loaded here at all."
        )
        raise RuntimeError(
            "QNNExecutionProvider is not available in this onnxruntime build. "
            "Install 'onnxruntime-qnn' (not plain 'onnxruntime') into a native "
            "ARM64 Python 3.10-3.12 environment. Available providers: "
            f"{ort.get_available_providers()}.{arch_hint}"
        )

    providers: list[Any] = []
    if qnn_ep is not None:
        providers.append((qnn_ep, qnn_provider_options(cache_dir, generate_context=generate_context)))
    providers.append(CPU_EP)

    session = ort.InferenceSession(
        str(model_path), sess_options=session_options, providers=providers
    )

    active = session.get_providers()[0]
    if active not in _QNN_EP_ALIASES and not allow_fallback:
        raise NPUFallbackError(active, str(model_path))
    return session


def assert_npu_active(session) -> None:
    """The one reliable check that inference is on the NPU.

    Completing inference is not proof: ORT silently uses the CPU when QNN
    fails.
    """
    active = session.get_providers()[0]
    if active not in _QNN_EP_ALIASES:
        raise NPUFallbackError(active)


def reset_registration_guard() -> None:
    """Clear the process-wide registration guard. Tests only."""
    global _qnn_registered
    with _registration_lock:
        _qnn_registered = False
