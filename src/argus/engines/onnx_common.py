"""Shared ONNX Runtime session construction for the CPU and NPU backends.

Exists for one specific, reproducible failure. The shipped QuickSRNet-Medium
w8a8 artifact loads fine at optimization levels `disabled` and `basic`, and
fails at `extended` and `all` with::

    This is an invalid model. Error: two nodes with same node name (/model/cnn/0/Conv).

The message points at the artifact, but the artifact is valid: `onnx.checker`
passes with `full_check=True`, all 56 node names are unique, no node name
collides with an initializer, and no output name is duplicated. The collision
is created *during* ORT's extended-level QDQ fusion, which names a fused node
after a node that still exists. It is an ONNX Runtime issue on this graph
shape, not a bad export — which matters, because "re-export the model" is the
wrong remedy and would waste an AI Hub round trip.

So: try the configured level, and on exactly this failure retry one level down
with a warning that says which of the two it is. Dropping a fusion level
changes performance, not numerics, so this is not the kind of silent
degradation `engine.allow_cpu_fallback` guards against — but it is still
announced rather than hidden.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

#: Config value -> ORT enum name.
OPTIMIZATION_LEVELS = {
    "disabled": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}

#: The level to retry at. `basic` still does constant folding and redundant-node
#: elimination; it just skips the QDQ fusion transform that collides.
_RETRY_LEVEL = "basic"

#: Substring identifying the ORT fusion name collision described above.
_FUSION_COLLISION = "two nodes with same node name"


def make_session_options(level: str, **extra: Any):
    """An `ort.SessionOptions` at the named optimization level."""
    import onnxruntime as ort

    if level not in OPTIMIZATION_LEVELS:
        raise ValueError(
            f"unknown graph_optimization_level {level!r}; "
            f"expected one of {sorted(OPTIMIZATION_LEVELS)}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, OPTIMIZATION_LEVELS[level]
    )
    options.log_severity_level = 3  # warnings and above
    for key, value in extra.items():
        setattr(options, key, value)
    return options


def create_session_with_fusion_retry(
    build: Callable[[Any], Any], level: str, *, label: str
):
    """Build a session at `level`; retry at `basic` on the ORT fusion collision.

    `build` takes an `ort.SessionOptions` and returns a session, so the same
    retry applies to a plain CPU session and to a QNN EP session.
    """
    try:
        return build(make_session_options(level))
    except Exception as exc:
        if _FUSION_COLLISION not in str(exc) or level == _RETRY_LEVEL:
            raise
        print(
            f"[WARN] {label}: ONNX Runtime raised {_FUSION_COLLISION!r} while "
            f"applying '{level}' graph optimizations. The artifact is valid "
            "(onnx.checker passes, node names are unique) -- the collision is "
            "created by ORT's QDQ fusion on this graph. Retrying at "
            f"'{_RETRY_LEVEL}'; numerics are unchanged, fusion is not.",
            file=sys.stderr,
        )
        return build(make_session_options(_RETRY_LEVEL))
