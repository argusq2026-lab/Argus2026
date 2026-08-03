"""Privacy is a property of the wiring, so it is tested structurally.

Not "does a redaction filter strip frames" — a filter can be bypassed by the
next person who adds an argument — but "can any type that could hold pixels or
free text reach a sink at all". These tests inspect module imports and the type
annotations of every public callable on the boundary, so widening it fails CI
rather than review.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import typing
from pathlib import Path

import pytest

import argus.alerts as alerts
import argus.outputs as outputs
from argus.triage import TriageRecord

#: Modules that sit on or outside the alert boundary. Nothing here may hold a
#: frame, a crop, or a caption.
BOUNDARY_MODULES = (alerts, outputs)

#: Types that can carry imagery or free-form model output.
FORBIDDEN_TYPE_NAMES = {"ndarray", "Mat", "Image", "bytes", "bytearray", "memoryview"}

#: Packages whose presence would mean a boundary module can even name an image.
FORBIDDEN_IMPORTS = {"cv2", "numpy", "PIL", "onnxruntime", "onnxruntime_genai"}


def _module_path(module) -> Path:
    return Path(inspect.getfile(module))


@pytest.mark.parametrize("module", BOUNDARY_MODULES, ids=lambda m: m.__name__)
def test_boundary_modules_import_no_image_library(module):
    """A module that cannot name an image type cannot leak one."""
    tree = ast.parse(_module_path(module).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    leaked = imported & FORBIDDEN_IMPORTS
    assert not leaked, f"{module.__name__} imports {leaked}, widening the boundary"


@pytest.mark.parametrize("module", BOUNDARY_MODULES, ids=lambda m: m.__name__)
def test_no_public_callable_accepts_an_image_type(module):
    """Every parameter on the boundary must be a scalar, a path, or a record."""
    offenders = []
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        callables = []
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            callables.append((name, obj))
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for meth_name, meth in vars(obj).items():
                if inspect.isfunction(meth) and not meth_name.startswith("_"):
                    callables.append((f"{name}.{meth_name}", meth))

        for qualname, func in callables:
            hints = typing.get_type_hints(func)
            for param, annotation in hints.items():
                text = str(annotation)
                for forbidden in FORBIDDEN_TYPE_NAMES:
                    if forbidden in text:
                        offenders.append(f"{module.__name__}.{qualname}({param}: {text})")
    assert not offenders, f"image-capable parameters on the alert boundary: {offenders}"


def test_emit_alert_accepts_exactly_one_redacted_record():
    signature = inspect.signature(alerts.emit_alert)
    assert list(signature.parameters) == ["record"]
    assert typing.get_type_hints(alerts.emit_alert)["record"] is TriageRecord


def test_triage_record_fields_are_the_closed_set():
    """Adding a field here is the only way to widen what leaves perception."""
    fields = {f.name: f.type for f in dataclasses.fields(TriageRecord)}
    assert set(fields) == {"trainee_id", "score", "reason_codes", "ts"}


def test_triage_record_is_frozen():
    record = TriageRecord("t0", 0.5, ("possible_fall",), 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.trainee_id = "t1"  # type: ignore[misc]


def test_track_state_never_retains_a_caption(scoring):
    """A caption is scored into a number and dropped, not stored."""
    from argus.triage import TrackState

    state = TrackState(history_len=scoring.history_len)
    state.apply_caption("smoke and no helmet", scoring)
    assert state.last_vlm_anomaly_score > 0.0
    serialised = repr(state)
    assert "smoke" not in serialised
    assert "helmet" not in serialised


def test_triage_module_holds_no_pixels():
    """The scorer must not be able to touch imagery even accidentally."""
    import argus.triage as triage

    tree = ast.parse(_module_path(triage).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN_IMPORTS)


def test_json_payload_carries_only_the_four_fields(tmp_path):
    """What is written to disk is what was declared, field for field."""
    import json

    sink = outputs.JsonLogSink(tmp_path / "triage.jsonl")
    sink.write(1.0, [TriageRecord("cam0-t0", 0.75, ("possible_fall",), 1.0)])
    sink.close()

    payload = json.loads((tmp_path / "triage.jsonl").read_text(encoding="utf-8"))
    for record in payload["records"]:
        assert set(record) == {"trainee_id", "score", "reason_codes", "ts"}


def test_pipeline_does_not_store_frames_on_a_track():
    """The runner keeps frames as locals; a Track must hold none of them."""
    from argus.tracking.tracker import Track

    field_types = {f.name: str(f.type) for f in dataclasses.fields(Track)}
    for name, annotation in field_types.items():
        if name == "signature":
            continue  # a normalised histogram, not recoverable imagery
        assert "ndarray" not in annotation, f"Track.{name} can hold pixels"
