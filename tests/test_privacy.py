"""Privacy is a property of the wiring, so it is tested structurally.

Not "does a redaction filter strip frames" — a filter can be bypassed by the
next person who adds an argument — but "can any type that could hold pixels or
free text reach a sink at all". These tests inspect module imports and the type
annotations of every public callable on the boundary, so widening it fails CI
rather than review.

The boundary used to be "no camera frame or raw VLM caption may reach a
sink"; it is now stronger, not weaker: no frame ever exists past a phone's own
on-device pose/form model in the first place, and there is no free text
anywhere in the system — a phone's form-error codes are already a closed
vocabulary by the time `argus.ingest.protocol` accepts them.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import typing
from pathlib import Path

import pytest

import argus.alerts as alerts
import argus.console as console
import argus.ingest.protocol as ingest_protocol
import argus.ingest.server as ingest_server
import argus.ingest.session as ingest_session
import argus.outputs as outputs
from argus.triage import TriageRecord

#: Modules that sit on or outside the alert boundary. Nothing here may hold a
#: frame, a crop, or free text.
BOUNDARY_MODULES = (
    alerts,
    console,
    outputs,
    ingest_protocol,
    ingest_session,
    ingest_server,
)

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


@pytest.mark.parametrize("module", (alerts, outputs), ids=lambda m: m.__name__)
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


# -- the console's view -------------------------------------------------------
#
# `StationView` is the one type here wider than a `TriageRecord`: a console
# that draws a skeleton needs the numeric pose, and four scalar fields cannot
# express one. So it gets the same closed-set discipline rather than an
# exemption -- widening what a trainer's screen can see stays a visible,
# reviewable change. See the module docstring in argus/outputs.py.


def test_station_view_fields_are_the_closed_set():
    fields = {f.name for f in dataclasses.fields(outputs.StationView)}
    assert fields == {
        "station_id",
        "trainee_id",
        "connected",
        "last_seen_ts",
        "observations",
        "bbox_xyxy",
        "keypoints_xy",
        "keypoints_conf",
        "form_reason_codes",
        "exercise",
        # Nursing's counterpart to `exercise`: a short label ("cpr"), length
        # bounded by `argus.ingest.protocol` and rendered as text, never markup.
        "procedure",
        "rep_count",
        "form_ok",
        "session",
        "display_name",
        "subject_present",
        "use_case",
    }


def test_session_summary_fields_are_the_closed_set():
    """The nested view gets the same discipline as the one that holds it.
    Session accounting is where "just one more number for the dashboard"
    pressure lands, so what it may carry is a reviewed list."""
    fields = {f.name for f in dataclasses.fields(outputs.SessionSummary)}
    assert fields == {
        "rolling_score",
        "peak_score",
        "active_s",
        "reps",
        "reps_flagged",
        "hold_s",
        "hold_flagged_s",
        "fault_rate",
        "code_counts",
    }


def test_session_summary_holds_no_image_capable_field():
    for field in dataclasses.fields(outputs.SessionSummary):
        text = str(field.type)
        for forbidden in FORBIDDEN_TYPE_NAMES:
            assert forbidden not in text, f"SessionSummary.{field.name} can hold {forbidden}"


def test_station_view_is_frozen():
    view = outputs.StationView("s0", "t0", True, 1.0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.trainee_id = "t1"  # type: ignore[misc]


def test_station_view_holds_no_image_capable_field():
    for field in dataclasses.fields(outputs.StationView):
        text = str(field.type)
        for forbidden in FORBIDDEN_TYPE_NAMES:
            assert forbidden not in text, f"StationView.{field.name} can hold {forbidden}"


def test_a_station_view_cannot_reach_an_alert_sink():
    """The console's wider view is console-only. `emit_alert` and the JSON log
    still take a `TriageRecord`, so there is no parameter anywhere on the
    alert boundary through which a keypoint could travel."""
    for func in (alerts.emit_alert, outputs.JsonLogSink.write):
        hints = typing.get_type_hints(func)
        assert not any("StationView" in str(annotation) for annotation in hints.values())


def test_the_json_log_did_not_widen(tmp_path):
    """The console reads keypoints; what is written to disk did not change."""
    import json

    sink = outputs.JsonLogSink(tmp_path / "triage.jsonl")
    sink.write(1.0, [TriageRecord("t0", 0.5, (), 1.0)])
    sink.close()
    payload = json.loads((tmp_path / "triage.jsonl").read_text(encoding="utf-8"))
    assert set(payload) == {"ts", "records"}
    assert set(payload["records"][0]) == {"trainee_id", "score", "reason_codes", "ts"}


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
    sink.write(1.0, [TriageRecord("t0", 0.75, ("possible_fall",), 1.0)])
    sink.close()

    payload = json.loads((tmp_path / "triage.jsonl").read_text(encoding="utf-8"))
    for record in payload["records"]:
        assert set(record) == {"trainee_id", "score", "reason_codes", "ts"}


def test_station_session_holds_no_image_capable_field(scoring):
    """The ingest layer's per-trainee state must not be able to hold pixels
    either -- the network-era equivalent of the old per-camera Track check."""
    from argus.ingest.session import StationSession
    from argus.triage import TrackState

    session = StationSession(
        station_id="s0", trainee_id="t0", track=TrackState(history_len=scoring.history_len),
        last_seen_ts=0.0,
    )
    for field in dataclasses.fields(session):
        assert "ndarray" not in str(field.type), f"StationSession.{field.name} can hold pixels"


def test_ingest_protocol_rejects_a_form_code_outside_the_closed_vocabulary():
    """A phone cannot smuggle free text into the score through a reason code:
    only codes drawn from the configured vocabulary are ever accepted."""
    from argus.ingest.protocol import ProtocolError, parse_observation

    raw = {
        "type": "observation", "ts": 0.0,
        "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
        "keypoints_xy": [[0.0, 0.0]] * 17,
        "keypoints_conf": [0.0] * 17,
        "form_reason_codes": ["trainee looks unwell, possible medical event"],
    }
    with pytest.raises(ProtocolError):
        parse_observation(raw, {"knee_valgus": 0.8})
