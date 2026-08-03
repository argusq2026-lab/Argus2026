"""Determinism: the same input history must yield a byte-identical rank.

This is the product claim — the triage rank is reproducible and auditable even
though a VLM's decoding is not — so it is tested at three levels: the scorer,
the full pipeline over a fixed clip, and the serialised output.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from argus.outputs import to_json_dict
from argus.pipeline.runner import ArgusPipeline, FrameClock
from argus.triage import TrackState, compute_triage, rank_trainees
from tests.conftest import make_observation


def _history(scoring, n: int, seed_offset: float = 0.0) -> TrackState:
    state = TrackState(history_len=scoring.history_len)
    for i in range(n):
        x = 100.0 + seed_offset + (i % 5) * 3.0
        state.push(
            make_observation(
                ts=float(i), bbox=(x, 100.0, x + 60.0, 240.0), torso_y=140.0 + (i % 3)
            ),
            scoring,
        )
    return state


def test_the_scorer_is_a_pure_function(scoring):
    a = compute_triage("t0", _history(scoring, 30), 5.0, scoring)
    b = compute_triage("t0", _history(scoring, 30), 5.0, scoring)
    assert a == b


def test_rank_is_stable_across_dict_insertion_order(scoring):
    forward = {name: _history(scoring, 30, i) for i, name in enumerate("abcd")}
    reversed_order = {name: forward[name] for name in reversed(list(forward))}
    assert rank_trainees(forward, 5.0, scoring) == rank_trainees(reversed_order, 5.0, scoring)


def test_a_caption_changes_the_score_deterministically(scoring):
    with_anomaly = _history(scoring, 30)
    with_anomaly.apply_caption("trainee has fallen", scoring)
    without = _history(scoring, 30)

    a = compute_triage("t0", with_anomaly, 5.0, scoring)
    b = compute_triage("t0", with_anomaly, 5.0, scoring)
    assert a == b
    assert a.score > compute_triage("t0", without, 5.0, scoring).score


def _run_pipeline(default_config, demo_video, tmp_path, tag: str):
    cfg = dataclasses.replace(
        default_config,
        cameras=(
            dataclasses.replace(default_config.cameras[0], source=str(demo_video)),
        ),
        outputs=dataclasses.replace(
            default_config.outputs,
            console=False,
            json_log=str(tmp_path / f"{tag}.jsonl"),
            http_port=0,
            overlay_out="",
            overlay_window=False,
        ),
    )
    pipeline = ArgusPipeline(cfg, clock=FrameClock(15.0), headless=True)
    pipeline.run(max_ticks=35)
    return (tmp_path / f"{tag}.jsonl").read_text(encoding="utf-8")


@pytest.mark.timeout(180)
def test_two_runs_over_the_same_clip_are_byte_identical(
    default_config, demo_video, tmp_path
):
    """Same input, same output — including the timestamps, via the frame clock."""
    first = _run_pipeline(default_config, demo_video, tmp_path, "run_a")
    second = _run_pipeline(default_config, demo_video, tmp_path, "run_b")
    assert first == second
    assert first.strip(), "the run produced no records at all"


@pytest.mark.timeout(180)
def test_the_run_actually_ranks_trainees(default_config, demo_video, tmp_path):
    """Guards the determinism test above from passing on two empty files."""
    payload = _run_pipeline(default_config, demo_video, tmp_path, "content")
    lines = [json.loads(line) for line in payload.strip().splitlines()]
    assert len(lines) >= 30
    ranked = [line for line in lines if line["records"]]
    assert ranked, "no trainee was ever ranked"

    last = ranked[-1]["records"]
    scores = [r["score"] for r in last]
    assert scores == sorted(scores, reverse=True)
    assert len({r["trainee_id"] for r in last}) == len(last)


def test_record_serialisation_is_stable():
    from argus.triage import TriageRecord

    record = TriageRecord("cam0-t1", 0.6123, ("possible_fall", "vlm_anomaly"), 2.0)
    once = json.dumps(to_json_dict(record), sort_keys=True)
    twice = json.dumps(to_json_dict(record), sort_keys=True)
    assert once == twice
    assert json.loads(once)["reason_codes"] == ["possible_fall", "vlm_anomaly"]
