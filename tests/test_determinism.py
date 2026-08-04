"""Determinism: the same input history must yield a byte-identical rank.

This is the product claim — the triage rank is reproducible and auditable —
tested at three levels: the scorer itself, the full ingest scoring path over
a canned observation fixture, and the serialised output. Unlike the
prototype's VLM-caption design, nothing upstream of the scorer is free text
or non-deterministic decoding anymore: a phone's on-device classifier emits
closed-vocabulary codes directly, so there is nothing left to neutralize.
"""

from __future__ import annotations

import json

import pytest

from argus.ingest.session import SessionRegistry
from argus.outputs import JsonLogSink, to_json_dict
from argus.synthetic import TRAINEE_IDS, synthetic_tick
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


def test_a_form_code_changes_the_score_deterministically(scoring):
    with_error = _history(scoring, 30)
    with_error.push(make_observation(ts=30.0, form_reason_codes=("knee_valgus",)), scoring)
    without = _history(scoring, 30)

    a = compute_triage("t0", with_error, 5.0, scoring)
    b = compute_triage("t0", with_error, 5.0, scoring)
    assert a == b
    assert a.score > compute_triage("t0", without, 5.0, scoring).score


# -- the ingest scoring path, replayed offline -------------------------------
#
# Driven directly against SessionRegistry + rank_trainees rather than over a
# real WebSocket: the network layer is only framing, and a real socket would
# make this test's timing (and therefore its byte-identical claim) depend on
# the OS scheduler instead of the code under test.


def _replay(scoring, n_ticks: int, tag: str, tmp_path) -> str:
    registry = SessionRegistry(scoring, track_ttl_s=1e9)
    for trainee_id in TRAINEE_IDS:
        registry.register(f"station-{trainee_id}", trainee_id, now=0.0)

    sink = JsonLogSink(tmp_path / f"{tag}.jsonl")
    for tick in range(n_ticks):
        for trainee_id, obs in synthetic_tick(tick).items():
            registry.push_observation(trainee_id, obs, now=float(tick))
        records = rank_trainees(registry.tracks(), float(tick), scoring)
        sink.write(float(tick), records)
    sink.close()
    return (tmp_path / f"{tag}.jsonl").read_text(encoding="utf-8")


def test_two_replays_over_the_same_fixture_are_byte_identical(scoring, tmp_path):
    """Same input, same output -- including the timestamps."""
    first = _replay(scoring, 35, "run_a", tmp_path)
    second = _replay(scoring, 35, "run_b", tmp_path)
    assert first == second
    assert first.strip(), "the replay produced no records at all"


def test_the_replay_actually_ranks_trainees(scoring, tmp_path):
    """Guards the determinism test above from passing on two empty files."""
    payload = _replay(scoring, 35, "content", tmp_path)
    lines = [json.loads(line) for line in payload.strip().splitlines()]
    assert len(lines) == 35
    ranked = [line for line in lines if line["records"]]
    assert ranked, "no trainee was ever ranked"

    last = ranked[-1]["records"]
    scores = [r["score"] for r in last]
    assert scores == sorted(scores, reverse=True)
    assert len({r["trainee_id"] for r in last}) == len(last)


def test_record_serialisation_is_stable():
    from argus.triage import TriageRecord

    record = TriageRecord("t1", 0.6123, ("possible_fall", "form_error"), 2.0)
    once = json.dumps(to_json_dict(record), sort_keys=True)
    twice = json.dumps(to_json_dict(record), sort_keys=True)
    assert once == twice
    assert json.loads(once)["reason_codes"] == ["possible_fall", "form_error"]
