"""The VLM prefilter gate — the prototype's most consequential defect.

Its gate summed a hardcoded literal tuple, so it was 0.75 for every trainee on
every frame and the VLM would have been invoked for all of them. The top-K
bound was applied to the alert print, never to the sampling. Each test below
pins one property that was not true before.
"""

from __future__ import annotations

import dataclasses

import pytest

from argus.config import VLMConfig
from argus.pipeline.prefilter import select_for_vlm
from argus.triage import TriageRecord


@pytest.fixture
def vlm_cfg(default_config) -> VLMConfig:
    return default_config.vlm


def rec(trainee_id: str, score: float) -> TriageRecord:
    return TriageRecord(trainee_id, score, (), 0.0)


def test_unflagged_trainees_are_not_sampled(vlm_cfg):
    """The bug: every tracked trainee was sampled, flagged or not."""
    below = vlm_cfg.prefilter_score_threshold - 0.01
    assert select_for_vlm([rec("t0", below), rec("t1", 0.0)], {}, 10.0, vlm_cfg) == []


def test_flagged_trainees_are_sampled(vlm_cfg):
    selected = select_for_vlm([rec("t0", vlm_cfg.prefilter_score_threshold)], {}, 10.0, vlm_cfg)
    assert selected == ["t0"]


def test_top_k_bounds_calls_per_frame(vlm_cfg):
    """The bound existed but was never applied to sampling."""
    records = [rec(f"t{i}", 0.9) for i in range(20)]
    assert len(select_for_vlm(records, {}, 10.0, vlm_cfg)) == vlm_cfg.top_k


def test_the_highest_scoring_trainees_win_the_scarce_slots(vlm_cfg):
    records = [rec("low", 0.4), rec("highest", 0.95), rec("middle", 0.7)]
    cfg = dataclasses.replace(vlm_cfg, top_k=2)
    assert select_for_vlm(records, {}, 10.0, cfg) == ["highest", "middle"]


def test_ties_break_on_trainee_id_like_the_rank(vlm_cfg):
    records = [rec("t_c", 0.9), rec("t_a", 0.9), rec("t_b", 0.9)]
    cfg = dataclasses.replace(vlm_cfg, top_k=2)
    assert select_for_vlm(records, {}, 10.0, cfg) == ["t_a", "t_b"]


def test_cadence_is_enforced_per_trainee(vlm_cfg):
    records = [rec("t0", 0.9)]
    last = {"t0": 10.0}
    just_after = 10.0 + vlm_cfg.sample_interval_s / 2
    assert select_for_vlm(records, last, just_after, vlm_cfg) == []
    due = 10.0 + vlm_cfg.sample_interval_s
    assert select_for_vlm(records, last, due, vlm_cfg) == ["t0"]


def test_cadence_is_independent_across_trainees(vlm_cfg):
    records = [rec("t0", 0.9), rec("t1", 0.9)]
    last = {"t0": 10.0}
    assert select_for_vlm(records, last, 10.1, vlm_cfg) == ["t1"]


def test_top_k_zero_disables_the_vlm_entirely(vlm_cfg):
    cfg = dataclasses.replace(vlm_cfg, top_k=0)
    assert select_for_vlm([rec("t0", 1.0)], {}, 10.0, cfg) == []


def test_a_departed_trainee_is_not_sampled(vlm_cfg):
    records = [rec("gone", 0.9), rec("present", 0.9)]
    selected = select_for_vlm(records, {}, 10.0, vlm_cfg, eligible_ids={"present"})
    assert selected == ["present"]


def test_gate_is_a_pure_function(vlm_cfg):
    records = [rec(f"t{i}", 0.5 + i / 100) for i in range(10)]
    last = {"t3": 9.5}
    first = select_for_vlm(records, last, 10.0, vlm_cfg)
    second = select_for_vlm(records, last, 10.0, vlm_cfg)
    assert first == second
    assert last == {"t3": 9.5}, "the gate must not mutate its inputs"
