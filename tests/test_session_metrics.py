"""The rolling, session-long account behind the instantaneous score.

The instant score answers "who needs a human right now" and is what fires
alerts. It is also nearly unreadable: computed off a ~2 s window, recomputed
every rank tick, moving constantly. `SessionMetrics` is the other half — what
a trainee has actually *done* — and these tests are mostly about the two ways
that account can lie:

* **Counting work that did not happen.** A reconnect gap credited as hold
  time, or a phone's rep counter resetting and being read as a huge jump.
* **Reporting a rate on no evidence.** One bad rep out of one is not a 100%
  fault rate, and a trainee should not reach the top of a queue on one frame.
"""

from __future__ import annotations

import dataclasses

import pytest

from argus.triage import SessionMetrics, TrackState
from tests.conftest import make_observation


def _obs(ts, *, reps=None, codes=()):
    return dataclasses.replace(
        make_observation(ts=ts, form_reason_codes=codes), rep_count=reps
    )


# -- work in reps -------------------------------------------------------------


def test_reps_accumulate_from_the_phones_counter(scoring):
    m = SessionMetrics()
    for tick, reps in enumerate([0, 1, 2, 3]):
        m.observe(_obs(tick * 0.1, reps=reps), scoring)
    assert m.reps == 3


def test_the_first_observation_does_not_credit_reps_already_done(scoring):
    """A phone that joins mid-set reports rep 7. Argus did not see reps 1-6
    and must not claim to have."""
    m = SessionMetrics()
    m.observe(_obs(0.0, reps=7), scoring)
    assert m.reps == 0
    m.observe(_obs(0.1, reps=8), scoring)
    assert m.reps == 1


def test_a_counter_reset_is_a_new_set_not_a_negative_rep(scoring):
    m = SessionMetrics()
    for ts, reps in ((0.0, 8), (0.1, 9), (0.2, 0), (0.3, 1)):
        m.observe(_obs(ts, reps=reps), scoring)
    # 8->9 is one rep, 9->0 starts a set, 0->1 is one more.
    assert m.reps == 2


def test_a_rep_is_flagged_if_any_frame_in_it_carried_a_code(scoring):
    """Form faults happen mid-rep. Judging only the frame where the counter
    ticks would miss nearly all of them."""
    m = SessionMetrics()
    m.observe(_obs(0.0, reps=0), scoring)
    m.observe(_obs(0.1, reps=0, codes=("knee_valgus",)), scoring)
    m.observe(_obs(0.2, reps=0), scoring)
    m.observe(_obs(0.3, reps=1), scoring)          # rep completes
    assert (m.reps, m.reps_flagged) == (1, 1)


def test_a_clean_rep_after_a_flagged_one_is_not_also_flagged(scoring):
    m = SessionMetrics()
    m.observe(_obs(0.0, reps=0), scoring)
    m.observe(_obs(0.1, reps=0, codes=("knee_valgus",)), scoring)
    m.observe(_obs(0.2, reps=1), scoring)
    m.observe(_obs(0.3, reps=2), scoring)
    assert (m.reps, m.reps_flagged) == (2, 1)


def test_reps_missed_between_frames_are_counted_but_not_accused(scoring):
    """If three reps land between two frames we saw, all three happened —
    but we only witnessed a fault in one of them."""
    m = SessionMetrics()
    m.observe(_obs(0.0, reps=0), scoring)
    m.observe(_obs(0.1, reps=0, codes=("knee_valgus",)), scoring)
    m.observe(_obs(0.2, reps=3), scoring)
    assert (m.reps, m.reps_flagged) == (3, 1)


# -- work in seconds ----------------------------------------------------------


def test_a_held_exercise_accumulates_time_not_reps(scoring):
    """A plank has no reps. Its entire quality is how long it was held well,
    so the same `fault_rate` has to work in seconds."""
    m = SessionMetrics()
    for i in range(11):
        m.observe(_obs(i * 0.1), scoring)          # rep_count None
    assert m.reps == 0
    assert m.hold_s == pytest.approx(1.0)
    assert m.hold_flagged_s == 0.0


def test_flagged_hold_time_is_the_time_the_code_was_present(scoring):
    m = SessionMetrics()
    m.observe(_obs(0.0), scoring)
    m.observe(_obs(0.5, codes=("hips_sagging",)), scoring)
    m.observe(_obs(1.0, codes=("hips_sagging",)), scoring)
    m.observe(_obs(1.5), scoring)
    assert m.hold_s == pytest.approx(1.5)
    assert m.hold_flagged_s == pytest.approx(1.0)


def test_a_reconnect_gap_is_not_counted_as_work(scoring):
    """Otherwise a station that was away for three minutes comes back having
    "held" them, and the number an instructor trusts is fiction."""
    m = SessionMetrics()
    m.observe(_obs(0.0), scoring)
    m.observe(_obs(0.1), scoring)
    m.observe(_obs(90.0), scoring)                 # gone and back
    m.observe(_obs(90.1), scoring)
    assert m.hold_s == pytest.approx(0.2)
    assert m.active_s == pytest.approx(0.2)


def test_a_backwards_timestamp_is_ignored_rather_than_subtracted(scoring):
    m = SessionMetrics()
    m.observe(_obs(5.0), scoring)
    m.observe(_obs(4.0), scoring)
    assert m.active_s == 0.0


# -- the fault rate -----------------------------------------------------------


def test_no_rate_is_reported_on_too_little_work(scoring):
    """`None`, not 0.0 and not 1.0: "no faults yet" and "not enough evidence"
    are different things to show someone choosing who to walk over to."""
    m = SessionMetrics()
    m.observe(_obs(0.0, reps=0), scoring)
    m.observe(_obs(0.1, reps=0, codes=("knee_valgus",)), scoring)
    m.observe(_obs(0.2, reps=1), scoring)
    assert m.reps_flagged == 1
    assert m.fault_rate(scoring) is None


def test_a_rate_appears_once_enough_reps_are_done(scoring):
    m = SessionMetrics()
    m.observe(_obs(0.0, reps=0), scoring)
    for rep in range(1, 5):
        if rep == 2:
            m.observe(_obs(rep * 0.1 - 0.05, reps=rep - 1, codes=("knee_valgus",)), scoring)
        m.observe(_obs(rep * 0.1, reps=rep), scoring)
    assert m.reps >= scoring.min_reps_for_fault_rate
    assert m.fault_rate(scoring) == pytest.approx(m.reps_flagged / m.reps)


def test_a_held_exercise_earns_a_rate_in_seconds(scoring):
    m = SessionMetrics()
    ts = 0.0
    while ts < scoring.min_hold_s_for_fault_rate + 1:
        m.observe(_obs(ts, codes=("hips_sagging",) if ts > 5 else ()), scoring)
        ts += 0.2
    rate = m.fault_rate(scoring)
    assert rate is not None
    assert 0.0 < rate < 1.0


# -- the rolling score --------------------------------------------------------


def test_the_rolling_score_starts_at_the_first_reading(scoring):
    """No warm-up ramp from zero: a trainee already in trouble when the
    console opens should not read as calm for the first half-life."""
    m = SessionMetrics()
    m.observe_score(0.8, ts=0.0, half_life_s=20.0)
    assert m.rolling_score == pytest.approx(0.8)


def test_the_rolling_score_moves_by_half_a_half_life(scoring):
    m = SessionMetrics()
    m.observe_score(0.0, ts=0.0, half_life_s=10.0)
    m.observe_score(1.0, ts=10.0, half_life_s=10.0)
    assert m.rolling_score == pytest.approx(0.5)
    m.observe_score(1.0, ts=20.0, half_life_s=10.0)
    assert m.rolling_score == pytest.approx(0.75)


def test_the_rolling_score_decays_by_elapsed_time_not_by_tick_count(scoring):
    """So the number means the same thing whether ticks arrive every 0.5 s or
    every 2 s -- otherwise retuning `rank_interval_s` silently retunes this."""
    fast = SessionMetrics()
    fast.observe_score(1.0, ts=0.0, half_life_s=10.0)
    for i in range(1, 21):
        fast.observe_score(0.0, ts=i * 0.5, half_life_s=10.0)

    slow = SessionMetrics()
    slow.observe_score(1.0, ts=0.0, half_life_s=10.0)
    for i in range(1, 6):
        slow.observe_score(0.0, ts=i * 2.0, half_life_s=10.0)

    assert fast.rolling_score == pytest.approx(slow.rolling_score, abs=1e-9)


def test_the_peak_survives_what_the_mean_forgets(scoring):
    """A rolling mean is supposed to forget. "This trainee was briefly in real
    trouble" is not something an instructor should have to catch live."""
    m = SessionMetrics()
    m.observe_score(0.9, ts=0.0, half_life_s=1.0)
    for i in range(1, 60):
        m.observe_score(0.0, ts=float(i), half_life_s=1.0)
    assert m.rolling_score < 0.01
    assert m.peak_score == pytest.approx(0.9)


# -- wiring -------------------------------------------------------------------


def test_pushing_an_observation_updates_the_session(scoring):
    track = TrackState(history_len=scoring.history_len)
    track.push(_obs(0.0, reps=0), scoring)
    track.push(_obs(0.1, reps=1, codes=("knee_valgus",)), scoring)
    assert track.session.frames == 2
    assert track.session.reps == 1
    assert track.session.code_counts == {"knee_valgus": 1}


def test_the_scorer_stays_a_pure_function_of_history(scoring):
    """The rolling score is advanced by the rank tick, never by
    `compute_triage` -- the scorer's purity is what tests/test_determinism.py
    protects, and session state must not creep into it."""
    import inspect

    from argus import triage

    source = inspect.getsource(triage.compute_triage)
    assert "session" not in source
    assert "rolling" not in source
