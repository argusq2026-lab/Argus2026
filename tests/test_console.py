"""The trainer console: its snapshot endpoint, and the page that reads it.

The console is the first thing in Argus that renders something richer than a
`TriageRecord`, so these tests are about two questions rather than one: does
it show what a trainer needs, and is the extra it can see still closed.

Staleness gets the most coverage here on purpose. A station that has gone
silent and a station whose trainee is calm produce the same empty reason list,
and this is the only surface in the system where that difference can be seen —
so "silent looks like calm" is the regression worth guarding hardest.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request

import pytest

from argus.console import CONSOLE_HTML
from argus.ingest.session import SessionRegistry
from argus.outputs import ConsoleSettings, StationView, TriageHTTPServer, station_to_json_dict
from argus.triage import TriageRecord
from tests.conftest import make_observation


@pytest.fixture
def server():
    srv = TriageHTTPServer(0, console=ConsoleSettings(stale_after_s=2.0, track_ttl_s=10.0))
    srv.start()
    yield srv
    srv.stop()


def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _view(trainee_id="t0", **overrides) -> StationView:
    base = dict(
        station_id="s0",
        trainee_id=trainee_id,
        connected=True,
        last_seen_ts=10.0,
        observations=30,
        bbox_xyxy=(0.1, 0.1, 0.5, 0.9),
        keypoints_xy=tuple([(0.3, 0.2)] * 17),
        keypoints_conf=tuple([0.9] * 17),
    )
    base.update(overrides)
    return StationView(**base)


# -- building views from live sessions ---------------------------------------


def test_station_view_carries_the_latest_observation(scoring):
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("station-3", "t0", now=0.0)
    registry.push_observation("t0", make_observation(ts=1.0), now=1.0)
    registry.push_observation("t0", make_observation(ts=2.0), now=2.0)

    [view] = registry.station_views()
    assert view.station_id == "station-3"
    assert view.trainee_id == "t0"
    assert view.connected is True
    assert view.last_seen_ts == 2.0
    assert view.observations == 2
    assert len(view.keypoints_xy) == 17
    assert len(view.keypoints_conf) == 17


def test_station_view_before_the_first_observation_has_no_pose(scoring):
    """A handshake with no frames must be visible as exactly that, not drawn
    as a trainee standing at the origin."""
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)

    [view] = registry.station_views()
    assert view.observations == 0
    assert view.bbox_xyxy is None
    assert view.keypoints_xy is None
    assert view.keypoints_conf is None
    assert view.form_ok is None


def test_station_view_carries_use_case_before_the_first_observation(scoring):
    """A welding station that has connected but sent nothing yet must not be
    mislabelled `"fitness"` -- `TrackState`'s own default -- until a frame
    happens to arrive; `register()` sets it at handshake."""
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0, use_case="welding")

    [view] = registry.station_views()
    assert view.use_case == "welding"
    assert view.observations == 0


def test_station_view_for_a_welding_station_has_no_pose(scoring):
    """A welding observation has no `bbox_xyxy`/`keypoints_xy` at all (see
    `argus.triage.FrameObservation`) — building its `StationView` must not
    crash trying to `tuple()` a `None`, and `exercise` must still read as the
    empty string its type promises, not `None`."""
    from argus.triage import FrameObservation

    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)
    registry.push_observation(
        "t0", FrameObservation(ts=1.0, use_case="welding", payload={"torch_angle_deg": 12.0}), now=1.0
    )

    [view] = registry.station_views()
    assert view.use_case == "welding"
    assert view.bbox_xyxy is None
    assert view.keypoints_xy is None
    assert view.keypoints_conf is None
    assert view.exercise == ""


def test_station_view_carries_the_phone_informational_fields(scoring):
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)
    obs = dataclasses.replace(
        make_observation(ts=1.0, form_reason_codes=("knee_valgus",)),
        exercise="squat",
        rep_count=12,
        form_ok=False,
    )
    registry.push_observation("t0", obs, now=1.0)

    [view] = registry.station_views()
    assert view.exercise == "squat"
    assert view.rep_count == 12
    assert view.form_ok is False
    assert view.form_reason_codes == ("knee_valgus",)


def test_a_disconnected_session_is_still_a_view(scoring):
    """The grace window is exactly when a trainee is most likely to be
    misread as fine, so it must not drop out of the console."""
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)
    registry.push_observation("t0", make_observation(ts=1.0), now=1.0)
    registry.mark_disconnected("t0")

    [view] = registry.station_views()
    assert view.connected is False
    assert view.last_seen_ts == 1.0


def test_views_are_ordered_by_trainee_id_not_by_recency(scoring):
    """A grid that reordered as scores moved would be unreadable — a trainer
    watching one station would lose it mid-glance."""
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    for trainee_id in ("charlie", "alice", "bob"):
        registry.register("s0", trainee_id, now=0.0)
    registry.push_observation("charlie", make_observation(ts=9.0), now=9.0)

    assert [v.trainee_id for v in registry.station_views()] == ["alice", "bob", "charlie"]


# -- the snapshot endpoint ----------------------------------------------------


def test_console_endpoint_starts_empty(server):
    payload = _get(server.port, "/console")
    assert payload["records"] == []
    assert payload["stations"] == []
    assert payload["ts"] == 0.0


def test_console_endpoint_serves_records_and_stations(server):
    server.update(5.0, [TriageRecord("t0", 0.9, ("possible_fall",), 5.0)], [_view()])
    payload = _get(server.port, "/console")

    assert payload["ts"] == 5.0
    assert payload["rank_ts"] == 5.0
    assert payload["records"][0]["trainee_id"] == "t0"
    assert payload["stations"][0]["trainee_id"] == "t0"
    assert len(payload["stations"][0]["keypoints_xy"]) == 17


def test_console_endpoint_serves_the_config_the_page_renders_against(server):
    """The page must not hardcode a cadence or a threshold; it reads them."""
    cfg = _get(server.port, "/console")["config"]
    assert cfg["poll_interval_ms"] > 0
    assert cfg["stale_after_s"] == 2.0
    assert cfg["track_ttl_s"] == 10.0
    for key in ("keypoint_conf_threshold", "alert_threshold", "history_len"):
        assert key in cfg


def test_updating_stations_does_not_claim_the_rank_was_recomputed(server):
    """Stations refresh per observation; the rank only on its own interval.
    `/triage`'s `ts` means "when the rank was computed" and must keep meaning
    that, or a consumer reads a stale rank as a fresh one."""
    server.update(5.0, [TriageRecord("t0", 0.9, (), 5.0)], [_view()])
    server.update_stations(7.5, [_view(last_seen_ts=7.5)])

    assert _get(server.port, "/triage")["ts"] == 5.0
    console = _get(server.port, "/console")
    assert console["ts"] == 7.5
    assert console["rank_ts"] == 5.0
    assert console["records"][0]["trainee_id"] == "t0"


def test_staleness_is_computable_from_one_snapshot(server):
    """Age is `ts - last_seen_ts`, both on the server's clock. The page never
    needs the phone's clock, which is not synchronised with anything."""
    server.update_stations(12.0, [_view(last_seen_ts=9.5)])
    payload = _get(server.port, "/console")
    assert payload["ts"] - payload["stations"][0]["last_seen_ts"] == pytest.approx(2.5)


def test_console_snapshot_is_a_copy(server):
    stations = [_view()]
    server.update_stations(1.0, stations)
    stations.append(_view("t1"))
    assert len(_get(server.port, "/console")["stations"]) == 1


def test_triage_endpoint_is_unchanged_by_the_console(server):
    """The console reads a wider view; the alert boundary did not move."""
    server.update(5.0, [TriageRecord("t0", 0.9, ("possible_fall",), 5.0)], [_view()])
    payload = _get(server.port, "/triage")
    assert set(payload) == {"ts", "records"}
    assert set(payload["records"][0]) == {"trainee_id", "score", "reason_codes", "ts"}


# -- serialisation ------------------------------------------------------------


def test_station_serialises_tuples_as_lists():
    payload = station_to_json_dict(_view(form_reason_codes=("knee_valgus",)))
    assert payload["form_reason_codes"] == ["knee_valgus"]
    assert payload["bbox_xyxy"] == [0.1, 0.1, 0.5, 0.9]
    assert payload["keypoints_xy"][0] == [0.3, 0.2]
    assert json.dumps(payload)  # round-trips without a custom encoder


def test_absent_pose_serialises_as_null_not_as_an_empty_pose():
    payload = station_to_json_dict(
        _view(bbox_xyxy=None, keypoints_xy=None, keypoints_conf=None, observations=0)
    )
    assert payload["bbox_xyxy"] is None
    assert payload["keypoints_xy"] is None
    assert payload["keypoints_conf"] is None


# -- the page itself ----------------------------------------------------------


def test_the_page_is_served_at_root(server):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/", timeout=5) as response:
        assert response.status == 200
        assert "text/html" in response.headers["Content-Type"]
        body = response.read().decode("utf-8")
    assert body == CONSOLE_HTML


def test_the_page_reads_only_the_console_endpoint():
    """One atomic read. Two endpoints polled separately would let a skeleton
    from one instant sit beside a score from another."""
    assert 'fetch("/console"' in CONSOLE_HTML
    assert 'fetch("/triage"' not in CONSOLE_HTML


def test_the_page_never_assigns_innerhtml():
    """Structural, like tests/test_privacy.py: `trainee_id`, `station_id`, and
    `exercise` are all chosen by a phone. A page that cannot write markup
    cannot be made to render a phone's string as markup — that is a property
    of the wiring, not of remembering to escape at each call site."""
    assert "innerHTML" not in CONSOLE_HTML


def test_the_page_draws_a_skeleton():
    assert "<canvas" in CONSOLE_HTML or "createElement(\"canvas\")" in CONSOLE_HTML
    assert "keypoints_xy" in CONSOLE_HTML


def test_the_page_gates_keypoints_on_the_scorer_threshold():
    """A console that drew joints the scorer ignored would show a trainer a
    pose the rank was not computed from."""
    assert "keypoint_conf_threshold" in CONSOLE_HTML


def test_the_page_says_the_rank_is_unvalidated():
    """The weights have never been fitted to an incident (docs/VALIDATION.md).
    A console that presented the order as tuned would be overclaiming, so the
    caveat is on screen rather than only in the docs."""
    assert "not validated" in CONSOLE_HTML
    assert "VALIDATION.md" in CONSOLE_HTML


def test_the_page_distinguishes_silence_from_calm():
    assert "stale_after_s" in CONSOLE_HTML
    assert "Not reporting" in CONSOLE_HTML


def test_the_page_renders_reason_codes_as_prose():
    """`possible_fall` is a value, not a sentence a trainer should read."""
    assert "possible_fall: " in CONSOLE_HTML
    assert "Possible fall" in CONSOLE_HTML


def test_the_page_surfaces_the_phone_informational_fields():
    for field in ("exercise", "rep_count", "form_ok"):
        assert field in CONSOLE_HTML


def test_the_page_loads_nothing_from_the_network():
    """No CDN, no font host: the console has to work on a gym's Wi-Fi, and on
    a laptop that is deliberately not routing anywhere."""
    for scheme in ("http://", "https://", "//cdn"):
        assert scheme not in CONSOLE_HTML


# -- join approval from the console -------------------------------------------


def _pending(request_id="join-1", **overrides):
    from argus.outputs import PendingJoinView

    base = dict(
        request_id=request_id,
        station_id="rack-3",
        trainee_id="t0",
        display_name="Alex",
        requested_ts=10.0,
        expires_ts=130.0,
    )
    base.update(overrides)
    return PendingJoinView(**base)


def _post(port: int, path: str, body: dict) -> tuple[int, dict]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture
def approving_server():
    """A server with an admission queue attached, recording every decision."""
    decisions = []

    def decide(request_id: str, approve: bool) -> bool:
        decisions.append((request_id, approve))
        return request_id == "join-1"

    srv = TriageHTTPServer(0, console=ConsoleSettings(), on_join_decision=decide)
    srv.start()
    srv.decisions = decisions
    yield srv
    srv.stop()


def test_pending_joins_are_served_to_the_console(approving_server):
    approving_server.update_stations(20.0, [], [_pending()])
    payload = _get(approving_server.port, "/console")
    assert payload["pending"][0]["display_name"] == "Alex"
    assert payload["pending"][0]["request_id"] == "join-1"
    # Countdown is computable from one snapshot, on the server's clock.
    assert payload["pending"][0]["expires_ts"] - payload["ts"] == pytest.approx(110.0)


def test_the_console_can_approve_a_join(approving_server):
    status, body = _post(approving_server.port, "/join/decide",
                         {"request_id": "join-1", "approve": True})
    assert status == 200
    assert body == {"settled": True}
    assert approving_server.decisions == [("join-1", True)]


def test_the_console_can_decline_a_join(approving_server):
    status, _ = _post(approving_server.port, "/join/decide",
                      {"request_id": "join-1", "approve": False})
    assert status == 200
    assert approving_server.decisions == [("join-1", False)]


def test_deciding_a_request_that_is_no_longer_waiting_says_so(approving_server):
    """The phone may have hung up or timed out between the page drawing the
    button and someone pressing it. The console has to be able to tell that
    apart from a decision that landed."""
    status, body = _post(approving_server.port, "/join/decide",
                         {"request_id": "join-gone", "approve": True})
    assert status == 409
    assert body["settled"] is False


@pytest.mark.parametrize(
    "body",
    [{}, {"request_id": "join-1"}, {"approve": True}, {"request_id": 1, "approve": True},
     {"request_id": "join-1", "approve": "yes"}],
    ids=["empty", "no-approve", "no-id", "id-not-a-string", "approve-not-a-bool"],
)
def test_a_malformed_decision_is_rejected_rather_than_guessed(approving_server, body):
    status, _ = _post(approving_server.port, "/join/decide", body)
    assert status == 400
    assert approving_server.decisions == []


def test_no_other_post_route_exists(approving_server):
    status, _ = _post(approving_server.port, "/frames", {})
    assert status == 404


# -- POST /session/use_case ---------------------------------------------------


@pytest.fixture
def use_case_server():
    """A server with a use-case-change callback attached, recording calls."""
    changes = []

    def change(use_case: str):
        changes.append(use_case)
        if use_case == "welding":
            return True, ""
        return False, f"use_case {use_case!r} is not implemented"

    srv = TriageHTTPServer(
        0,
        console=ConsoleSettings(use_case="fitness", known_use_cases=("fitness", "welding")),
        on_use_case_change=change,
    )
    srv.start()
    srv.changes = changes
    yield srv
    srv.stop()


def test_the_console_can_change_the_use_case(use_case_server):
    status, body = _post(use_case_server.port, "/session/use_case", {"use_case": "welding"})
    assert status == 200
    assert body == {"ok": True, "use_case": "welding"}
    assert use_case_server.changes == ["welding"]


def test_changing_to_an_unimplemented_use_case_is_rejected(use_case_server):
    status, body = _post(use_case_server.port, "/session/use_case", {"use_case": "lab"})
    assert status == 400
    assert body["ok"] is False
    assert "lab" in body["error"]


@pytest.mark.parametrize(
    "body", [{}, {"use_case": 7}], ids=["empty", "use_case-not-a-string"]
)
def test_a_malformed_use_case_request_is_rejected(use_case_server, body):
    status, _ = _post(use_case_server.port, "/session/use_case", body)
    assert status == 400
    assert use_case_server.changes == []


def test_a_server_without_a_use_case_callback_refuses_changes(server):
    status, body = _post(server.port, "/session/use_case", {"use_case": "welding"})
    assert status == 503
    assert "error" in body


def test_set_use_case_updates_what_the_console_reports(use_case_server):
    """The dropdown's own effect: `GET /console`'s `config.use_case` reflects
    the change immediately, not just the callback's side effect."""
    use_case_server.set_use_case("welding")
    payload = _get(use_case_server.port, "/console")
    assert payload["config"]["use_case"] == "welding"


def test_a_server_without_an_admission_queue_refuses_decisions(server):
    status, body = _post(server.port, "/join/decide", {"request_id": "join-1", "approve": True})
    assert status == 503
    assert "admission" in body["error"]


def test_the_page_offers_approve_and_decline():
    assert "Approve" in CONSOLE_HTML
    assert "Decline" in CONSOLE_HTML
    assert '"/join/decide"' in CONSOLE_HTML


def test_the_page_shows_the_trainee_id_being_admitted():
    """A display name is phone-chosen and can be anything; the trainee id is
    what an alert is actually dispatched against, so an instructor approving
    should see it."""
    assert "trainee " in CONSOLE_HTML
    assert "display_name" in CONSOLE_HTML


# -- scoring profiles ---------------------------------------------------------
#
# A profile that zeroes a feature suppresses both its contribution and its
# reason code. That is correct, and it means a plank card reading "nothing
# flagged" looks identical to one where fall and stillness had actually been
# checked. The console is the only place that difference can be shown, so the
# weights have to reach it.


def test_the_shipped_plank_profile_reaches_the_console(default_config):
    """Read from the real config, not a hand-built one: what matters is that
    the profile an operator actually ships is what the page renders against."""
    scoring = default_config.scoring
    srv = TriageHTTPServer(
        0,
        console=ConsoleSettings(
            default_weights=dict(scoring.weights),
            exercise_weights={k: dict(v) for k, v in scoring.exercise_weights.items()},
        ),
    )
    srv.start()
    try:
        served = _get(srv.port, "/console")["config"]
        assert served["default_weights"]["fall"] > 0
        # The two the plank profile switches off, and the one it leans on.
        assert served["exercise_weights"]["plank"]["fall"] == 0.0
        assert served["exercise_weights"]["plank"]["stillness"] == 0.0
        assert served["exercise_weights"]["plank"]["form_error"] > 0
    finally:
        srv.stop()


def test_a_running_server_publishes_the_configured_profiles(default_config):
    """The wiring, not just the dataclass: a `ConsoleSettings` built by hand
    in a test proves nothing about what `argus run` actually publishes."""
    import asyncio
    import socket

    from argus.ingest.server import IngestServer

    with socket.socket() as probe:          # a port that is free right now
        probe.bind(("127.0.0.1", 0))
        http_port = probe.getsockname()[1]

    cfg = dataclasses.replace(
        default_config,
        ingest=dataclasses.replace(default_config.ingest, ws_host="127.0.0.1", ws_port=0),
        outputs=dataclasses.replace(
            default_config.outputs, console=False, http_port=http_port
        ),
        discovery=dataclasses.replace(default_config.discovery, enabled=False),
    )

    async def run():
        server = IngestServer(cfg)
        await server.start()
        try:
            served = _get(server.http_port, "/console")["config"]
            assert served["exercise_weights"]["plank"]["fall"] == 0.0
            assert served["default_weights"] == dict(cfg.scoring.weights)
        finally:
            await server.stop()

    asyncio.run(run())


def test_the_page_can_name_which_checks_a_profile_switches_off():
    """`fall` weighted 0 for planks means a trainee who collapses mid-plank
    raises nothing. Correct, and unacceptable to leave unsaid."""
    assert "notWatchedFor" in CONSOLE_HTML
    assert "not watching for" in CONSOLE_HTML
    assert "exercise_weights" in CONSOLE_HTML
    assert "default_weights" in CONSOLE_HTML


def test_the_page_resolves_a_profile_the_same_way_the_scorer_does():
    """An exercise with no profile falls back to the defaults, not to zero."""
    assert "profileFor" in CONSOLE_HTML
    assert "toLowerCase" in CONSOLE_HTML


def test_the_page_names_the_plank_form_codes_in_prose():
    """`hips_piked` is jargon; a trainer should read a sentence."""
    assert "hips_sagging: " in CONSOLE_HTML
    assert "hips_piked: " in CONSOLE_HTML
    assert "Hips sagging" in CONSOLE_HTML


# -- the rolling session view -------------------------------------------------


def test_the_session_summary_reaches_the_console(scoring):
    """The instant score is what fires alerts; this is what a human reads."""
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)
    for i in range(6):
        obs = dataclasses.replace(
            make_observation(ts=i * 0.1, form_reason_codes=("knee_valgus",) if i == 2 else ()),
            rep_count=i,
        )
        registry.push_observation("t0", obs, now=i * 0.1)
    registry.tracks()["t0"].session.observe_score(0.4, ts=1.0, half_life_s=20.0)

    [view] = registry.station_views()
    assert view.session is not None
    assert view.session.reps == 5
    assert view.session.reps_flagged == 1
    assert view.session.rolling_score == pytest.approx(0.4)
    assert view.session.peak_score == pytest.approx(0.4)
    assert view.session.code_counts == {"knee_valgus": 1}


def test_a_held_exercise_reports_seconds_not_reps(scoring):
    """A plank has no reps, so its work is time -- the console has to be able
    to say "18s of 2m10s flagged" as readily as "3 of 14 reps"."""
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)
    ts = 0.0
    while ts < 12.0:
        registry.push_observation("t0", make_observation(ts=ts), now=ts)
        ts += 0.2

    [view] = registry.station_views()
    assert view.session.reps == 0
    assert view.session.hold_s > 10.0
    assert view.session.fault_rate == pytest.approx(0.0)


def test_a_station_with_no_frames_has_no_session(scoring):
    registry = SessionRegistry(scoring, track_ttl_s=10.0)
    registry.register("s0", "t0", now=0.0)
    assert registry.station_views()[0].session is None


def test_the_session_serialises_into_the_snapshot(server):
    from argus.outputs import SessionSummary

    server.update_stations(1.0, [_view(session=SessionSummary(
        rolling_score=0.31, peak_score=0.62, active_s=95.0,
        reps=14, reps_flagged=3, hold_s=0.0, hold_flagged_s=0.0,
        fault_rate=3 / 14, code_counts={"knee_valgus": 22},
    ))])
    served = _get(server.port, "/console")["stations"][0]["session"]
    assert served["rolling_score"] == 0.31
    assert served["peak_score"] == 0.62
    assert served["reps_flagged"] == 3
    assert served["code_counts"] == {"knee_valgus": 22}


def test_the_page_reads_the_rolling_score_not_the_instant_one():
    """The whole point: an instructor watching a number that moves every tick
    cannot compare it with the same trainee a minute ago."""
    assert "rolling_score" in CONSOLE_HTML
    assert "session avg" in CONSOLE_HTML


def test_the_page_still_colours_on_the_instant_score():
    """A fall must colour the card on the frame it happens, not once a mean
    catches up. If this ever reads `rolling` the smoothing has eaten the
    alert."""
    assert "score >= state.cfg.alert_threshold" in CONSOLE_HTML


def test_the_page_shows_the_peak_the_mean_forgot():
    assert "peak_score" in CONSOLE_HTML
    assert "peak " in CONSOLE_HTML


def test_the_page_expresses_work_in_reps_or_seconds():
    assert "reps flagged" in CONSOLE_HTML
    assert "hold_s" in CONSOLE_HTML
    assert "fault rate" in CONSOLE_HTML


def test_the_page_does_not_show_a_rate_it_was_not_given():
    """`fault_rate` is null until enough work is seen, and null must stay
    visible as "not enough to say" rather than becoming a reassuring 0%."""
    assert "too little work to call a rate yet" in CONSOLE_HTML


def test_the_page_names_persistent_form_failure_as_its_own_thing():
    """"This rep was wrong" and "they cannot hold the movement" ask different
    things of an instructor -- a cue versus stop-and-reset -- so they must not
    read as the same sentence."""
    assert "persistent_form_fault: " in CONSOLE_HTML
    assert "Cannot hold form" in CONSOLE_HTML
