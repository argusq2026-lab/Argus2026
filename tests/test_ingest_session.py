"""Per-station session bookkeeping: identity, reconnect grace, and TTL eviction."""

from __future__ import annotations

import pytest

from argus.ingest.session import DuplicateTraineeError, SessionRegistry
from tests.conftest import make_observation


@pytest.fixture
def registry(scoring) -> SessionRegistry:
    return SessionRegistry(scoring, track_ttl_s=10.0)


def test_register_creates_a_fresh_session(registry):
    registry.register("station-a", "t0", now=0.0)
    assert "t0" in registry
    assert len(registry) == 1


def test_a_second_live_connection_for_the_same_trainee_is_rejected(registry):
    registry.register("station-a", "t0", now=0.0)
    with pytest.raises(DuplicateTraineeError):
        registry.register("station-b", "t0", now=1.0)


def test_reconnecting_after_disconnect_resumes_the_same_track(registry, scoring):
    registry.register("station-a", "t0", now=0.0)
    registry.push_observation("t0", make_observation(ts=0.0), now=0.0)
    registry.mark_disconnected("t0")

    # still in its grace window -- history must survive
    session = registry.register("station-a", "t0", now=1.0)
    assert len(session.track.history) == 1


def test_disconnect_then_reconnect_is_not_a_duplicate(registry):
    registry.register("station-a", "t0", now=0.0)
    registry.mark_disconnected("t0")
    registry.register("station-a", "t0", now=1.0)  # must not raise


def test_push_observation_updates_last_seen(registry):
    registry.register("station-a", "t0", now=0.0)
    registry.push_observation("t0", make_observation(ts=5.0), now=5.0)
    assert registry.expire_stale(now=5.0 + 9.9) == []
    assert registry.expire_stale(now=5.0 + 10.1) == ["t0"]


def test_expire_stale_drops_only_silent_sessions(registry):
    registry.register("station-a", "old", now=0.0)
    registry.register("station-b", "fresh", now=100.0)
    expired = registry.expire_stale(now=100.0)
    assert expired == ["old"]
    assert "old" not in registry
    assert "fresh" in registry


def test_expire_stale_evicts_even_a_still_connected_but_silent_session(registry):
    """A hung connection that never sends another observation is presumed
    gone too, same as one that cleanly disconnected."""
    registry.register("station-a", "t0", now=0.0)
    assert registry.expire_stale(now=11.0) == ["t0"]
    assert "t0" not in registry


def test_tracks_returns_every_registered_session_connected_or_not(registry):
    registry.register("station-a", "t0", now=0.0)
    registry.register("station-b", "t1", now=0.0)
    registry.mark_disconnected("t0")
    assert set(registry.tracks()) == {"t0", "t1"}
