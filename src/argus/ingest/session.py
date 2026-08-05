"""Per-station session bookkeeping.

One phone in front of one trainee is one `StationSession`, keyed by
`trainee_id` — the triage key an instructor is actually dispatched against.
There is no re-identification problem here the way there was for a laptop
camera watching many trainees at once: a phone's own connection *is* the
identity, so this registry's whole job is bookkeeping — who is currently
connected, and who has gone silent long enough to be presumed gone — not
re-association.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.config import ScoringConfig
from argus.outputs import StationView
from argus.triage import FrameObservation, TrackState


@dataclass
class StationSession:
    """Everything kept for one trainee. No pixels, no free text — `track`
    holds only the numeric history `argus.triage` already scores.

    `connected` tracks the live socket separately from the session's
    existence: a dropped connection does not erase `track` immediately, so a
    phone that reconnects within `track_ttl_s` (a lift Wi-Fi blip, an app
    restart) resumes the same rolling history and alert-suppression state
    instead of starting the trainee over from zero.
    """

    station_id: str
    trainee_id: str
    track: TrackState
    last_seen_ts: float
    connected: bool = True


class DuplicateTraineeError(ValueError):
    """A second *live* connection claimed a `trainee_id` already connected.

    Silently replacing the first connection would let a misconfigured (or
    malicious) phone impersonate another trainee's track and inherit their
    alert-suppression state — the network-era analogue of the old tracker's
    identity-swap concern. A `trainee_id` that is registered but currently
    disconnected is not live, so reconnecting to it is not a collision.
    """


class SessionRegistry:
    """All sessions, keyed by `trainee_id` — connected or in their grace window."""

    def __init__(self, scoring: ScoringConfig, track_ttl_s: float):
        self._scoring = scoring
        self._ttl = track_ttl_s
        self._sessions: dict[str, StationSession] = {}

    def __contains__(self, trainee_id: str) -> bool:
        return trainee_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def is_connected(self, trainee_id: str) -> bool:
        """Whether a *live* socket already claims this id.

        Distinct from `in`: a session inside its disconnect grace window is
        registered but not connected, and reconnecting to it is the normal
        case rather than a collision.
        """
        session = self._sessions.get(trainee_id)
        return session is not None and session.connected

    def register(self, station_id: str, trainee_id: str, now: float) -> StationSession:
        """Start a new session, or resume one within its disconnect grace window."""
        existing = self._sessions.get(trainee_id)
        if existing is not None:
            if existing.connected:
                raise DuplicateTraineeError(f"trainee_id {trainee_id!r} is already connected")
            existing.station_id = station_id
            existing.connected = True
            existing.last_seen_ts = now
            return existing

        session = StationSession(
            station_id=station_id,
            trainee_id=trainee_id,
            track=TrackState(history_len=self._scoring.history_len),
            last_seen_ts=now,
        )
        self._sessions[trainee_id] = session
        return session

    def mark_disconnected(self, trainee_id: str) -> None:
        """The socket closed; keep the track until `expire_stale` evicts it."""
        session = self._sessions.get(trainee_id)
        if session is not None:
            session.connected = False

    def push_observation(self, trainee_id: str, obs: FrameObservation, now: float) -> None:
        """Raises `KeyError` if the session was already expired by `expire_stale`
        (silent this long that the ttl elapsed while a connection lingered) —
        the caller closes the socket and tells the phone to reconnect fresh."""
        session = self._sessions[trainee_id]
        session.track.push(obs, self._scoring)
        session.last_seen_ts = now

    def tracks(self) -> dict[str, TrackState]:
        """Every session's history, connected or still in its grace window."""
        return {trainee_id: s.track for trainee_id, s in self._sessions.items()}

    def station_views(self) -> list[StationView]:
        """A snapshot of every session for the trainer console.

        Ordered by `trainee_id` rather than by score: the console ranks the
        help queue itself, and a station grid whose cards reordered every
        time a score ticked would be unreadable — a trainer looking at one
        station would lose it mid-glance.

        Carries each session's *latest* observation only, not its history.
        The rolling history is what the scorer reads; the console draws one
        pose, so handing it thirty would be shipping 29 poses nothing renders.
        """
        views = []
        for trainee_id, session in sorted(self._sessions.items()):
            history = session.track.history
            latest = history[-1] if history else None
            views.append(
                StationView(
                    station_id=session.station_id,
                    trainee_id=trainee_id,
                    connected=session.connected,
                    last_seen_ts=session.last_seen_ts,
                    observations=len(history),
                    bbox_xyxy=latest.bbox_xyxy if latest else None,
                    keypoints_xy=tuple(latest.keypoints_xy) if latest else None,
                    keypoints_conf=tuple(latest.keypoints_conf) if latest else None,
                    form_reason_codes=latest.form_reason_codes if latest else (),
                    exercise=latest.exercise if latest else "",
                    rep_count=latest.rep_count if latest else None,
                    form_ok=latest.form_ok if latest else None,
                )
            )
        return views

    def expire_stale(self, now: float) -> list[str]:
        """Drop sessions silent for longer than `track_ttl_s`; return their ids."""
        stale = [
            trainee_id
            for trainee_id, s in self._sessions.items()
            if now - s.last_seen_ts > self._ttl
        ]
        for trainee_id in stale:
            del self._sessions[trainee_id]
        return stale
