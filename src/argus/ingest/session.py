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
from argus.outputs import SessionSummary, StationView
from argus.triage import FrameObservation, TrackState, history_len_for


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
    #: Carried from the `hello` so the console can name a person rather than
    #: a device. Re-set on reconnect: the same rack may be a different trainee.
    display_name: str = ""


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

    def register(
        self,
        station_id: str,
        trainee_id: str,
        now: float,
        display_name: str = "",
        use_case: str = "fitness",
    ) -> StationSession:
        """Start a new session, or resume one within its disconnect grace window.

        `use_case` is set on the track immediately, at handshake, rather than
        waiting for the first `observation` to set it (`TrackState.push`
        already does that, latest-wins) — the same reasoning as showing the
        station on the console at handshake: a welding station that connects
        and sends nothing should read as "waiting", not be mislabelled
        `"fitness"` — `TrackState`'s own default — until a frame arrives.
        """
        existing = self._sessions.get(trainee_id)
        if existing is not None:
            if existing.connected:
                raise DuplicateTraineeError(f"trainee_id {trainee_id!r} is already connected")
            existing.station_id = station_id
            existing.connected = True
            existing.last_seen_ts = now
            existing.display_name = display_name
            existing.track.use_case = use_case
            return existing

        track = TrackState(history_len=history_len_for(use_case, self._scoring))
        track.use_case = use_case
        session = StationSession(
            station_id=station_id,
            trainee_id=trainee_id,
            track=track,
            last_seen_ts=now,
            display_name=display_name,
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

    def note_idle(self, trainee_id: str, now: float) -> None:
        """The station is alive and watching nobody.

        Refreshes `last_seen_ts` exactly as an observation does — that is the
        point: a healthy station pointed at an empty rack must not be evicted
        and forced to reconnect. Raises `KeyError` on an already-expired
        session, same as `push_observation`, so the caller's handling is one
        path rather than two.
        """
        session = self._sessions[trainee_id]
        session.track.note_idle()
        session.last_seen_ts = now

    def tracks(self) -> dict[str, TrackState]:
        """Every session's history, connected or still in its grace window."""
        return {trainee_id: s.track for trainee_id, s in self._sessions.items()}

    def _summarise(self, track: TrackState) -> SessionSummary:
        """Project a track's running account into the console's closed view.

        `fault_rate` is resolved here rather than on the page, so the "is
        there enough work to call this a rate" judgement lives next to the
        thresholds that decide it instead of being re-implemented in
        JavaScript where it could drift.
        """
        metrics = track.session
        return SessionSummary(
            rolling_score=round(metrics.rolling_score, 4),
            peak_score=round(metrics.peak_score, 4),
            active_s=round(metrics.active_s, 2),
            reps=metrics.reps,
            reps_flagged=metrics.reps_flagged,
            hold_s=round(metrics.hold_s, 2),
            hold_flagged_s=round(metrics.hold_flagged_s, 2),
            fault_rate=metrics.fault_rate(self._scoring),
            code_counts=dict(metrics.code_counts),
        )

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
                    display_name=session.display_name,
                    connected=session.connected,
                    last_seen_ts=session.last_seen_ts,
                    observations=len(history),
                    subject_present=session.track.subject_present,
                    use_case=session.track.use_case,
                    bbox_xyxy=latest.bbox_xyxy if latest else None,
                    keypoints_xy=tuple(latest.keypoints_xy) if latest and latest.keypoints_xy is not None else None,
                    keypoints_conf=tuple(latest.keypoints_conf) if latest and latest.keypoints_conf is not None else None,
                    form_reason_codes=latest.form_reason_codes if latest else (),
                    # `latest.exercise` is fitness's own field and is `None`
                    # (not `""`) for a use case that never sets it, e.g.
                    # welding — `StationView.exercise` is a bare `str`, so
                    # that `None` is normalized here rather than reaching a
                    # field whose contract says "empty string means unset".
                    exercise=(latest.exercise or "") if latest else "",
                    # Same `None`-to-`""` normalization as `exercise` above,
                    # and for the same reason: `procedure` is nursing's own
                    # field and is `None` for every other use case.
                    procedure=(latest.procedure or "") if latest else "",
                    rep_count=latest.rep_count if latest else None,
                    form_ok=latest.form_ok if latest else None,
                    session=self._summarise(session.track) if latest else None,
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
