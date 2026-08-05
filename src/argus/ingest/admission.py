"""Who is allowed onto this instructor's floor.

A phone that has found a session (`argus.discovery`) and sent a well-formed
`hello` is not necessarily one the instructor wants scoring a trainee. In
`session.approval = "manual"` the connection is parked here — held open,
acknowledged as pending, and shown on the console — until the instructor
approves it, denies it, or it times out.

In `"auto"`, the default, nothing in this module runs at all: `hello` is
acknowledged immediately, exactly as it was before admission was a concept.
That asymmetry is deliberate. An unwanted phone on the console is a nuisance
an instructor can see and disconnect; a trainee standing at a rack
unmonitored because nobody noticed a prompt is the failure this system exists
to prevent. Gating admission by default would trade the second for the first.

Three properties are worth stating, because each of them is a way this could
quietly go wrong:

* **A request always ends.** Approved, denied, superseded, timed out, or
  withdrawn because the phone hung up — every path resolves the waiter and
  tells it why. A join that silently never resolves is a phone that looks
  hung and a trainee nobody is watching.
* **A reconnecting phone is not blocked by its own stale request.** A second
  request for a `trainee_id` already pending supersedes the first rather than
  being refused, because the usual cause is a phone that dropped and came
  back, and refusing would lock it out for the whole timeout.
* **Decisions arrive from another thread.** The console's HTTP handler runs
  off the event loop, so `decide` hands the wake-up back to the loop rather
  than touching an `asyncio.Event` across threads.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from enum import Enum

from argus.outputs import PendingJoinView


class Decision(str, Enum):
    """How a join request ended. Every request ends as exactly one of these."""

    APPROVED = "approved"
    DENIED = "denied"
    #: The instructor never answered within `session.join_timeout_s`.
    TIMED_OUT = "timed_out"
    #: A newer request for the same `trainee_id` replaced this one — almost
    #: always the same phone reconnecting after a drop.
    SUPERSEDED = "superseded"
    #: The phone hung up while waiting, so there is nobody left to admit.
    WITHDRAWN = "withdrawn"


@dataclass
class JoinRequest:
    """One phone waiting at the door."""

    request_id: str
    station_id: str
    trainee_id: str
    #: Optional human-readable label the phone offered, so an instructor
    #: approving has something to recognise beyond an opaque device id.
    #: Phone-chosen and length-bounded by `argus.ingest.protocol`; display-only.
    display_name: str
    #: Server clock, the same base as everything else the console renders.
    requested_ts: float
    expires_ts: float
    decision: Decision | None = None
    _resolved: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """What to call this request on screen."""
        return self.display_name or self.trainee_id

    def resolve(self, decision: Decision) -> bool:
        """Settle this request. Safe to call from any thread; first call wins.

        Returns False if it was already settled — a double-click on Approve,
        or a decision that raced the timeout. Second callers are told they
        lost rather than silently overwriting the first outcome.
        """
        if self.decision is not None:
            return False
        self.decision = decision
        loop, event = self._loop, self._resolved
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(event.set)
        else:
            event.set()
        return True

    async def wait(self, timeout_s: float) -> Decision:
        """Block until settled. Resolves as `TIMED_OUT` rather than raising."""
        try:
            await asyncio.wait_for(self._resolved.wait(), timeout=timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            self.resolve(Decision.TIMED_OUT)
        return self.decision or Decision.TIMED_OUT


class AdmissionQueue:
    """Every phone currently waiting for the instructor to decide.

    Guarded by a plain lock rather than being loop-affine: requests are
    created on the event loop and decided from the console's HTTP thread, so
    the container is genuinely shared and pretending otherwise would be the
    bug rather than the simplification.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, JoinRequest] = {}
        self._counter = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)

    def submit(
        self, station_id: str, trainee_id: str, display_name: str, now: float, timeout_s: float
    ) -> JoinRequest:
        """Park a phone at the door, superseding its own earlier request.

        The caller must already be on the event loop: the returned request
        binds to the running loop so a decision from the console's thread can
        wake it.
        """
        request = JoinRequest(
            request_id="",
            station_id=station_id,
            trainee_id=trainee_id,
            display_name=display_name,
            requested_ts=now,
            expires_ts=now + timeout_s,
        )
        request._loop = asyncio.get_running_loop()

        with self._lock:
            self._counter += 1
            request.request_id = f"join-{self._counter}"
            stale = [r for r in self._pending.values() if r.trainee_id == trainee_id]
            for old in stale:
                del self._pending[old.request_id]
            self._pending[request.request_id] = request

        # Resolved outside the lock: `resolve` hands work to an event loop and
        # holding a lock across that is how a deadlock gets written.
        for old in stale:
            old.resolve(Decision.SUPERSEDED)
        return request

    def decide(self, request_id: str, approve: bool) -> bool:
        """Approve or deny by id. Returns False if it is no longer pending.

        Called from the console's HTTP thread. A stale id is the normal case,
        not an error: the phone may have hung up or timed out between the page
        rendering the button and someone pressing it.
        """
        with self._lock:
            request = self._pending.pop(request_id, None)
        if request is None:
            return False
        return request.resolve(Decision.APPROVED if approve else Decision.DENIED)

    def withdraw(self, request_id: str) -> None:
        """The phone hung up while waiting; stop offering it for approval."""
        with self._lock:
            request = self._pending.pop(request_id, None)
        if request is not None:
            request.resolve(Decision.WITHDRAWN)

    def expire(self, now: float) -> list[JoinRequest]:
        """Settle requests nobody answered. Called from the rank tick.

        `JoinRequest.wait` times out on its own, so this is not what unblocks
        the phone — it is what stops an unanswered prompt sitting on the
        console forever if that waiter is gone.
        """
        with self._lock:
            expired = [r for r in self._pending.values() if now >= r.expires_ts]
            for request in expired:
                del self._pending[request.request_id]
        for request in expired:
            request.resolve(Decision.TIMED_OUT)
        return expired

    def pending(self) -> list[JoinRequest]:
        """Everyone waiting, oldest first — the order they should be answered."""
        with self._lock:
            return sorted(self._pending.values(), key=lambda r: (r.requested_ts, r.request_id))

    def pending_views(self) -> list[PendingJoinView]:
        """The same queue as the console is allowed to see it.

        A `JoinRequest` owns an event loop handle and an `asyncio.Event`;
        `PendingJoinView` is the closed, serialisable projection of it, the
        same split `SessionRegistry.station_views` makes for sessions.
        """
        return [
            PendingJoinView(
                request_id=r.request_id,
                station_id=r.station_id,
                trainee_id=r.trainee_id,
                display_name=r.display_name,
                requested_ts=r.requested_ts,
                expires_ts=r.expires_ts,
            )
            for r in self.pending()
        ]
