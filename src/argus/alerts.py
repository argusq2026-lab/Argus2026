"""The alert boundary.

`emit_alert` is the only function permitted to carry perception output to a
human. Its signature accepts one :class:`~argus.triage.TriageRecord` and
nothing else — four scalar fields — so there is no parameter through which a
frame, a crop, or a VLM caption could pass. The boundary is the type, not a
check inside the function: a runtime redaction filter can be bypassed by the
next person who adds an argument, whereas widening this signature is a visible
change to a module that `tests/test_privacy.py` inspects.

Like `argus.outputs`, this module imports no image library at all.
"""

from __future__ import annotations

import sys
from typing import Callable

from argus.triage import TriageRecord

#: A sink is anything that consumes one redacted record.
AlertSink = Callable[[TriageRecord], None]


def format_alert(record: TriageRecord) -> str:
    reasons = ", ".join(record.reason_codes) if record.reason_codes else "-"
    return (
        f"[ALERT] {record.trainee_id} score={record.score:.2f} "
        f"reasons=[{reasons}] ts={record.ts:.1f}"
    )


def emit_alert(record: TriageRecord) -> None:
    """Surface one trainee to a human instructor, on stderr."""
    print(format_alert(record), file=sys.stderr, flush=True)


class ConsoleAlertSink:
    """Console sink that suppresses repeats of an unchanged alert.

    A trainee who is still on the floor and still flagged would otherwise
    reprint every frame at 15 FPS, which trains an instructor to ignore the
    console. Re-alerts when the reason codes change, or when the score crosses
    a decile — i.e. when something about the situation is new.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last: dict[str, tuple[tuple[str, ...], int]] = {}

    def __call__(self, record: TriageRecord) -> None:
        if not self.enabled:
            return
        key = (record.reason_codes, int(record.score * 10))
        if self._last.get(record.trainee_id) == key:
            return
        self._last[record.trainee_id] = key
        emit_alert(record)

    def forget(self, trainee_id: str) -> None:
        """Drop suppression state for a trainee who has left the floor."""
        self._last.pop(trainee_id, None)
