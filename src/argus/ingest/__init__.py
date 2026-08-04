"""The WebSocket ingest boundary — where a phone's numeric observations enter Argus.

This package replaces local camera capture entirely. One phone per trainee
opens one WebSocket connection here, sends a `hello`, then streams repeated
`observation` messages: bounding box, COCO-17 keypoints, and the phone's own
on-device form/exercise classifier output, all numeric, normalized to [0, 1]
of the phone's own frame. See `docs/PROTOCOL.md` for the exact wire format.

Like `argus.outputs` and `argus.alerts`, nothing in this package imports an
image library or names an image-capable type — there is no frame anywhere
past the phone's own camera pipeline, so there is nothing here that could
leak one even by accident.
"""

from argus.ingest.server import IngestServer, TickResult

__all__ = ["IngestServer", "TickResult"]
