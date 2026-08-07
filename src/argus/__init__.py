"""Argus — many eyes, one mind.

Deterministic, explainable triage of who on a training floor needs a human
instructor now. One phone per trainee runs its own on-device pose and
form/exercise classifier and streams structured numeric results to this
process over WebSocket (see `argus.ingest` and `docs/PROTOCOL.md`); Argus
ranks who needs attention and gives the trainer a live view.

Privacy is a property of the wiring, not of a filter: no frame or free-text
caption has a path to a sink, because every sink's signature accepts only
:class:`~argus.triage.TriageRecord` — and no frame ever exists past the
phone's own camera pipeline in the first place.
"""

from argus.config import ArgusConfig, ConfigError, load_config
from argus.triage import TriageRecord

__version__ = "0.2.1"

__all__ = ["ArgusConfig", "ConfigError", "load_config", "TriageRecord", "__version__"]
