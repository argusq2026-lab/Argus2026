"""Argus — many eyes, one mind.

Deterministic, explainable triage of who on a training floor needs a human
instructor now. Detection, pose, and super-resolution run on the Snapdragon X
Elite Hexagon NPU; a small VLM samples only already-flagged trainees.

Privacy is a property of the wiring, not of a filter: raw frames and raw VLM
captions have no path to a sink, because every sink's signature accepts only
:class:`~argus.triage.TriageRecord`.
"""

from argus.config import ArgusConfig, ConfigError, load_config
from argus.triage import TriageRecord

__version__ = "0.1.0"

__all__ = ["ArgusConfig", "ConfigError", "load_config", "TriageRecord", "__version__"]
