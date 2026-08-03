"""The multi-camera capture -> triage -> emit loop."""

from argus.pipeline.runner import ArgusPipeline, FrameClock, TickResult, WallClock

__all__ = ["ArgusPipeline", "FrameClock", "WallClock", "TickResult"]
