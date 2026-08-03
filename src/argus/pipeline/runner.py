"""The capture -> infer -> track -> triage -> emit loop, across N cameras.

Per tick, every enabled camera contributes one frame. Each camera owns its own
tracker (so ids and motion models never mix) and its own VLM cadence budget (so
a busy floor cannot starve a quiet one). The triage rank is then computed once
over the union of all cameras' tracks, which is what an instructor actually
needs: one ordered list of who to go to, not one list per camera.

Ordering inside a tick matters and is the fix for the prototype's dead VLM
gate. Observations are pushed first, the rank is computed from them, and only
then does the prefilter decide who is worth captioning — a gate that reads a
trainee's real score cannot be evaluated before that score exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from argus.alerts import AlertSink
from argus.config import ArgusConfig
from argus.engines.factory import VisionStack, build_vision_stack
from argus.outputs import JsonLogSink, TriageHTTPServer
from argus.pipeline.overlay import OverlayWriter, draw_overlay
from argus.pipeline.prefilter import select_for_vlm
from argus.pipeline.source import CameraSource, open_sources
from argus.triage import (
    FrameObservation,
    TriageRecord,
    needs_instructor,
    rank_trainees,
)
from argus.tracking import MultiObjectTracker
from argus.vision.preprocess import crop
from argus.vlm import VLMCaptioner, build_captioner


class FrameClock:
    """Deterministic clock: timestamps derive from the tick index, not the wall.

    A run over the same video therefore produces byte-identical output on a
    fast machine and a slow one, which is what makes the rank reproducible and
    the determinism test meaningful.
    """

    def __init__(self, fps: float = 15.0):
        self.fps = fps if fps > 0 else 15.0
        self.tick = 0

    def now(self) -> float:
        return self.tick / self.fps

    def advance(self) -> None:
        self.tick += 1


class WallClock:
    """Monotonic clock, for live cameras where frame index is not time."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._start

    def advance(self) -> None:
        pass


@dataclass
class CameraRuntime:
    """Everything the pipeline keeps per camera."""

    source: CameraSource
    tracker: MultiObjectTracker
    #: None when the vision stack is shared across cameras (every real backend).
    stack: VisionStack | None = None
    last_vlm_ts: dict[str, float] = field(default_factory=dict)
    overlay: OverlayWriter | None = None
    frames_processed: int = 0
    vlm_calls: int = 0


@dataclass
class TickResult:
    """One tick's merged outcome."""

    ts: float
    records: list[TriageRecord]
    alerts: list[TriageRecord]
    frames_read: int
    vlm_calls: int


class ArgusPipeline:
    """Owns the cameras, the vision stack, the trackers, and the sinks."""

    def __init__(
        self,
        cfg: ArgusConfig,
        stack: VisionStack | None = None,
        captioner: VLMCaptioner | None = None,
        clock: FrameClock | WallClock | None = None,
        alert_sink: AlertSink | None = None,
        headless: bool = True,
    ):
        self.cfg = cfg
        self.headless = headless
        # A real backend's sessions are shared across cameras: the graphs are
        # identical and an NPU context is expensive to duplicate. The mock is
        # the exception -- its synthetic scene advances on a per-runner call
        # counter, so a shared runner would interleave two cameras into one
        # scene and hand each of them every other frame. Each camera therefore
        # gets its own mock stack, which is also the more faithful fixture:
        # two cameras watching the same floor independently.
        self._per_camera_stack = stack is None and cfg.engine.kind == "mock"
        self._stack = (
            stack if stack is not None
            else (None if self._per_camera_stack else build_vision_stack(cfg))
        )
        self._owns_stack = stack is None
        self._captioner = captioner if captioner is not None else build_captioner(cfg.vlm)
        self._owns_captioner = captioner is None
        self._alert_sink = alert_sink

        sources = open_sources(cfg.enabled_cameras())
        self._cameras = [
            CameraRuntime(
                source=source,
                tracker=MultiObjectTracker(source.camera_id, cfg.tracking, cfg.scoring),
                stack=build_vision_stack(cfg) if self._per_camera_stack else None,
                overlay=(
                    OverlayWriter(
                        _per_camera_path(cfg.outputs.overlay_out, source.camera_id,
                                         len(cfg.enabled_cameras())),
                        source.fps,
                    )
                    if cfg.outputs.overlay_out
                    else None
                ),
            )
            for source in sources
        ]

        if clock is not None:
            self._clock: FrameClock | WallClock = clock
        elif any(camera.source.is_live for camera in self._cameras):
            self._clock = WallClock()
        else:
            self._clock = FrameClock(self._cameras[0].source.fps)

        self._json_sink = JsonLogSink(cfg.outputs.json_log) if cfg.outputs.json_log else None
        self._http: TriageHTTPServer | None = None
        if cfg.outputs.http_port:
            self._http = TriageHTTPServer(cfg.outputs.http_port, cfg.outputs.http_host)
            self._http.start()

    # -- properties ---------------------------------------------------------

    @property
    def cameras(self) -> list[CameraRuntime]:
        return self._cameras

    @property
    def http_port(self) -> int | None:
        return self._http.port if self._http is not None else None

    @property
    def exhausted(self) -> bool:
        return all(camera.source.exhausted for camera in self._cameras)

    # -- one tick -----------------------------------------------------------

    def tick(self) -> TickResult:
        ts = self._clock.now()
        frames_read = 0
        vlm_calls = 0

        for camera in self._cameras:
            frame = camera.source.read()
            if frame is None:
                continue
            frames_read += 1
            camera.frames_processed += 1
            vlm_calls += self._process_camera(camera, frame.image_bgr, ts)

        records = self._merged_rank(ts)
        alerts = needs_instructor(records, self.cfg.scoring)

        if self._alert_sink is not None:
            for record in alerts:
                self._alert_sink(record)
        if self._json_sink is not None:
            self._json_sink.write(ts, records)
        if self._http is not None:
            self._http.update(ts, records)

        self._clock.advance()
        return TickResult(ts, records, alerts, frames_read, vlm_calls)

    def _stack_for(self, camera: CameraRuntime) -> VisionStack:
        return camera.stack if camera.stack is not None else self._stack

    def _process_camera(
        self, camera: CameraRuntime, image: np.ndarray, ts: float
    ) -> int:
        stack = self._stack_for(camera)
        detections = stack.detector.detect(image)
        assigned = camera.tracker.update(detections, image, ts)
        frame_area = float(image.shape[0] * image.shape[1])

        for track_id, detection in assigned.items():
            pose = stack.pose.estimate(image, detection.bbox_xyxy, frame_area)
            camera.tracker.tracks[track_id].state.push(
                FrameObservation(
                    ts=ts,
                    bbox_xyxy=detection.bbox_xyxy,
                    keypoints_xy=[(float(x), float(y)) for x, y in pose.keypoints_xy],
                    keypoints_conf=[float(c) for c in pose.keypoints_conf],
                    vlm_caption=None,
                ),
                self.cfg.scoring,
            )

        vlm_calls = self._sample_vlm(camera, image, ts)

        if camera.overlay is not None or self.cfg.outputs.overlay_window:
            self._draw(camera, image, assigned, ts)
        return vlm_calls

    def _sample_vlm(
        self, camera: CameraRuntime, image: np.ndarray, ts: float
    ) -> int:
        """Caption the highest-scoring flagged trainees that are due.

        The per-camera rank is computed here, after this frame's observations
        have landed, because the gate is defined on the trainee's *current*
        triage score.
        """
        states = camera.tracker.track_states()
        if not states:
            return 0

        angle = camera.source.reference_angle_deg
        angles = {tid: angle for tid in states} if angle is not None else None
        camera_records = rank_trainees(states, ts, self.cfg.scoring, reference_angles=angles)

        selected = select_for_vlm(
            camera_records,
            camera.last_vlm_ts,
            ts,
            self.cfg.vlm,
            eligible_ids=states.keys(),
        )

        for trainee_id in selected:
            track = camera.tracker.tracks[trainee_id]
            person_crop = crop(image, track.last_bbox)
            if person_crop.size == 0:
                continue
            # The caption is a local: scored into a number, then dropped. It is
            # never stored on the track and never reaches a sink.
            caption = self._captioner.caption(person_crop)
            track.state.apply_caption(caption, self.cfg.scoring)
            camera.last_vlm_ts[trainee_id] = ts
            camera.vlm_calls += 1
        return len(selected)

    def _merged_rank(self, ts: float) -> list[TriageRecord]:
        """One ordered list across every camera. Ids are already namespaced."""
        states = {}
        angles: dict[str, float] = {}
        for camera in self._cameras:
            angle = camera.source.reference_angle_deg
            for trainee_id, state in camera.tracker.track_states().items():
                states[trainee_id] = state
                if angle is not None:
                    angles[trainee_id] = angle
        return rank_trainees(states, ts, self.cfg.scoring, reference_angles=angles or None)

    def _draw(
        self, camera: CameraRuntime, image: np.ndarray, assigned, ts: float
    ) -> None:
        import cv2

        tracks = camera.tracker.tracks
        publishable = camera.tracker.track_states()
        boxes = {
            tid: (tracks[tid].last_bbox if tid in assigned else tracks[tid].kalman.bbox)
            for tid in publishable
        }
        coasting = frozenset(tid for tid in publishable if tid not in assigned)
        records = {
            r.trainee_id: r
            for r in rank_trainees(publishable, ts, self.cfg.scoring)
        }
        annotated = draw_overlay(
            image, boxes, records, self.cfg.scoring.alert_threshold, coasting
        )
        if camera.overlay is not None:
            camera.overlay.write(annotated)
        if self.cfg.outputs.overlay_window and not self.headless:
            cv2.imshow(f"Argus [{camera.camera_id}]", annotated)
            cv2.waitKey(1)

    # -- run / teardown -----------------------------------------------------

    def run(self, max_ticks: int | None = None) -> int:
        """Loop until every source is exhausted or `max_ticks` is reached."""
        ticks = 0
        try:
            while not self.exhausted:
                if max_ticks is not None and ticks >= max_ticks:
                    break
                result = self.tick()
                if result.frames_read == 0:
                    break
                ticks += 1
        finally:
            self.close()
        return ticks

    def close(self) -> None:
        for camera in self._cameras:
            camera.source.release()
            if camera.overlay is not None:
                camera.overlay.release()
            if camera.stack is not None:
                camera.stack.close()
        if self._json_sink is not None:
            self._json_sink.close()
        if self._http is not None:
            self._http.stop()
        if self._owns_stack and self._stack is not None:
            self._stack.close()
        if self._owns_captioner:
            self._captioner.close()
        if self.cfg.outputs.overlay_window and not self.headless:
            import cv2

            cv2.destroyAllWindows()


def _per_camera_path(template: str, camera_id: str, camera_count: int) -> str:
    """Give each camera its own overlay file when more than one is configured."""
    if camera_count <= 1:
        return template
    from pathlib import Path

    path = Path(template)
    return str(path.with_name(f"{path.stem}_{camera_id}{path.suffix}"))
