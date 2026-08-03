"""Multi-object tracker with motion prediction and appearance re-association.

Replaces the prototype's centroid tracker, whose failure mode was structural:
it matched only against the last observed centroid, kept no velocity, and
deleted a track after two seconds unseen. A trainee stepping behind equipment
came back as a new `trainee_id` with an empty history — so their stillness and
fall windows reset, and any alert already raised about them pointed at an ID
that no longer existed.

This tracker keeps a Kalman estimate advancing through the occlusion and
re-associates on a combined motion + appearance cost, so the same person keeps
the same key. Matching is greedy over the globally-cheapest admissible pair,
which is deterministic — the same detections in the same order always produce
the same assignment, which the rank's reproducibility depends on.

Track ids are namespaced by camera (`cam0-t3`) so two sources can never collide
in the merged rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from argus.config import ScoringConfig, TrackingConfig
from argus.tracking.appearance import blend, distance, signature
from argus.tracking.kalman import KalmanBoxFilter
from argus.triage import TrackState
from argus.vision.detect import Detection
from argus.vision.nms import iou_matrix
from argus.vision.preprocess import crop


@dataclass
class Track:
    """One tracked trainee: identity, motion estimate, appearance, and history."""

    track_id: str
    kalman: KalmanBoxFilter
    state: TrackState
    signature: np.ndarray | None = None
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_seen_ts: float = 0.0
    #: Frames this track has spent coasting on prediction alone, cumulatively.
    occluded_frames: int = 0

    @property
    def confirmed_after(self) -> int:
        return self.hits

    def is_publishable(self, min_hits: int) -> bool:
        """Whether this track is solid enough to appear in a triage rank.

        A one-frame detection spike must not create a trainee an instructor
        could be sent to.
        """
        return self.hits >= min_hits


@dataclass
class TrackerStats:
    """Observability for identity churn — the thing this tracker exists to fix."""

    created: int = 0
    deleted: int = 0
    reassociated_after_occlusion: int = 0


class MultiObjectTracker:
    """Per-camera tracker. One instance per source; ids are camera-namespaced."""

    def __init__(
        self,
        camera_id: str,
        cfg: TrackingConfig,
        scoring: ScoringConfig,
    ):
        self.camera_id = camera_id
        self._cfg = cfg
        self._scoring = scoring
        self._next_index = 0
        self._tracks: dict[str, Track] = {}
        self.stats = TrackerStats()

    # -- public API ---------------------------------------------------------

    @property
    def tracks(self) -> dict[str, Track]:
        return self._tracks

    def track_states(self, min_hits: int | None = None) -> dict[str, TrackState]:
        """Publishable tracks' triage histories, keyed by trainee id."""
        threshold = self._cfg.min_hits if min_hits is None else min_hits
        return {
            tid: t.state
            for tid, t in self._tracks.items()
            if t.is_publishable(threshold)
        }

    def update(
        self, detections: list[Detection], frame_bgr: np.ndarray, ts: float
    ) -> dict[str, Detection]:
        """Advance one frame. Returns the detection assigned to each track id."""
        for track in self._tracks.values():
            track.kalman.predict()
            track.age += 1
            track.time_since_update += 1

        signatures = [
            signature(crop(frame_bgr, det.bbox_xyxy), self._cfg.hist_bins)
            for det in detections
        ]

        matches, unmatched_dets = self._match(detections, signatures)

        assigned: dict[str, Detection] = {}
        for track_id, det_index in matches:
            track = self._tracks[track_id]
            if track.time_since_update > 1:
                self.stats.reassociated_after_occlusion += 1
                track.occluded_frames += track.time_since_update - 1
            track.kalman.update(detections[det_index].bbox_xyxy)
            track.signature = blend(
                track.signature, signatures[det_index], self._cfg.appearance_momentum
            )
            track.hits += 1
            track.time_since_update = 0
            track.last_bbox = detections[det_index].bbox_xyxy
            track.last_seen_ts = ts
            assigned[track_id] = detections[det_index]

        for det_index in unmatched_dets:
            track = self._create(detections[det_index], signatures[det_index], ts)
            assigned[track.track_id] = detections[det_index]

        self._prune()
        return assigned

    def predicted_bbox(self, track_id: str) -> tuple[float, float, float, float]:
        """Where a coasting track is believed to be, for overlay drawing."""
        return self._tracks[track_id].kalman.bbox

    # -- internals ----------------------------------------------------------

    def _match(
        self, detections: list[Detection], signatures: list[np.ndarray | None]
    ) -> tuple[list[tuple[str, int]], list[int]]:
        """Greedy globally-cheapest admissible assignment."""
        track_ids = sorted(self._tracks)  # sorted => deterministic tie-breaking
        if not track_ids or not detections:
            return [], list(range(len(detections)))

        track_boxes = np.array(
            [self._tracks[tid].kalman.bbox for tid in track_ids], dtype=np.float32
        )
        det_boxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        ious = iou_matrix(track_boxes, det_boxes)

        w = self._cfg.appearance_weight
        cost = np.full(ious.shape, np.inf, dtype=np.float32)
        for i, tid in enumerate(track_ids):
            track_sig = self._tracks[tid].signature
            for j in range(len(detections)):
                if ious[i, j] < self._cfg.iou_match_threshold:
                    continue
                app = distance(track_sig, signatures[j])
                if app > self._cfg.appearance_gate:
                    continue
                cost[i, j] = (1.0 - w) * (1.0 - ious[i, j]) + w * app

        matches: list[tuple[str, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        while True:
            masked = cost.copy()
            for i in used_tracks:
                masked[i, :] = np.inf
            for j in used_dets:
                masked[:, j] = np.inf
            if not np.isfinite(masked).any():
                break
            flat = int(np.argmin(masked))
            i, j = divmod(flat, masked.shape[1])
            if not np.isfinite(masked[i, j]):
                break
            matches.append((track_ids[i], j))
            used_tracks.add(i)
            used_dets.add(j)

        unmatched = [j for j in range(len(detections)) if j not in used_dets]
        return matches, unmatched

    def _create(
        self, det: Detection, sig: np.ndarray | None, ts: float
    ) -> Track:
        track_id = f"{self.camera_id}-t{self._next_index}"
        self._next_index += 1
        track = Track(
            track_id=track_id,
            kalman=KalmanBoxFilter(det.bbox_xyxy),
            state=TrackState(history_len=self._scoring.history_len),
            signature=sig,
            last_bbox=det.bbox_xyxy,
            last_seen_ts=ts,
        )
        self._tracks[track_id] = track
        self.stats.created += 1
        return track

    def _prune(self) -> None:
        stale = [
            tid
            for tid, t in self._tracks.items()
            if t.time_since_update > self._cfg.max_age_frames
        ]
        for tid in stale:
            del self._tracks[tid]
            self.stats.deleted += 1
