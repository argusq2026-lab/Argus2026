"""Identity: the property that makes `trainee_id` usable as a triage key."""

from __future__ import annotations

import numpy as np
import pytest

from argus.tracking.appearance import blend, distance, signature
from argus.tracking.kalman import KalmanBoxFilter, bbox_to_z, z_to_bbox
from argus.tracking.tracker import MultiObjectTracker
from argus.vision.detect import Detection


def make_frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 40, dtype=np.uint8)


def frame_with_person(box, colour, width=320, height=240) -> np.ndarray:
    """A frame with one solid-coloured rectangle, so appearance is meaningful."""
    frame = make_frame(width, height)
    x0, y0, x1, y1 = (int(v) for v in box)
    frame[y0:y1, x0:x1] = colour
    return frame


# -- Kalman -----------------------------------------------------------------


def test_bbox_state_round_trip():
    box = (10.0, 20.0, 50.0, 120.0)
    np.testing.assert_allclose(z_to_bbox(bbox_to_z(box)), box, atol=1e-6)


def test_constant_velocity_is_learned_and_projected():
    kf = KalmanBoxFilter((0.0, 0.0, 20.0, 60.0))
    for step in range(1, 12):
        kf.predict()
        kf.update((10.0 * step, 0.0, 20.0 + 10.0 * step, 60.0))
    vx, _ = kf.velocity
    assert vx > 5.0, "the filter should have learned the rightward motion"
    predicted = kf.predict()
    assert predicted[0] > 10.0 * 11, "prediction must lead the last observation"


def test_area_velocity_cannot_drive_area_negative():
    kf = KalmanBoxFilter((0.0, 0.0, 40.0, 40.0))
    kf.x[6] = -1e9  # pathological shrink rate
    x0, y0, x1, y1 = kf.predict()
    assert x1 > x0 and y1 > y0


# -- appearance -------------------------------------------------------------


def test_signature_is_normalised():
    patch = np.full((40, 20, 3), (10, 200, 30), dtype=np.uint8)
    sig = signature(patch, bins=12)
    assert sig is not None
    assert sig.sum() == pytest.approx(1.0, abs=1e-5)


def test_signature_rejects_a_degenerate_crop():
    assert signature(np.zeros((0, 0, 3), dtype=np.uint8), bins=12) is None
    assert signature(np.zeros((2, 2, 3), dtype=np.uint8), bins=12) is None


def test_distance_separates_different_colours():
    red = signature(np.full((40, 20, 3), (0, 0, 220), dtype=np.uint8), bins=12)
    blue = signature(np.full((40, 20, 3), (220, 0, 0), dtype=np.uint8), bins=12)
    assert distance(red, red) == pytest.approx(0.0, abs=1e-5)
    assert distance(red, blue) > 0.9


def test_unknown_signature_is_neutral_not_blocking():
    """An unknown signature must let the motion term decide."""
    sig = signature(np.full((40, 20, 3), (0, 0, 220), dtype=np.uint8), bins=12)
    assert distance(sig, None) == 0.5
    assert distance(None, None) == 0.5


def test_blend_keeps_the_signature_normalised():
    a = signature(np.full((40, 20, 3), (0, 0, 220), dtype=np.uint8), bins=12)
    b = signature(np.full((40, 20, 3), (220, 0, 0), dtype=np.uint8), bins=12)
    merged = blend(a, b, momentum=0.9)
    assert merged.sum() == pytest.approx(1.0, abs=1e-5)
    assert blend(None, b, 0.9) is b
    assert blend(a, None, 0.9) is a


# -- tracker ----------------------------------------------------------------


def test_ids_are_namespaced_by_camera(default_config):
    tracker = MultiObjectTracker("cam7", default_config.tracking, default_config.scoring)
    tracker.update([Detection((10.0, 10.0, 50.0, 130.0), 0.9)], make_frame(), 0.0)
    assert all(tid.startswith("cam7-") for tid in tracker.tracks)


def test_a_walking_trainee_keeps_one_id(default_config):
    tracker = MultiObjectTracker("cam0", default_config.tracking, default_config.scoring)
    for step in range(20):
        x = 10.0 + 4.0 * step
        box = (x, 40.0, x + 40.0, 160.0)
        tracker.update([Detection(box, 0.9)], frame_with_person(box, (0, 0, 200)), step / 15)
    assert len(tracker.tracks) == 1
    assert tracker.stats.created == 1


def test_id_survives_an_occlusion(default_config):
    """The prototype's centroid tracker lost the id here, resetting the
    trainee's whole triage history mid-incident."""
    tracker = MultiObjectTracker("cam0", default_config.tracking, default_config.scoring)

    def box_at(step: int):
        x = 10.0 + 4.0 * step
        return (x, 40.0, x + 40.0, 160.0)

    for step in range(10):
        box = box_at(step)
        tracker.update([Detection(box, 0.9)], frame_with_person(box, (0, 0, 200)), step / 15)
    original = set(tracker.tracks)
    assert len(original) == 1

    # Fully occluded for 12 frames -- no detections at all.
    for step in range(10, 22):
        tracker.update([], make_frame(), step / 15)
    assert set(tracker.tracks) == original, "track must coast, not be deleted"

    # Reappears where the motion model predicted, same appearance.
    box = box_at(22)
    tracker.update([Detection(box, 0.9)], frame_with_person(box, (0, 0, 200)), 22 / 15)
    assert set(tracker.tracks) == original
    assert tracker.stats.created == 1
    assert tracker.stats.reassociated_after_occlusion == 1


def test_a_track_is_deleted_after_max_age(default_config):
    tracker = MultiObjectTracker("cam0", default_config.tracking, default_config.scoring)
    box = (10.0, 40.0, 50.0, 160.0)
    tracker.update([Detection(box, 0.9)], frame_with_person(box, (0, 0, 200)), 0.0)
    for step in range(default_config.tracking.max_age_frames + 2):
        tracker.update([], make_frame(), (step + 1) / 15)
    assert tracker.tracks == {}
    assert tracker.stats.deleted == 1


def test_appearance_prevents_an_id_swap_when_paths_cross(default_config):
    """Two trainees converging: motion alone is ambiguous, colour is not."""
    tracker = MultiObjectTracker("cam0", default_config.tracking, default_config.scoring)

    red, blue = (0, 0, 220), (220, 0, 0)
    for step in range(12):
        left = (20.0 + 6.0 * step, 40.0, 60.0 + 6.0 * step, 160.0)
        right = (240.0 - 6.0 * step, 40.0, 280.0 - 6.0 * step, 160.0)
        frame = make_frame()
        frame[40:160, int(left[0]) : int(left[2])] = red
        frame[40:160, int(right[0]) : int(right[2])] = blue
        tracker.update(
            [Detection(left, 0.9), Detection(right, 0.9)], frame, step / 15
        )

    assert len(tracker.tracks) == 2
    assert tracker.stats.created == 2, "no identity should have been re-created"


def test_a_one_frame_spike_is_not_publishable(default_config):
    """A single detection must not create a trainee an instructor is sent to."""
    tracker = MultiObjectTracker("cam0", default_config.tracking, default_config.scoring)
    box = (10.0, 40.0, 50.0, 160.0)
    tracker.update([Detection(box, 0.9)], frame_with_person(box, (0, 0, 200)), 0.0)
    assert tracker.tracks
    assert tracker.track_states() == {}


def test_matching_is_deterministic(default_config):
    """Same detections in the same order must give the same assignment."""
    def run():
        tracker = MultiObjectTracker("cam0", default_config.tracking, default_config.scoring)
        assignments = []
        for step in range(8):
            boxes = [
                (20.0 + 5 * step, 40.0, 60.0 + 5 * step, 160.0),
                (150.0 + 3 * step, 40.0, 190.0 + 3 * step, 160.0),
            ]
            frame = make_frame()
            for box, colour in zip(boxes, [(0, 0, 220), (220, 0, 0)]):
                frame[40:160, int(box[0]) : int(box[2])] = colour
            assigned = tracker.update([Detection(b, 0.9) for b in boxes], frame, step / 15)
            assignments.append(sorted(assigned))
        return assignments

    assert run() == run()
