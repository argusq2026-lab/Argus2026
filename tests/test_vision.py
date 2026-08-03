"""Pre/post-processing, keypoint remapping, and BlazePose decode."""

from __future__ import annotations

import numpy as np
import pytest

from argus.vision.blazepose import (
    DETECTOR_HEADS,
    decode_detector_head,
    decode_landmarks,
    sigmoid,
    ssd_anchor_centres,
)
from argus.vision.keypoints import (
    BLAZEPOSE_NAMES,
    BLAZEPOSE_NUM_LANDMARKS,
    COCO_FROM_BLAZEPOSE,
    COCO_NAMES,
    UNMAPPED_COCO_INDICES,
    blazepose_to_coco,
    empty_coco_keypoints,
)
from argus.vision.nms import iou_matrix, nms_cpu
from argus.vision.preprocess import (
    clamp_box,
    crop_padded,
    from_nchw_uint8,
    letterbox,
    square_roi,
    to_nchw_uint8,
    to_nhwc_uint8,
    undo_letterbox,
)


# -- letterbox --------------------------------------------------------------


def test_letterbox_preserves_aspect_and_pads_to_square():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    canvas, scale, (left, top) = letterbox(frame, 640)
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert (left, top) == (0, 80)
    assert canvas.dtype == np.uint8


def test_letterbox_pads_with_the_configured_value():
    frame = np.full((100, 400, 3), 255, dtype=np.uint8)
    canvas, _, (left, top) = letterbox(frame, 640, pad_value=114)
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[top + 5, left + 5].tolist() == [255, 255, 255]


def test_undo_letterbox_round_trips_a_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, scale, offset = letterbox(frame, 640)
    source = np.array([[100.0, 50.0, 300.0, 400.0]], dtype=np.float32)
    canvas_box = source * scale + np.array([offset[0], offset[1], offset[0], offset[1]])
    np.testing.assert_allclose(undo_letterbox(canvas_box, scale, offset), source, atol=1e-4)


def test_undo_letterbox_handles_empty_input():
    assert undo_letterbox(np.empty((0, 4)), 1.0, (0, 0)).shape == (0, 4)


def test_letterbox_rejects_non_image():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), dtype=np.uint8), 64)


# -- layout -----------------------------------------------------------------


def test_to_nchw_produces_the_detector_contract():
    """YOLO-X takes NCHW uint8 -- the prototype emitted NHWC int8."""
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    tensor = to_nchw_uint8(frame)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.uint8


def test_to_nhwc_produces_the_pose_contract():
    """The two BlazePose binaries take NHWC -- a different layout to the detector."""
    tensor = to_nhwc_uint8(np.zeros((128, 128, 3), dtype=np.uint8))
    assert tensor.shape == (1, 128, 128, 3)
    assert tensor.dtype == np.uint8


def test_layout_conversion_swaps_bgr_to_rgb():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[..., 0] = 10  # blue in OpenCV order
    frame[..., 2] = 30  # red
    tensor = to_nchw_uint8(frame, rgb=True)
    assert tensor[0, 0, 0, 0] == 30  # channel 0 is now red
    assert tensor[0, 2, 0, 0] == 10


def test_nchw_round_trip_restores_bgr():
    frame = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    np.testing.assert_array_equal(from_nchw_uint8(to_nchw_uint8(frame)), frame)


# -- crops ------------------------------------------------------------------


def test_clamp_box_never_returns_an_empty_crop():
    assert clamp_box((-50.0, -50.0, -10.0, -10.0), 100, 100) == (0, 0, 1, 1)


def test_crop_padded_zero_fills_outside_the_frame():
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    patch, box = crop_padded(frame, (-20.0, -20.0, 20.0, 20.0))
    assert patch.shape == (40, 40, 3)
    assert box == (-20.0, -20.0, 20.0, 20.0)
    assert patch[0, 0].tolist() == [0, 0, 0]      # outside the frame
    assert patch[-1, -1].tolist() == [255, 255, 255]  # inside


def test_square_roi_is_centred_and_square():
    x0, y0, x1, y1 = square_roi((10.0, 0.0, 30.0, 80.0), 1.0)
    assert (x1 - x0) == pytest.approx(y1 - y0)
    assert (x0 + x1) / 2 == pytest.approx(20.0)
    assert (y0 + y1) / 2 == pytest.approx(40.0)


# -- NMS --------------------------------------------------------------------


def test_nms_keeps_highest_score_and_drops_overlap():
    boxes = np.array(
        [[0, 0, 100, 100], [10, 10, 105, 105], [500, 500, 560, 560]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms_cpu(boxes, scores, 0.45) == [0, 2]


def test_nms_empty_input():
    assert nms_cpu(np.empty((0, 4)), np.empty((0,)), 0.45) == []


def test_iou_matrix_shape_and_identity():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    ious = iou_matrix(boxes, boxes)
    assert ious.shape == (2, 2)
    np.testing.assert_allclose(np.diag(ious), [1.0, 1.0], atol=1e-5)
    assert ious[0, 1] == pytest.approx(0.0)


# -- keypoint remap ---------------------------------------------------------


def test_blazepose_and_coco_name_tables_are_consistent():
    assert len(BLAZEPOSE_NAMES) == BLAZEPOSE_NUM_LANDMARKS
    assert len(COCO_NAMES) == 17
    for coco_idx, src in enumerate(COCO_FROM_BLAZEPOSE):
        if src is None:
            continue
        assert BLAZEPOSE_NAMES[src] == COCO_NAMES[coco_idx]


def test_upper_body_export_has_no_knees_or_ankles():
    assert UNMAPPED_COCO_INDICES == (13, 14, 15, 16)


def test_remap_moves_joints_to_their_coco_index():
    """The joints the scorer indexes must be the joints it thinks they are."""
    xy = np.zeros((BLAZEPOSE_NUM_LANDMARKS, 2), dtype=np.float32)
    vis = np.full((BLAZEPOSE_NUM_LANDMARKS,), 0.9, dtype=np.float32)
    xy[BLAZEPOSE_NAMES.index("left_shoulder")] = (11.0, 11.0)
    xy[BLAZEPOSE_NAMES.index("right_wrist")] = (16.0, 16.0)
    xy[BLAZEPOSE_NAMES.index("nose")] = (1.0, 1.0)

    coco_xy, coco_conf = blazepose_to_coco(xy, vis)
    assert tuple(coco_xy[5]) == (11.0, 11.0)   # KP_LEFT_SHOULDER
    assert tuple(coco_xy[10]) == (16.0, 16.0)  # KP_RIGHT_WRIST
    assert tuple(coco_xy[0]) == (1.0, 1.0)     # KP_NOSE
    assert coco_conf[5] == pytest.approx(0.9)


def test_unmapped_joints_get_zero_confidence_not_a_guess():
    xy = np.ones((BLAZEPOSE_NUM_LANDMARKS, 2), dtype=np.float32)
    vis = np.ones((BLAZEPOSE_NUM_LANDMARKS,), dtype=np.float32)
    _, conf = blazepose_to_coco(xy, vis)
    for idx in UNMAPPED_COCO_INDICES:
        assert conf[idx] == 0.0


def test_remap_rejects_a_wrong_landmark_count():
    with pytest.raises(ValueError):
        blazepose_to_coco(np.zeros((33, 2), np.float32), np.zeros((33,), np.float32))


def test_empty_keypoints_read_as_fully_occluded():
    xy, conf = empty_coco_keypoints()
    assert xy.shape == (17, 2)
    assert conf.shape == (17,)
    assert not conf.any()


# -- BlazePose SSD ----------------------------------------------------------


def test_anchor_counts_match_the_declared_output_shapes():
    """512 + 384 == 896, exactly the box_scores_1 / box_scores_2 split."""
    centres = ssd_anchor_centres(DETECTOR_HEADS)
    assert [len(c) for c in centres] == [512, 384]
    assert sum(len(c) for c in centres) == 896


def test_anchor_centres_are_normalised_and_grid_aligned():
    head0 = ssd_anchor_centres(DETECTOR_HEADS)[0]
    assert head0.min() > 0.0 and head0.max() < 1.0
    # anchors_per_cell == 2, so the first two share a centre
    np.testing.assert_allclose(head0[0], head0[1])
    np.testing.assert_allclose(head0[0], [0.5 / 16, 0.5 / 16], atol=1e-6)


def test_sigmoid_is_stable_for_large_negative_logits():
    assert sigmoid(np.array([-1e4], dtype=np.float32))[0] == pytest.approx(0.0)
    assert sigmoid(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.5)


def test_detector_decode_round_trips_a_known_box():
    centres = ssd_anchor_centres(DETECTOR_HEADS)[0]
    n = len(centres)
    input_size = 128
    anchor = 272
    want_cx, want_cy, want_w, want_h = 0.5, 0.5, 0.9, 0.9

    coords = np.zeros((n, 12), dtype=np.float32)
    coords[anchor, 0] = (want_cx - centres[anchor, 0]) * input_size
    coords[anchor, 1] = (want_cy - centres[anchor, 1]) * input_size
    coords[anchor, 2] = want_w * input_size
    coords[anchor, 3] = want_h * input_size
    logits = np.full((n, 1), -60.0, dtype=np.float32)
    logits[anchor, 0] = 0.0

    boxes, scores, align = decode_detector_head(logits, coords, centres, input_size)
    assert boxes.shape == (n, 4)
    assert align.shape == (n, 4, 2)
    np.testing.assert_allclose(
        boxes[anchor],
        [want_cx - want_w / 2, want_cy - want_h / 2, want_cx + want_w / 2, want_cy + want_h / 2],
        atol=1e-5,
    )
    assert scores[anchor] == pytest.approx(0.5)
    assert scores.max() == pytest.approx(0.5)


def test_detector_decode_rejects_an_anchor_count_mismatch():
    centres = ssd_anchor_centres(DETECTOR_HEADS)[0]
    with pytest.raises(ValueError, match="anchor count"):
        decode_detector_head(
            np.zeros((10, 1), np.float32), np.zeros((10, 12), np.float32), centres, 128
        )


def test_landmark_decode_drops_z_and_clips_visibility():
    lm = np.zeros((25, 4), dtype=np.float32)
    lm[:, 0] = 0.25
    lm[:, 1] = 0.75
    lm[:, 2] = -5.0   # z, must be ignored
    lm[:, 3] = 1.4    # out-of-range visibility, must be clipped
    xy, vis = decode_landmarks(lm)
    assert xy.shape == (25, 2)
    np.testing.assert_allclose(xy[0], [0.25, 0.75])
    assert vis.max() == pytest.approx(1.0)
