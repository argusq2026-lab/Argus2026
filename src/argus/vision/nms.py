"""CPU non-maximum suppression.

Carried over verbatim from the Argus prototype, where it was covered by unit
tests. It stays on the CPU because the exported YOLO-X graph simply does not
contain NMS: the artifact ends at decoded `boxes` / `scores` / `class_idx`, so
suppression is the caller's job. (The prototype's claim that NMS was "killed
from the NPU graph as a bottleneck" was fiction — the real per-layer profile
contains no NonMaxSuppression node at all.)
"""

from __future__ import annotations

import numpy as np


def nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS. Returns indices into `boxes`, highest score first."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b)). Used by the tracker's matcher."""
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float32)
    a = a_xyxy.astype(np.float32)[:, None, :]
    b = b_xyxy.astype(np.float32)[None, :, :]
    xx0 = np.maximum(a[..., 0], b[..., 0])
    yy0 = np.maximum(a[..., 1], b[..., 1])
    xx1 = np.minimum(a[..., 2], b[..., 2])
    yy1 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
    area_a = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    area_b = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return (inter / np.maximum(area_a + area_b - inter, 1e-9)).astype(np.float32)
