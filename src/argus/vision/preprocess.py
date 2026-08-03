"""CPU pre-processing: letterbox, layout conversion, crop extraction.

These are the operations that stay off the NPU graph by design. Not because
they were "killed" as bottlenecks — the real per-layer profile shows all four
graphs run 100% on the NPU with nothing evicted — but because they are frame
marshalling, which is CPU work by definition.

Two contract details the prototype got wrong, both of which produce
plausible-looking garbage rather than an error:

* **Layout.** YOLO-X and QuickSRNet take **NCHW**; the two BlazePose context
  binaries take **NHWC**. There is no single layout for the pipeline.
* **Channel order.** Every artifact was exported expecting **RGB**. OpenCV
  hands us BGR. The conversion happens here, at the model boundary, so a
  frame is BGR everywhere it is drawn on and RGB only where it is inferred on.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Padding used by YOLO-X's own letterbox. Configurable per-model, since
#: BlazePose ROI extraction pads with black instead.
YOLOX_PAD_VALUE = 114


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV capture order -> the order every exported artifact expects."""
    return image[..., ::-1]


def letterbox(
    frame: np.ndarray, size: int, pad_value: int = YOLOX_PAD_VALUE
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Aspect-preserving resize into a centred square canvas.

    Returns ``(canvas_hwc_uint8, scale, (left, top))``. The canvas keeps the
    input's channel order — conversion to RGB and to NCHW happens in
    :func:`to_nchw_uint8`, so this stays a pure geometry operation that the
    unit tests can reason about.

    ``scale`` and ``(left, top)`` are everything needed to map a box predicted
    in canvas space back to source pixels; see :func:`undo_letterbox`.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"letterbox expects an HxWx3 image, got shape {frame.shape}")
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def undo_letterbox(
    boxes_xyxy: np.ndarray, scale: float, offset: tuple[int, int]
) -> np.ndarray:
    """Map boxes from letterboxed canvas space back to source-frame pixels."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32)
    left, top = offset
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] -= left
    out[:, [1, 3]] -= top
    return out / max(scale, 1e-9)


def to_nchw_uint8(image_hwc: np.ndarray, *, rgb: bool = True) -> np.ndarray:
    """HWC (BGR) -> (1, 3, H, W) uint8, optionally converting to RGB."""
    img = bgr_to_rgb(image_hwc) if rgb else image_hwc
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.uint8)


def to_nhwc_uint8(image_hwc: np.ndarray, *, rgb: bool = True) -> np.ndarray:
    """HWC (BGR) -> (1, H, W, 3) uint8, optionally converting to RGB."""
    img = bgr_to_rgb(image_hwc) if rgb else image_hwc
    return np.ascontiguousarray(img[np.newaxis, ...], dtype=np.uint8)


def from_nchw_uint8(tensor: np.ndarray, *, rgb: bool = True) -> np.ndarray:
    """(1, 3, H, W) uint8 -> HWC, converting RGB back to OpenCV's BGR."""
    img = tensor[0].transpose(1, 2, 0)
    return np.ascontiguousarray(bgr_to_rgb(img) if rgb else img, dtype=np.uint8)


def resize_exact(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to exactly (width, height). Used where a graph has no dynamic axis."""
    w, h = size
    if image.shape[1] == w and image.shape[0] == h:
        return image
    interp = cv2.INTER_AREA if (image.shape[0] > h or image.shape[1] > w) else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interp)


def clamp_box(
    bbox_xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Integer, in-bounds, non-empty crop bounds for a float box."""
    x0, y0, x1, y1 = bbox_xyxy
    x0 = int(max(0, min(round(x0), width - 1)))
    y0 = int(max(0, min(round(y0), height - 1)))
    x1 = int(max(x0 + 1, min(round(x1), width)))
    y1 = int(max(y0 + 1, min(round(y1), height)))
    return x0, y0, x1, y1


def crop(frame: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Extract a clamped crop. Returns an empty array for a degenerate box."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = clamp_box(bbox_xyxy, w, h)
    return frame[y0:y1, x0:x1]


def square_roi(
    bbox_xyxy: tuple[float, float, float, float], scale: float
) -> tuple[float, float, float, float]:
    """Expand a box to a centred square, scaled by `scale`.

    BlazePose's landmark stage expects a square ROI slightly larger than the
    person box; feeding it a raw aspect-distorted crop shifts every landmark.
    """
    x0, y0, x1, y1 = bbox_xyxy
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * scale / 2.0
    return cx - side, cy - side, cx + side, cy + side


def crop_padded(
    frame: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Crop a possibly out-of-frame box, zero-padding the outside.

    Returns the crop and the box actually represented by it — which equals the
    requested box, so landmark coordinates normalised to the crop map straight
    back to frame space even when the ROI hangs off an edge.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in bbox_xyxy)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    out = np.zeros((y1 - y0, x1 - x0, frame.shape[2]), dtype=frame.dtype)

    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out, (float(x0), float(y0), float(x1), float(y1))
