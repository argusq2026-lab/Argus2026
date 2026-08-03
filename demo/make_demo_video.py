"""Render the synthetic multi-trainee demo clip.

The scene comes from :mod:`argus.synthetic`, which the mock backend also reads,
so every rectangle drawn here is a person the mock reports a detection for.
That alignment is load-bearing rather than cosmetic: the tracker identifies a
trainee partly by the colour histogram of their crop, so boxes floating over
unrelated background would fabricate identity churn the real system would not
have.

**This is a pipeline fixture, not an accuracy benchmark.** The shapes are
drawn, not photographed: a real YOLO-X will not fire on them, and no accuracy
claim can be made from a run over this clip. Closing that gap needs real
trainee-floor footage — see docs/VALIDATION.md.

Usage:
    python demo/make_demo_video.py [--out demo/trainees_demo.mp4] [--frames 150]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from argus.synthetic import SCENE_FRAME_SIZE, canvas_to_frame, synthetic_people  # noqa: E402

WIDTH, HEIGHT = SCENE_FRAME_SIZE
FPS = 15
#: Detector canvas edge the scene is defined against (YOLO-X is 640x640).
CANVAS = 640


def render_frame(index: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    cv2.putText(
        frame, "ARGUS DEMO (synthetic)", (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )
    for person in synthetic_people(index):
        x0, y0, x1, y1 = canvas_to_frame(person.xyxy_norm, WIDTH, HEIGHT, CANVAS)
        cv2.rectangle(
            frame,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            person.colour_bgr,
            -1,
        )
    return frame


def make_demo_video(out_path: str | Path, n_frames: int = 150) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")
    try:
        for index in range(n_frames):
            writer.write(render_frame(index))
    finally:
        writer.release()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "trainees_demo.mp4"))
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    path = make_demo_video(args.out, args.frames)
    print(f"Wrote {args.frames} frames to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
