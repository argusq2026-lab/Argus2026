"""Constant-velocity Kalman filter over a bounding box.

State is SORT's parameterisation — ``[cx, cy, area, aspect, vx, vy, v_area]``
— which models a person walking toward or away from the camera as a smooth
area change rather than as four independently drifting edges.

The filter is what lets a track survive an occlusion: while a trainee is
behind a machine, `predict()` keeps advancing an estimate, so when they emerge
there is still something for the matcher to associate against. A tracker that
only remembers last-seen centroids has nothing to match after a few frames,
which is precisely how the prototype's IDs churned.

No randomness anywhere: same detections in, same state out.
"""

from __future__ import annotations

import numpy as np

_STATE_DIM = 7
_MEAS_DIM = 4


def bbox_to_z(bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """xyxy -> [cx, cy, area, aspect]."""
    x0, y0, x1, y1 = bbox_xyxy
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    return np.array([x0 + w / 2.0, y0 + h / 2.0, w * h, w / h], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> tuple[float, float, float, float]:
    """[cx, cy, area, aspect] -> xyxy, clamped away from degenerate sizes."""
    cx, cy, area, aspect = (float(v) for v in z[:4])
    area = max(area, 1e-6)
    aspect = max(aspect, 1e-6)
    w = float(np.sqrt(area * aspect))
    h = area / max(w, 1e-6)
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


class KalmanBoxFilter:
    """Linear Kalman filter for one box. Units are pixels; dt is one frame."""

    def __init__(self, bbox_xyxy: tuple[float, float, float, float]):
        self.F = np.eye(_STATE_DIM)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((_MEAS_DIM, _STATE_DIM))
        self.H[:_MEAS_DIM, :_MEAS_DIM] = np.eye(_MEAS_DIM)

        # Measurement noise: area and aspect are noisier than the centre,
        # because a partially-occluded box loses extent before it loses centre.
        self.R = np.eye(_MEAS_DIM)
        self.R[2:, 2:] *= 10.0

        # Initial covariance: velocities are entirely unknown at birth.
        self.P = np.eye(_STATE_DIM) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.Q = np.eye(_STATE_DIM)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((_STATE_DIM,), dtype=np.float64)
        self.x[:_MEAS_DIM] = bbox_to_z(bbox_xyxy)

    def predict(self) -> tuple[float, float, float, float]:
        """Advance one frame and return the predicted box."""
        # Area cannot go negative; clamp the velocity that would take it there.
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return z_to_bbox(self.x)

    def update(self, bbox_xyxy: tuple[float, float, float, float]) -> None:
        """Fold in a measured box."""
        z = bbox_to_z(bbox_xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        identity = np.eye(_STATE_DIM)
        self.P = (identity - K @ self.H) @ self.P

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return z_to_bbox(self.x)

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])
