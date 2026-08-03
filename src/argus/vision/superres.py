"""Super-resolution — QuickSRNet-Medium w8a8, a fixed 128 -> 512 4x upscale.

The graph has no dynamic axis: a crop must be resized to exactly 128x128
first, and the output is always 512x512. Layout is NCHW, matching the
detector and unlike the two pose binaries.

It is applied only to crops below `min_bbox_area_frac` of the frame — a
distant trainee whose 40-pixel-tall crop would otherwise give the 256x256
landmark stage nothing to work with. Upscaling a crop that is already large
costs 632 us of NPU time for no information gain.

Cost note from the real profile (job jg9d81zm5): 632 us / 2,418,643 cycles,
of which the single ``DepthToSpace`` pixel-shuffle is 1,796,483 — **74.3%**.
That is inherent to the architecture's final upsample, not a mis-placement,
and it is why super-res is gated rather than always-on.
"""

from __future__ import annotations

import numpy as np

from argus.config import SuperResConfig
from argus.engines.base import ModelRunner
from argus.vision.preprocess import from_nchw_uint8, resize_exact, to_nchw_uint8


class SuperResolver:
    """Conditionally upscales a small crop before the pose stage."""

    def __init__(self, runner: ModelRunner, cfg: SuperResConfig):
        self._runner = runner
        self._cfg = cfg
        spec = runner.spec
        self._input = spec.input()
        if len(self._input.shape) != 4 or self._input.shape[1] != 3:
            raise ValueError(
                f"{spec.file_name}: expected NCHW (1, 3, H, W) input, got {self._input.shape}"
            )
        self._in_hw = (int(self._input.shape[2]), int(self._input.shape[3]))
        self._output = spec.outputs[0]

    @property
    def input_hw(self) -> tuple[int, int]:
        return self._in_hw

    def should_upscale(self, crop: np.ndarray, frame_area: float) -> bool:
        if not self._cfg.enabled or crop.size == 0:
            return False
        area_frac = (crop.shape[0] * crop.shape[1]) / max(frame_area, 1.0)
        return area_frac < self._cfg.min_bbox_area_frac

    def upscale(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Run the 4x upscale unconditionally. Returns a BGR HWC image."""
        h, w = self._in_hw
        resized = resize_exact(crop_bgr, (w, h))
        tensor = to_nchw_uint8(resized, rgb=True)
        outputs = self._runner.run({self._input.name: tensor})
        return from_nchw_uint8(outputs[self._output.name], rgb=True)

    def maybe_upscale(self, crop_bgr: np.ndarray, frame_area: float) -> np.ndarray:
        """Upscale only if the crop is small enough to be worth it."""
        if not self.should_upscale(crop_bgr, frame_area):
            return crop_bgr
        return self.upscale(crop_bgr)

    def close(self) -> None:
        self._runner.close()
