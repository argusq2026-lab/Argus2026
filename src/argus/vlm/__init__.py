"""VLM scene understanding — sampled, never streamed.

The VLM sees only trainees the deterministic scorer has already flagged, at
most `vlm.top_k` per frame per camera, at most once per
`vlm.sample_interval_s` each. That gate is the whole reason the per-frame
latency budget holds: the vision head is ~4.2 ms of NPU time per trainee lane,
and a VLM is orders of magnitude more than that.

Whatever the VLM emits, it emits into :func:`argus.triage.score_vlm_caption`,
which matches it against a closed vocabulary and returns a number. The caption
string itself never leaves the calling frame — it is not stored on the track,
not logged, and structurally cannot reach a sink, because every sink's type
signature accepts only :class:`~argus.triage.TriageRecord`. That is what keeps
the rank deterministic despite non-deterministic decoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from argus.config import VLMConfig


@runtime_checkable
class VLMCaptioner(Protocol):
    """Turns a crop into a caption. The caption is scored, then discarded."""

    def caption(self, crop_bgr: np.ndarray) -> str: ...

    def close(self) -> None: ...


class MockVLMCaptioner:
    """Deterministic stand-in used until a Genie bundle is staged.

    Returns a phrase containing no vocabulary term, so `vlm_anomaly` scores
    0.0 and the rank is driven entirely by the pose/motion features. It is
    clearly labelled and it never fabricates an anomaly it did not see.
    """

    def caption(self, crop_bgr: np.ndarray) -> str:
        return "[mock-vlm] trainee at station; no anomaly vocabulary asserted"

    def close(self) -> None:
        pass


class GenieVLMCaptioner:
    """Real captioner backed by an on-device Genie / onnxruntime-genai bundle.

    Constructed only when `vlm.kind = "genie"` and `vlm.bundle_dir` points at a
    staged bundle. Building that bundle needs a non-arm64 host (qai-hub-models
    pulls torch, which has no win-arm64 wheel) — see `scripts/fetch_vlm.py` and
    the `quad-build-npu-bundle` skill.
    """

    def __init__(self, cfg: VLMConfig):
        bundle = Path(cfg.bundle_dir)
        if not cfg.bundle_dir or not bundle.is_dir():
            raise FileNotFoundError(
                "vlm.kind = 'genie' requires vlm.bundle_dir to point at a staged "
                f"bundle; got {cfg.bundle_dir!r}. Build one with "
                "scripts/fetch_vlm.py, or set vlm.kind = 'mock'."
            )
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-genai is required for vlm.kind = 'genie'. It is a "
                "win-arm64 wheel and lives in the .venv-npu created by "
                "`run.ps1 -Npu`, not in the core venv."
            ) from exc

        self._og = og
        self._model = og.Model(str(bundle))
        self._processor = self._model.create_multimodal_processor()
        self._tokenizer_stream = self._processor.create_stream()
        self._prompt = cfg.prompt

    def caption(self, crop_bgr: np.ndarray) -> str:
        og = self._og
        rgb = np.ascontiguousarray(crop_bgr[..., ::-1])
        images = og.Images.open_bytes(rgb.tobytes())
        inputs = self._processor(self._prompt, images=images)

        params = og.GeneratorParams(self._model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=64, do_sample=False)  # greedy: reproducible

        generator = og.Generator(self._model, params)
        chunks: list[str] = []
        while not generator.is_done():
            generator.generate_next_token()
            chunks.append(self._tokenizer_stream.decode(generator.get_next_tokens()[0]))
        return "".join(chunks).strip()

    def close(self) -> None:
        self._model = None


def build_captioner(cfg: VLMConfig) -> VLMCaptioner:
    """Construct the configured captioner. Never silently downgrades to mock."""
    if cfg.kind == "mock":
        return MockVLMCaptioner()
    if cfg.kind == "genie":
        return GenieVLMCaptioner(cfg)
    raise ValueError(f"unknown vlm.kind: {cfg.kind!r}")


__all__ = [
    "VLMCaptioner",
    "MockVLMCaptioner",
    "GenieVLMCaptioner",
    "build_captioner",
]
