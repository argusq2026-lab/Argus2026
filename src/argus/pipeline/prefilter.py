"""The VLM prefilter gate.

This is the fix for the prototype's most consequential defect. Its gate read:

    prefilter_score = sum(w for name, w in (("fall", 0.4), ("stillness", 0.2),
                                            ("occlusion", 0.15)))
    if due and prefilter_score > 0:

`prefilter_score` sums a hardcoded literal tuple, so it is 0.75 for every
trainee on every frame — the condition is `if due and True`. The VLM would
therefore be invoked for *every* tracked trainee, not just flagged ones, and
`VLM_PREFILTER_TOP_K` was applied only to the alert print further down, never
to the sampling. The documented latency budget assumes the opposite.

The gate below reads each trainee's actual current triage score, requires it to
clear `vlm.prefilter_score_threshold`, requires the per-trainee cadence to be
due, and then hard-bounds the result to `vlm.top_k` highest-scoring
candidates. It is a pure function of its arguments, so the bound is unit
-testable without a camera, a model, or a VLM.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence

from argus.config import VLMConfig
from argus.triage import TriageRecord


def select_for_vlm(
    records: Sequence[TriageRecord],
    last_sample_ts: Mapping[str, float],
    now: float,
    cfg: VLMConfig,
    eligible_ids: Container[str] | None = None,
) -> list[str]:
    """Trainee ids to caption this frame, highest score first, at most `top_k`.

    Args:
        records: this frame's triage rank.
        last_sample_ts: trainee id -> when it was last captioned.
        now: this frame's timestamp.
        cfg: VLM section of the config.
        eligible_ids: optional restriction to trainees still present on this
            camera, so a record for a since-deleted track cannot be sampled.
    """
    if cfg.top_k <= 0:
        return []

    candidates = [
        record
        for record in records
        if record.score >= cfg.prefilter_score_threshold
        and (eligible_ids is None or record.trainee_id in eligible_ids)
        and now - last_sample_ts.get(record.trainee_id, float("-inf"))
        >= cfg.sample_interval_s
    ]
    # Ties break on trainee_id, matching rank_trainees, so which trainees get
    # the scarce VLM slots is reproducible rather than dict-order dependent.
    candidates.sort(key=lambda r: (-r.score, r.trainee_id))
    return [r.trainee_id for r in candidates[: cfg.top_k]]
