"""Shared fixtures.

The COCO-17 pose fixtures below are carried over from the prototype's suite,
with one correction: a camera-facing trainee has their *left* shoulder at the
larger image x, because COCO labels joints from the subject's perspective. The
prototype's baseline had it the other way round, which is why its
`off_task_reference_angle_deg` default of 0.0 looked correct in tests and would
have flagged every attentive trainee on real keypoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from argus.config import ArgusConfig, ScoringConfig, load_config  # noqa: E402
from argus.triage import FrameObservation, TrackState  # noqa: E402


def _npu_available() -> tuple[bool, str]:
    """Whether a real Hexagon NPU session could actually be created here.

    Auto-detected rather than opt-in, so the `npu` tests run for free on the
    X-Elite and skip with a reason everywhere else — including in the emulated
    x86-64 venv the rest of the suite uses.
    """
    import importlib.util
    import platform

    if platform.machine().lower() not in ("arm64", "aarch64"):
        return False, f"needs a native ARM64 interpreter (this one is {platform.machine()})"
    if importlib.util.find_spec("onnxruntime_qnn") is None:
        return False, "onnxruntime-qnn is not installed (see `run.ps1 -Npu`)"
    return True, ""


def pytest_collection_modifyitems(config, items):
    available, reason = _npu_available()
    if available:
        return
    skip = pytest.mark.skip(reason=f"NPU unavailable: {reason}")
    for item in items:
        if "npu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def default_config() -> ArgusConfig:
    return load_config()


@pytest.fixture
def scoring(default_config: ArgusConfig) -> ScoringConfig:
    return default_config.scoring


def standing_pose_kp_xy(torso_y: float = 140.0) -> list[tuple[float, float]]:
    """An upright, camera-facing COCO-17 pose.

    The shoulder line is level and the subject's left shoulder sits at the
    larger x, so `score_off_task` reads 180 degrees -- the config's
    "facing the station" reference. Override individual indices for a rotated
    or dropped pose; do not edit this baseline.
    """
    kp = [(130.0, 100.0)] * 17
    kp[5] = (150.0, torso_y)          # left shoulder  (image right)
    kp[6] = (110.0, torso_y)          # right shoulder (image left)
    kp[9] = (160.0, torso_y + 40.0)   # left wrist
    kp[10] = (100.0, torso_y + 40.0)  # right wrist
    kp[11] = (145.0, torso_y + 60.0)  # left hip
    kp[12] = (115.0, torso_y + 60.0)  # right hip
    return kp


def make_observation(
    ts: float,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 160.0, 240.0),
    kp_xy: list[tuple[float, float]] | None = None,
    kp_conf: list[float] | None = None,
    vlm_caption: str | None = None,
    torso_y: float = 140.0,
) -> FrameObservation:
    """A well-formed 17-keypoint observation, fully confident by default."""
    return FrameObservation(
        ts=ts,
        bbox_xyxy=bbox,
        keypoints_xy=kp_xy if kp_xy is not None else standing_pose_kp_xy(torso_y),
        keypoints_conf=kp_conf if kp_conf is not None else [0.9] * 17,
        vlm_caption=vlm_caption,
    )


@pytest.fixture
def track_state(scoring: ScoringConfig) -> TrackState:
    return TrackState(history_len=scoring.history_len)


@pytest.fixture(scope="session")
def demo_video(tmp_path_factory) -> Path:
    """A short synthetic clip, generated fresh so no repo asset is required."""
    from demo.make_demo_video import make_demo_video

    out = tmp_path_factory.mktemp("demo") / "demo.mp4"
    return make_demo_video(out, n_frames=40)


@pytest.fixture(scope="session")
def models_root(default_config: ArgusConfig) -> Path:
    root = Path(default_config.models.root)
    if not root.is_absolute():
        root = default_config.models.base_dir / root
    return root
