"""Versioned configuration for Argus.

Every tunable — scoring weights, thresholds, model paths, camera sources, VLM
cadence and top-K — lives in a TOML file, not in code. Retuning is a config
edit plus a `config_version` bump; it is never a source change, so a tuning
decision is reviewable and diffable on its own.

`config_version` is validated on load. A config written against an older
schema fails loudly rather than silently falling back to defaults, because a
silently-defaulted weight would change who an instructor is sent to.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1

#: Shipped default config. Present both in a source checkout (repo-root
#: ``configs/``) and in an installed wheel (``argus/_data/``).
_PACKAGED_DEFAULT = Path(__file__).resolve().parent / "_data" / "argus.default.toml"
_REPO_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "argus.default.toml"


class ConfigError(ValueError):
    """Raised for a malformed, mis-versioned, or internally inconsistent config."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringConfig:
    """Everything the deterministic scorer reads. Frozen: a scorer must not be
    able to mutate its own tuning mid-run, or the rank stops being reproducible."""

    weights: dict[str, float]
    anomaly_vocab: dict[str, float]
    alert_threshold: float = 0.5
    history_len: int = 30
    keypoint_conf_threshold: float = 0.3
    stillness_motion_threshold_px: float = 4.0
    #: Shoulder-line angle, in degrees, that means "facing the station".
    #: 180 -- not 0 -- is the camera-facing value: COCO and MediaPipe both
    #: label joints from the *subject's* perspective, so a person facing the
    #: camera has their left shoulder at the larger image x, and
    #: atan2(rs.y - ls.y, rs.x - ls.x) is 180 degrees.
    off_task_reference_angle_deg: float = 180.0

    REQUIRED_WEIGHTS = ("fall", "stillness", "occlusion", "off_task", "vlm_anomaly")

    def __post_init__(self) -> None:
        missing = [k for k in self.REQUIRED_WEIGHTS if k not in self.weights]
        if missing:
            raise ConfigError(f"[scoring.weights] missing required weight(s): {missing}")
        extra = [k for k in self.weights if k not in self.REQUIRED_WEIGHTS]
        if extra:
            raise ConfigError(
                f"[scoring.weights] unknown weight(s) {extra}: a weight the scorer "
                "does not read would silently do nothing"
            )
        if any(w < 0 for w in self.weights.values()):
            raise ConfigError("[scoring.weights] weights must be non-negative")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"[scoring.weights] must sum to 1.0 (got {total:.6f}); otherwise "
                "`score` is not comparable against `alert_threshold`"
            )
        if self.history_len < 2:
            raise ConfigError("[scoring] history_len must be >= 2")
        if not 0.0 <= self.alert_threshold <= 1.0:
            raise ConfigError("[scoring] alert_threshold must be in [0, 1]")
        if any(not 0.0 <= v <= 1.0 for v in self.anomaly_vocab.values()):
            raise ConfigError("[scoring.anomaly_vocab] weights must be in [0, 1]")


@dataclass(frozen=True)
class EngineConfig:
    kind: str = "mock"
    allow_cpu_fallback: bool = False
    cache_dir: str = ".qnn_cache"
    context_binary_mode: str = "epcontext"
    qairt_sdk_root: str = ""
    #: ONNX Runtime graph optimization level. See argus.engines.onnx_common:
    #: `extended` triggers an ORT QDQ-fusion node-name collision on the shipped
    #: QuickSRNet graph, which is retried at `basic` with a warning.
    graph_optimization_level: str = "extended"

    KINDS = ("mock", "onnx-cpu", "qnn-npu")
    CONTEXT_MODES = ("epcontext", "netrun")
    OPTIMIZATION_LEVELS = ("disabled", "basic", "extended", "all")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ConfigError(f"[engine] kind must be one of {self.KINDS}, got {self.kind!r}")
        if self.context_binary_mode not in self.CONTEXT_MODES:
            raise ConfigError(
                f"[engine] context_binary_mode must be one of {self.CONTEXT_MODES}, "
                f"got {self.context_binary_mode!r}"
            )
        if self.graph_optimization_level not in self.OPTIMIZATION_LEVELS:
            raise ConfigError(
                f"[engine] graph_optimization_level must be one of "
                f"{self.OPTIMIZATION_LEVELS}, got {self.graph_optimization_level!r}"
            )


@dataclass(frozen=True)
class ModelsConfig:
    root: str = "models"
    detector: str = ""
    detector_metadata: str = ""
    detector_labels: str = ""
    pose_detector: str = ""
    pose_landmark: str = ""
    pose_metadata: str = ""
    super_res: str = ""
    super_res_metadata: str = ""

    #: Set by `ArgusConfig.resolve_paths` so relative model paths resolve
    #: against the config file's directory rather than the process CWD.
    base_dir: Path = Path(".")

    def path(self, name: str) -> Path:
        """Absolute path to a named artifact. Raises if the key is unset."""
        rel = getattr(self, name, "")
        if not rel:
            raise ConfigError(f"[models] {name} is not configured")
        root = Path(self.root)
        if not root.is_absolute():
            root = self.base_dir / root
        return (root / rel).resolve()


@dataclass(frozen=True)
class DetectorConfig:
    score_threshold: float = 0.35
    nms_iou_threshold: float = 0.45
    person_class_index: int = 0
    max_detections: int = 64
    letterbox_pad_value: int = 114


@dataclass(frozen=True)
class PoseConfig:
    detector_score_threshold: float = 0.4
    detector_nms_iou_threshold: float = 0.3
    roi_scale: float = 1.25
    landmark_presence_threshold: float = 0.5
    #: The BlazePose detector emits pre-sigmoid logits. Exposed rather than
    #: assumed, because metadata.json declares no value_range for these two
    #: outputs and a re-export could plausibly bake the sigmoid in.
    detector_scores_are_logits: bool = True
    #: Whether the landmark tensor's 4th channel needs a sigmoid. The shipped
    #: export appears to apply it already (its quantized range is ~[0, 0.88]),
    #: but that is inference from the quantization parameters, not a
    #: measurement — see docs/VALIDATION.md.
    landmark_visibility_is_logit: bool = False


@dataclass(frozen=True)
class SuperResConfig:
    enabled: bool = True
    min_bbox_area_frac: float = 0.02


@dataclass(frozen=True)
class TrackingConfig:
    max_age_frames: int = 45
    min_hits: int = 3
    iou_match_threshold: float = 0.2
    appearance_weight: float = 0.35
    appearance_gate: float = 0.45
    appearance_momentum: float = 0.9
    hist_bins: int = 12

    def __post_init__(self) -> None:
        if not 0.0 <= self.appearance_weight <= 1.0:
            raise ConfigError("[tracking] appearance_weight must be in [0, 1]")
        if self.hist_bins < 2:
            raise ConfigError("[tracking] hist_bins must be >= 2")


@dataclass(frozen=True)
class VLMConfig:
    kind: str = "mock"
    sample_interval_s: float = 1.0
    top_k: int = 3
    prefilter_score_threshold: float = 0.35
    bundle_dir: str = ""
    prompt: str = ""

    KINDS = ("mock", "genie")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ConfigError(f"[vlm] kind must be one of {self.KINDS}, got {self.kind!r}")
        if self.top_k < 0:
            raise ConfigError("[vlm] top_k must be >= 0")


@dataclass(frozen=True)
class CameraConfig:
    id: str
    source: str | int
    enabled: bool = True
    reference_angle_deg: float | None = None
    base_dir: Path = Path(".")

    def resolved_source(self) -> str | int:
        """A camera index passes through; a relative file path is resolved
        against the config's directory so `argus run` works from any CWD."""
        if isinstance(self.source, int):
            return self.source
        p = Path(self.source)
        if p.is_absolute():
            return str(p)
        candidate = (self.base_dir / p).resolve()
        return str(candidate) if candidate.exists() else self.source


@dataclass(frozen=True)
class OutputsConfig:
    console: bool = True
    json_log: str = ""
    http_port: int = 0
    http_host: str = "127.0.0.1"
    overlay_window: bool = False
    overlay_out: str = ""


@dataclass(frozen=True)
class ArgusConfig:
    scoring: ScoringConfig
    engine: EngineConfig = field(default_factory=EngineConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    super_res: SuperResConfig = field(default_factory=SuperResConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    cameras: tuple[CameraConfig, ...] = ()
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    source_path: Path | None = None

    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(c for c in self.cameras if c.enabled)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _build(cls, data: dict[str, Any], section: str, **extra: Any):
    """Construct a frozen section dataclass, rejecting unknown keys.

    An unknown key is an error rather than a warning: a typo'd
    ``alert_threshhold`` that silently kept the default would mean an operator
    thinks they retuned the system when they did not.
    """
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"[{section}] unknown key(s): {sorted(unknown)}")
    try:
        return cls(**{**data, **extra})
    except TypeError as exc:
        raise ConfigError(f"[{section}] {exc}") from exc


def default_config_path() -> Path:
    """Path to the shipped default config, wheel or source checkout."""
    if _PACKAGED_DEFAULT.is_file():
        return _PACKAGED_DEFAULT
    if _REPO_DEFAULT.is_file():
        return _REPO_DEFAULT
    raise ConfigError(
        "packaged default config not found; expected one of "
        f"{_PACKAGED_DEFAULT} or {_REPO_DEFAULT}"
    )


def load_config(path: str | Path | None = None) -> ArgusConfig:
    """Load and validate a config file. `None` loads the shipped default."""
    cfg_path = Path(path) if path is not None else default_config_path()
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    version = raw.get("config_version")
    if version is None:
        raise ConfigError(f"{cfg_path}: missing `config_version`")
    if version != CONFIG_VERSION:
        raise ConfigError(
            f"{cfg_path}: config_version {version} is not supported by this build "
            f"(expects {CONFIG_VERSION}). Migrate the file rather than running with "
            "defaults -- a silently-defaulted weight changes the triage rank."
        )

    base_dir = cfg_path.resolve().parent
    # A config in configs/ tunes the repo that contains it, so model and demo
    # paths are written relative to the repo root, not to configs/ itself.
    asset_dir = base_dir.parent if base_dir.name == "configs" else base_dir

    scoring_raw = dict(_section(raw, "scoring"))
    weights = scoring_raw.pop("weights", None)
    vocab = scoring_raw.pop("anomaly_vocab", None)
    if weights is None:
        raise ConfigError("[scoring.weights] is required")
    if vocab is None:
        raise ConfigError("[scoring.anomaly_vocab] is required")
    scoring = _build(
        ScoringConfig,
        scoring_raw,
        "scoring",
        weights={k: float(v) for k, v in weights.items()},
        anomaly_vocab={k.lower(): float(v) for k, v in vocab.items()},
    )

    cameras_raw = raw.get("cameras", [])
    if not isinstance(cameras_raw, list):
        raise ConfigError("[[cameras]] must be an array of tables")
    cameras = tuple(
        _build(CameraConfig, dict(c), "cameras", base_dir=asset_dir) for c in cameras_raw
    )
    ids = [c.id for c in cameras]
    if len(set(ids)) != len(ids):
        raise ConfigError(
            f"[[cameras]] ids must be unique (got {ids}); trainee_id is namespaced "
            "by camera id, so a duplicate would merge two floors into one identity"
        )

    known_top = {
        "config_version", "scoring", "engine", "models", "detector", "pose",
        "super_res", "tracking", "vlm", "cameras", "outputs",
    }
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ConfigError(f"unknown top-level section(s): {sorted(unknown_top)}")

    return ArgusConfig(
        scoring=scoring,
        engine=_build(EngineConfig, _section(raw, "engine"), "engine"),
        models=_build(ModelsConfig, _section(raw, "models"), "models", base_dir=asset_dir),
        detector=_build(DetectorConfig, _section(raw, "detector"), "detector"),
        pose=_build(PoseConfig, _section(raw, "pose"), "pose"),
        super_res=_build(SuperResConfig, _section(raw, "super_res"), "super_res"),
        tracking=_build(TrackingConfig, _section(raw, "tracking"), "tracking"),
        vlm=_build(VLMConfig, _section(raw, "vlm"), "vlm"),
        cameras=cameras,
        outputs=_build(OutputsConfig, _section(raw, "outputs"), "outputs"),
        source_path=cfg_path.resolve(),
    )


def override(cfg: ArgusConfig, **sections: Any) -> ArgusConfig:
    """Return a copy of `cfg` with whole sections replaced.

    Used by the CLI to apply explicit flag overrides on top of a file. Flags
    override config; config never overrides flags.
    """
    return replace(cfg, **sections)
