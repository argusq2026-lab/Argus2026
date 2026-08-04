"""Versioned configuration for Argus.

Every tunable — scoring weights, thresholds, and the WebSocket ingest
settings — lives in a TOML file, not in code. Retuning is a config edit plus
a `config_version` bump; it is never a source change, so a tuning decision is
reviewable and diffable on its own.

`config_version` is validated on load. A config written against an older
schema fails loudly rather than silently falling back to defaults, because a
silently-defaulted weight would change who an instructor is sent to.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

CONFIG_VERSION = 2

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
    form_error_vocab: dict[str, float]
    alert_threshold: float = 0.5
    history_len: int = 30
    keypoint_conf_threshold: float = 0.3
    #: bbox_xyxy and keypoints_xy arrive normalized to [0, 1] of the phone's
    #: own frame (see docs/PROTOCOL.md), so "near-zero motion" is a fraction
    #: of frame size, not a pixel count.
    stillness_motion_threshold_frac: float = 0.003
    #: Shoulder-line angle, in degrees, that means "facing the station".
    #: 180 -- not 0 -- is the camera-facing value: COCO and MediaPipe both
    #: label joints from the *subject's* perspective, so a person facing the
    #: camera has their left shoulder at the larger image x, and
    #: atan2(rs.y - ls.y, rs.x - ls.x) is 180 degrees.
    off_task_reference_angle_deg: float = 180.0

    REQUIRED_WEIGHTS = ("fall", "stillness", "occlusion", "off_task", "form_error")

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
        if any(not 0.0 <= v <= 1.0 for v in self.form_error_vocab.values()):
            raise ConfigError("[scoring.form_error_vocab] weights must be in [0, 1]")


@dataclass(frozen=True)
class IngestConfig:
    """The WebSocket ingest server that replaces local camera capture.

    Every trainee's phone opens one connection here and streams `hello` then
    repeated `observation` messages — see `docs/PROTOCOL.md` for the wire
    format. Nothing here ever names a frame type: the server is numbers in,
    numbers out, same as the scorer it feeds.
    """

    ws_host: str = "0.0.0.0"
    #: 0 = let the OS assign a free port (useful for tests; `IngestServer.
    #: ws_port` reports what was actually bound).
    ws_port: int = 8765
    protocol_version: int = 1
    #: How often the merged rank is recomputed and pushed to the alert/output
    #: sinks — decoupled from any single phone's own streaming rate.
    rank_interval_s: float = 0.5
    #: A station silent this long is presumed to have left the floor; its
    #: track is dropped and console alert-suppression state is forgotten.
    track_ttl_s: float = 10.0

    def __post_init__(self) -> None:
        if not 0 <= self.ws_port <= 65535:
            raise ConfigError("[ingest] ws_port must be in [0, 65535]")
        if self.rank_interval_s <= 0:
            raise ConfigError("[ingest] rank_interval_s must be > 0")
        if self.track_ttl_s <= 0:
            raise ConfigError("[ingest] track_ttl_s must be > 0")


@dataclass(frozen=True)
class OutputsConfig:
    console: bool = True
    json_log: str = ""
    http_port: int = 0
    http_host: str = "127.0.0.1"


@dataclass(frozen=True)
class ArgusConfig:
    scoring: ScoringConfig
    ingest: IngestConfig = field(default_factory=IngestConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    source_path: Path | None = None


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

    scoring_raw = dict(_section(raw, "scoring"))
    weights = scoring_raw.pop("weights", None)
    vocab = scoring_raw.pop("form_error_vocab", None)
    if weights is None:
        raise ConfigError("[scoring.weights] is required")
    if vocab is None:
        raise ConfigError("[scoring.form_error_vocab] is required")
    scoring = _build(
        ScoringConfig,
        scoring_raw,
        "scoring",
        weights={k: float(v) for k, v in weights.items()},
        form_error_vocab={k.lower(): float(v) for k, v in vocab.items()},
    )

    known_top = {"config_version", "scoring", "ingest", "outputs"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ConfigError(f"unknown top-level section(s): {sorted(unknown_top)}")

    return ArgusConfig(
        scoring=scoring,
        ingest=_build(IngestConfig, _section(raw, "ingest"), "ingest"),
        outputs=_build(OutputsConfig, _section(raw, "outputs"), "outputs"),
        source_path=cfg_path.resolve(),
    )


def override(cfg: ArgusConfig, **sections: Any) -> ArgusConfig:
    """Return a copy of `cfg` with whole sections replaced.

    Used by the CLI to apply explicit flag overrides on top of a file. Flags
    override config; config never overrides flags.
    """
    return replace(cfg, **sections)
