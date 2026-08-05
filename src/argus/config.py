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

CONFIG_VERSION = 5

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
    #: Per-exercise weight overrides, keyed by the `exercise` field of an
    #: observation. A movement can make a feature meaningless or actively
    #: wrong: a correct plank is horizontal and motionless, which the `fall`
    #: and `stillness` features are built to treat as an emergency. Rather
    #: than special-casing exercises in the scorer, an exercise names its own
    #: full weight vector here and the scorer looks it up. A feature weighted
    #: 0 contributes nothing *and* emits no reason code, so one number both
    #: silences a misfiring signal and keeps it off the trainer's dashboard.
    #: An exercise with no profile scores on `weights`, unchanged.
    exercise_weights: dict[str, dict[str, float]] = field(default_factory=dict)
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

    def _validate_weights(self, weights: dict[str, float], section: str) -> None:
        """Every weight vector, default or per-exercise, obeys the same rules.

        Per-exercise profiles are held to the full contract rather than
        treated as sparse patches: a profile that named only the weights it
        changed would leave the rest implicit, and "what does a plank actually
        score on" would need two places to answer.
        """
        missing = [k for k in self.REQUIRED_WEIGHTS if k not in weights]
        if missing:
            raise ConfigError(f"[{section}] missing required weight(s): {missing}")
        extra = [k for k in weights if k not in self.REQUIRED_WEIGHTS]
        if extra:
            raise ConfigError(
                f"[{section}] unknown weight(s) {extra}: a weight the scorer "
                "does not read would silently do nothing"
            )
        if any(w < 0 for w in weights.values()):
            raise ConfigError(f"[{section}] weights must be non-negative")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"[{section}] must sum to 1.0 (got {total:.6f}); otherwise "
                "`score` is not comparable against `alert_threshold`"
            )

    def weights_for(self, exercise: str | None) -> dict[str, float]:
        """The weight vector for `exercise`, falling back to the default set.

        An unrecognised exercise scores on the default weights rather than
        raising: unlike a form-error code, `exercise` is a free-form label the
        protocol has always described as informational, so a phone reporting
        one this config has no profile for is not a version mismatch.
        """
        if exercise is None:
            return self.weights
        return self.exercise_weights.get(exercise.lower(), self.weights)

    def __post_init__(self) -> None:
        self._validate_weights(self.weights, "scoring.weights")
        for name, profile in self.exercise_weights.items():
            self._validate_weights(profile, f"scoring.exercise_weights.{name}")
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
class SessionConfig:
    """Who is running this floor, and who gets to join it.

    A session is one instructor's floor: a name phones can recognise in a room
    where more than one laptop is running, plus the rule for admitting a phone
    that asks to join.

    `approval` defaults to `"auto"`, which is exactly the behaviour that
    existed before admission was a concept — a well-formed `hello` is
    acknowledged immediately. `"manual"` parks the connection until the
    instructor decides. Auto is the default because the failure modes are not
    symmetric: an unwanted phone on the console is a nuisance an instructor
    can see and disconnect, whereas a trainee standing at a rack unmonitored
    because nobody noticed a prompt is the thing this system exists to
    prevent.
    """

    #: Shown in the beacon and on the console. Free-form; an empty name is
    #: legal and shows as the server's address instead.
    name: str = ""
    approval: str = "auto"
    #: How long an unanswered join request waits before the phone is told to
    #: stop waiting. A prompt nobody is going to answer should end, and end
    #: with a reason, rather than leave a phone hanging indefinitely.
    join_timeout_s: float = 120.0

    APPROVAL_MODES = ("auto", "manual")
    MAX_NAME_LEN = 64

    def __post_init__(self) -> None:
        if self.approval not in self.APPROVAL_MODES:
            raise ConfigError(
                f"[session] approval must be one of {list(self.APPROVAL_MODES)}, "
                f"got {self.approval!r}"
            )
        if len(self.name) > self.MAX_NAME_LEN:
            raise ConfigError(
                f"[session] name must be at most {self.MAX_NAME_LEN} characters"
            )
        if self.join_timeout_s <= 0:
            raise ConfigError("[session] join_timeout_s must be > 0")


@dataclass(frozen=True)
class DiscoveryConfig:
    """The LAN beacon that lets a phone find this laptop without being told.

    Outward-facing, so it is worth being explicit about what enabling it
    does: the laptop broadcasts its own WebSocket address to the local
    network once per `interval_s`. It carries no trainee data — see
    `argus.discovery` — and it advertises a port that `ingest.ws_host =
    "0.0.0.0"` already left open, so it publishes the fact of the service
    rather than any new access to it. Set `enabled = false` on a network
    where announcing the service at all is not wanted; phones can still be
    pointed at the address by hand.
    """

    enabled: bool = True
    port: int = 8766
    interval_s: float = 1.0
    #: Blank uses the limited-broadcast address. Set a subnet-directed
    #: address (e.g. "192.168.1.255") on a network that drops it.
    broadcast: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ConfigError("[discovery] port must be in [0, 65535]")
        if self.interval_s <= 0:
            raise ConfigError("[discovery] interval_s must be > 0")


@dataclass(frozen=True)
class OutputsConfig:
    #: Whether the *stderr alert* sink prints. Distinct from the `console_*`
    #: keys below, which tune the trainer console page served at `GET /`.
    console: bool = True
    json_log: str = ""
    http_port: int = 0
    http_host: str = "127.0.0.1"
    #: How often the trainer console re-reads `GET /console`. Served to the
    #: page in the snapshot rather than baked into it, so the refresh cadence
    #: is a config edit like every other cadence here. The default is well
    #: under a phone's own 5-15 Hz so a skeleton moves rather than steps.
    console_poll_interval_ms: int = 200
    #: How long a station may go silent before the console shows it as stale
    #: rather than calm. This is a display threshold, not an eviction one:
    #: `ingest.track_ttl_s` still decides when a track is actually dropped.
    #: Keep it well under that ttl -- set longer, a station is evicted before
    #: it is ever drawn as stale, so a trainee who went silent leaves the
    #: grid without having been flagged. `argus doctor` warns when it is.
    console_stale_after_s: float = 2.0
    #: Whether `POST /join/decide` is accepted from anywhere but this machine.
    #: Off by default, and checked against the *client's* address rather than
    #: the bind address, so serving the console on a second screen does not
    #: quietly hand the rest of the LAN the power to decide who monitors a
    #: trainee. Turning it on makes admission decisions unauthenticated to
    #: anyone who can reach `http_host`.
    allow_remote_join_control: bool = False

    def __post_init__(self) -> None:
        if self.console_poll_interval_ms <= 0:
            raise ConfigError("[outputs] console_poll_interval_ms must be > 0")
        if self.console_stale_after_s <= 0:
            raise ConfigError("[outputs] console_stale_after_s must be > 0")


@dataclass(frozen=True)
class ArgusConfig:
    scoring: ScoringConfig
    ingest: IngestConfig = field(default_factory=IngestConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
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
    exercise_weights = scoring_raw.pop("exercise_weights", {})
    if weights is None:
        raise ConfigError("[scoring.weights] is required")
    if vocab is None:
        raise ConfigError("[scoring.form_error_vocab] is required")
    if not isinstance(exercise_weights, dict):
        raise ConfigError("[scoring.exercise_weights] must be a table of tables")
    for name, profile in exercise_weights.items():
        if not isinstance(profile, dict):
            raise ConfigError(
                f"[scoring.exercise_weights.{name}] must be a table of weights"
            )
    scoring = _build(
        ScoringConfig,
        scoring_raw,
        "scoring",
        weights={k: float(v) for k, v in weights.items()},
        form_error_vocab={k.lower(): float(v) for k, v in vocab.items()},
        exercise_weights={
            name.lower(): {k: float(v) for k, v in profile.items()}
            for name, profile in exercise_weights.items()
        },
    )

    known_top = {"config_version", "scoring", "ingest", "outputs", "session", "discovery"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ConfigError(f"unknown top-level section(s): {sorted(unknown_top)}")

    return ArgusConfig(
        scoring=scoring,
        ingest=_build(IngestConfig, _section(raw, "ingest"), "ingest"),
        outputs=_build(OutputsConfig, _section(raw, "outputs"), "outputs"),
        session=_build(SessionConfig, _section(raw, "session"), "session"),
        discovery=_build(DiscoveryConfig, _section(raw, "discovery"), "discovery"),
        source_path=cfg_path.resolve(),
    )


def override(cfg: ArgusConfig, **sections: Any) -> ArgusConfig:
    """Return a copy of `cfg` with whole sections replaced.

    Used by the CLI to apply explicit flag overrides on top of a file. Flags
    override config; config never overrides flags.
    """
    return replace(cfg, **sections)
