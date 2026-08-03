"""The config is the tuning contract, so its failure modes are tested."""

from __future__ import annotations

import pytest

from argus.config import (
    CONFIG_VERSION,
    ConfigError,
    ScoringConfig,
    default_config_path,
    load_config,
)


def _write(tmp_path, body: str):
    path = tmp_path / "argus.toml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL = f"""
config_version = {CONFIG_VERSION}
[scoring]
[scoring.weights]
fall = 0.4
stillness = 0.2
occlusion = 0.15
off_task = 0.1
vlm_anomaly = 0.15
[scoring.anomaly_vocab]
fall = 1.0
"""


def test_shipped_default_loads():
    cfg = load_config()
    assert cfg.source_path == default_config_path()
    assert pytest.approx(sum(cfg.scoring.weights.values())) == 1.0
    assert cfg.enabled_cameras(), "the default config must ship a runnable camera"


def test_default_reference_angle_is_camera_facing():
    """180, not 0 -- see ScoringConfig.off_task_reference_angle_deg."""
    assert load_config().scoring.off_task_reference_angle_deg == 180.0


def test_minimal_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.engine.kind == "mock"
    assert cfg.cameras == ()


def test_missing_version_is_rejected(tmp_path):
    body = MINIMAL.replace(f"config_version = {CONFIG_VERSION}", "")
    with pytest.raises(ConfigError, match="config_version"):
        load_config(_write(tmp_path, body))


def test_wrong_version_is_rejected(tmp_path):
    body = MINIMAL.replace(f"config_version = {CONFIG_VERSION}", "config_version = 99")
    with pytest.raises(ConfigError, match="not supported"):
        load_config(_write(tmp_path, body))


def test_weights_must_sum_to_one(tmp_path):
    body = MINIMAL.replace("fall = 0.4\n", "fall = 0.9\n", 1)
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(_write(tmp_path, body))


def test_unknown_weight_is_rejected(tmp_path):
    body = MINIMAL.replace("[scoring.anomaly_vocab]", "shouting = 0.0\n[scoring.anomaly_vocab]")
    with pytest.raises(ConfigError, match="unknown weight"):
        load_config(_write(tmp_path, body))


def test_typo_in_a_key_is_rejected_not_defaulted(tmp_path):
    """A silently-defaulted threshold means an operator thinks they retuned
    the system when they did not."""
    body = MINIMAL.replace("[scoring]", "[scoring]\nalert_threshhold = 0.9")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, body))


def test_duplicate_camera_ids_are_rejected(tmp_path):
    body = MINIMAL + """
[[cameras]]
id = "cam0"
source = 0
[[cameras]]
id = "cam0"
source = 1
"""
    with pytest.raises(ConfigError, match="unique"):
        load_config(_write(tmp_path, body))


def test_unknown_engine_kind_is_rejected(tmp_path):
    body = MINIMAL + '\n[engine]\nkind = "tpu"\n'
    with pytest.raises(ConfigError, match="engine"):
        load_config(_write(tmp_path, body))


def test_scoring_config_is_frozen(scoring):
    """A scorer must not be able to retune itself mid-run."""
    with pytest.raises(Exception):
        scoring.alert_threshold = 0.9  # type: ignore[misc]


def test_history_len_floor():
    with pytest.raises(ConfigError, match="history_len"):
        ScoringConfig(
            weights={"fall": 1.0, "stillness": 0.0, "occlusion": 0.0,
                     "off_task": 0.0, "vlm_anomaly": 0.0},
            anomaly_vocab={},
            history_len=1,
        )
