"""The config is the tuning contract, so its failure modes are tested."""

from __future__ import annotations

import pytest

from argus.config import (
    CONFIG_VERSION,
    ConfigError,
    IngestConfig,
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
form_error = 0.15
[scoring.form_error_vocab]
knee_valgus = 0.8
"""


def test_shipped_default_loads():
    cfg = load_config()
    assert cfg.source_path == default_config_path()
    assert pytest.approx(sum(cfg.scoring.weights.values())) == 1.0


def test_default_reference_angle_is_camera_facing():
    """180, not 0 -- see ScoringConfig.off_task_reference_angle_deg."""
    assert load_config().scoring.off_task_reference_angle_deg == 180.0


def test_minimal_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.ingest.ws_port == 8765
    assert cfg.outputs.http_port == 0


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
    body = MINIMAL.replace(
        "[scoring.form_error_vocab]", "shouting = 0.0\n[scoring.form_error_vocab]"
    )
    with pytest.raises(ConfigError, match="unknown weight"):
        load_config(_write(tmp_path, body))


def test_typo_in_a_key_is_rejected_not_defaulted(tmp_path):
    """A silently-defaulted threshold means an operator thinks they retuned
    the system when they did not."""
    body = MINIMAL.replace("[scoring]", "[scoring]\nalert_threshhold = 0.9")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, body))


def test_unknown_top_level_section_is_rejected(tmp_path):
    body = MINIMAL + '\n[cameras]\nid = "cam0"\n'
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(_write(tmp_path, body))


def test_unknown_ingest_key_is_rejected(tmp_path):
    body = MINIMAL + "\n[ingest]\nws_prot = 1\n"
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, body))


def test_ws_port_out_of_range_is_rejected(tmp_path):
    body = MINIMAL + "\n[ingest]\nws_port = 70000\n"
    with pytest.raises(ConfigError, match="ws_port"):
        load_config(_write(tmp_path, body))


def test_ws_port_zero_is_allowed_for_os_assignment():
    IngestConfig(ws_port=0)


def test_rank_interval_must_be_positive(tmp_path):
    body = MINIMAL + "\n[ingest]\nrank_interval_s = 0\n"
    with pytest.raises(ConfigError, match="rank_interval_s"):
        load_config(_write(tmp_path, body))


def test_scoring_config_is_frozen(scoring):
    """A scorer must not be able to retune itself mid-run."""
    with pytest.raises(Exception):
        scoring.alert_threshold = 0.9  # type: ignore[misc]


def test_session_use_case_defaults_to_fitness(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.session.use_case == "fitness"


@pytest.mark.parametrize("use_case", ["welding", "nursing"])
def test_session_use_case_accepts_a_known_use_case(tmp_path, use_case):
    body = MINIMAL + f'\n[session]\nuse_case = "{use_case}"\n'
    cfg = load_config(_write(tmp_path, body))
    assert cfg.session.use_case == use_case


def test_session_use_case_rejects_one_no_scorer_implements(tmp_path):
    """A typo, or a use case that sounds plausible but was never wired up,
    fails at startup rather than accepting every phone's hello and never
    scoring any of them.

    `"lab"` is deliberate: the phone's dashboard already offers it as a tile,
    so it is exactly the kind of plausible-but-unwired name an operator would
    reach for."""
    body = MINIMAL + '\n[session]\nuse_case = "lab"\n'
    with pytest.raises(ConfigError, match="not implemented"):
        load_config(_write(tmp_path, body))


def test_session_use_case_rejects_empty_string(tmp_path):
    body = MINIMAL + '\n[session]\nuse_case = ""\n'
    with pytest.raises(ConfigError, match="use_case"):
        load_config(_write(tmp_path, body))


def test_history_len_floor():
    with pytest.raises(ConfigError, match="history_len"):
        ScoringConfig(
            weights={"fall": 1.0, "stillness": 0.0, "occlusion": 0.0,
                     "off_task": 0.0, "form_error": 0.0},
            form_error_vocab={},
            history_len=1,
        )
