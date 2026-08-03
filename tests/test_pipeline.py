"""Multi-camera wiring, the VLM gate in situ, and the CLI end-to-end."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from argus.config import CameraConfig
from argus.pipeline.runner import ArgusPipeline, FrameClock

REPO_ROOT = Path(__file__).resolve().parent.parent


class CountingCaptioner:
    """Records how often the VLM was actually invoked, and on what."""

    def __init__(self, caption: str = "[test] nothing unusual"):
        self.calls = 0
        self.crop_shapes: list[tuple[int, ...]] = []
        self._caption = caption

    def caption(self, crop_bgr: np.ndarray) -> str:
        self.calls += 1
        self.crop_shapes.append(crop_bgr.shape)
        return self._caption

    def close(self) -> None:
        pass


def _cfg(default_config, tmp_path, sources: list[str], **outputs):
    cameras = tuple(
        CameraConfig(id=f"cam{i}", source=src, enabled=True, base_dir=REPO_ROOT)
        for i, src in enumerate(sources)
    )
    return dataclasses.replace(
        default_config,
        cameras=cameras,
        outputs=dataclasses.replace(
            default_config.outputs,
            console=False,
            json_log=outputs.get("json_log", ""),
            http_port=outputs.get("http_port", 0),
            overlay_out=outputs.get("overlay_out", ""),
            overlay_window=False,
        ),
    )


# -- single camera ----------------------------------------------------------


@pytest.mark.timeout(180)
def test_pipeline_ranks_trainees_over_the_demo_clip(default_config, demo_video, tmp_path):
    pipeline = ArgusPipeline(
        _cfg(default_config, tmp_path, [str(demo_video)]),
        clock=FrameClock(15.0),
        headless=True,
    )
    try:
        results = [pipeline.tick() for _ in range(35)]
    finally:
        pipeline.close()

    assert results[0].frames_read == 1
    final = results[-1].records
    assert len(final) == 3, "the mock scene has three trainees"
    assert all(r.trainee_id.startswith("cam0-") for r in final)
    assert [r.score for r in final] == sorted((r.score for r in final), reverse=True)


@pytest.mark.timeout(180)
def test_a_fall_is_detected_and_alerted(default_config, demo_video, tmp_path):
    """The synthetic scene drops one trainee at frame 20; the rank must react."""
    pipeline = ArgusPipeline(
        _cfg(default_config, tmp_path, [str(demo_video)]),
        clock=FrameClock(15.0),
        headless=True,
    )
    try:
        alerts = []
        for _ in range(35):
            alerts.extend(pipeline.tick().alerts)
    finally:
        pipeline.close()

    assert alerts, "a fall in the scene produced no alert at all"
    assert any("possible_fall" in a.reason_codes for a in alerts)


# -- multi camera -----------------------------------------------------------


@pytest.mark.timeout(240)
def test_two_cameras_merge_into_one_global_rank(default_config, demo_video, tmp_path):
    pipeline = ArgusPipeline(
        _cfg(default_config, tmp_path, [str(demo_video), str(demo_video)]),
        clock=FrameClock(15.0),
        headless=True,
    )
    try:
        result = None
        for _ in range(30):
            result = pipeline.tick()
    finally:
        pipeline.close()

    assert result.frames_read == 2
    prefixes = [r.trainee_id.split("-")[0] for r in result.records]
    assert set(prefixes) == {"cam0", "cam1"}, "both sources must appear in one rank"
    # Three trainees per camera and no more: a shared mock runner would
    # interleave the two cameras' scenes and manufacture extra identities.
    assert prefixes.count("cam0") == 3
    assert prefixes.count("cam1") == 3
    scores = [r.score for r in result.records]
    assert scores == sorted(scores, reverse=True), "the merged rank must be ordered"


@pytest.mark.timeout(240)
def test_each_camera_keeps_its_own_tracker_state(default_config, demo_video, tmp_path):
    pipeline = ArgusPipeline(
        _cfg(default_config, tmp_path, [str(demo_video), str(demo_video)]),
        clock=FrameClock(15.0),
        headless=True,
    )
    try:
        for _ in range(20):
            pipeline.tick()
        ids_per_camera = [set(cam.tracker.tracks) for cam in pipeline.cameras]
    finally:
        pipeline.close()

    assert ids_per_camera[0] and ids_per_camera[1]
    assert not (ids_per_camera[0] & ids_per_camera[1]), "ids must not collide"


# -- the VLM gate, in the running pipeline ----------------------------------


@pytest.mark.timeout(180)
def test_the_vlm_is_not_called_for_every_trainee(default_config, demo_video, tmp_path):
    """The prototype's dead gate would have called it once per trainee per
    cadence window. The real gate calls it only for flagged trainees."""
    captioner = CountingCaptioner()
    cfg = _cfg(default_config, tmp_path, [str(demo_video)])
    pipeline = ArgusPipeline(
        cfg, captioner=captioner, clock=FrameClock(15.0), headless=True
    )
    try:
        ticks = 35
        for _ in range(ticks):
            pipeline.tick()
        tracked = len(pipeline.cameras[0].tracker.tracks)
    finally:
        pipeline.close()

    assert tracked == 3
    ungated = tracked * (ticks / cfg.vlm.sample_interval_s / 15.0)
    assert captioner.calls <= ungated, "the gate did not reduce VLM calls at all"


@pytest.mark.timeout(180)
def test_vlm_calls_respect_top_k_per_tick(default_config, demo_video, tmp_path):
    """Force every trainee over the threshold; the bound must still hold."""
    base = _cfg(default_config, tmp_path, [str(demo_video)])
    cfg = dataclasses.replace(
        base,
        vlm=dataclasses.replace(
            base.vlm, prefilter_score_threshold=0.0, sample_interval_s=0.0, top_k=1
        ),
    )
    captioner = CountingCaptioner()
    pipeline = ArgusPipeline(
        cfg, captioner=captioner, clock=FrameClock(15.0), headless=True
    )
    try:
        per_tick = [pipeline.tick().vlm_calls for _ in range(30)]
    finally:
        pipeline.close()

    assert max(per_tick) <= 1, f"top_k=1 violated: {max(per_tick)} calls in one tick"
    assert captioner.calls > 0, "the gate blocked everything, so it proves nothing"


@pytest.mark.timeout(180)
def test_a_vocabulary_caption_raises_the_score(default_config, demo_video, tmp_path):
    base = _cfg(default_config, tmp_path, [str(demo_video)])
    cfg = dataclasses.replace(
        base,
        vlm=dataclasses.replace(base.vlm, prefilter_score_threshold=0.0, sample_interval_s=0.0),
    )

    def top_score(caption: str) -> float:
        pipeline = ArgusPipeline(
            cfg,
            captioner=CountingCaptioner(caption),
            clock=FrameClock(15.0),
            headless=True,
        )
        try:
            result = None
            for _ in range(30):
                result = pipeline.tick()
            return max(r.score for r in result.records)
        finally:
            pipeline.close()

    assert top_score("a trainee is unresponsive on the floor") > top_score("all normal")


# -- sinks in situ ----------------------------------------------------------


@pytest.mark.timeout(180)
def test_json_log_gets_one_line_per_tick(default_config, demo_video, tmp_path):
    log = tmp_path / "triage.jsonl"
    pipeline = ArgusPipeline(
        _cfg(default_config, tmp_path, [str(demo_video)], json_log=str(log)),
        clock=FrameClock(15.0),
        headless=True,
    )
    try:
        for _ in range(20):
            pipeline.tick()
    finally:
        pipeline.close()

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 20
    for line in lines:
        payload = json.loads(line)
        for record in payload["records"]:
            assert set(record) == {"trainee_id", "score", "reason_codes", "ts"}
            assert 0.0 <= record["score"] <= 1.0 + 1e-9


@pytest.mark.timeout(240)
def test_multi_camera_overlays_go_to_separate_files(default_config, demo_video, tmp_path):
    out = tmp_path / "overlay.mp4"
    pipeline = ArgusPipeline(
        _cfg(default_config, tmp_path, [str(demo_video), str(demo_video)],
             overlay_out=str(out)),
        clock=FrameClock(15.0),
        headless=True,
    )
    try:
        for _ in range(10):
            pipeline.tick()
    finally:
        pipeline.close()

    assert (tmp_path / "overlay_cam0.mp4").is_file()
    assert (tmp_path / "overlay_cam1.mp4").is_file()


# -- CLI --------------------------------------------------------------------


def _run_cli(*args: str, timeout: int = 240) -> subprocess.CompletedProcess:
    env_src = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "argus.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=timeout,
        env={**__import__("os").environ, "PYTHONPATH": env_src},
    )


def test_cli_help_exits_zero():
    result = _run_cli("--help", timeout=60)
    assert result.returncode == 0
    assert "bootstrap" in result.stdout and "doctor" in result.stdout


def test_cli_config_prints_the_effective_tuning():
    result = _run_cli("config", timeout=60)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["scoring"]["weights"]["fall"] == 0.4
    assert payload["engine"]["kind"] == "mock"


def test_cli_config_reflects_flag_overrides():
    result = _run_cli("config", "--engine", "onnx-cpu", timeout=60)
    assert json.loads(result.stdout)["engine"]["kind"] == "onnx-cpu"


def test_cli_doctor_runs_and_reports():
    result = _run_cli("doctor", timeout=120)
    assert "checks:" in result.stdout
    assert "[PASS] package numpy" in result.stdout


@pytest.mark.timeout(300)
def test_cli_run_end_to_end_over_the_demo_clip(tmp_path, demo_video):
    log = tmp_path / "triage.jsonl"
    result = _run_cli(
        "run",
        "--engine", "mock",
        "--camera", str(demo_video),
        "--json-log", str(log),
        "--max-ticks", "25",
        "--clock", "frame",
    )
    assert result.returncode == 0, result.stderr
    assert "processed 25 ticks" in result.stderr

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 25
    assert any(json.loads(line)["records"] for line in lines)


@pytest.mark.timeout(300)
def test_cli_run_serves_the_http_endpoint(tmp_path, demo_video):
    result = _run_cli(
        "run",
        "--engine", "mock",
        "--camera", str(demo_video),
        "--http-port", "8781",
        "--max-ticks", "10",
        "--clock", "frame",
    )
    assert result.returncode == 0, result.stderr
    assert "/triage" in result.stderr
