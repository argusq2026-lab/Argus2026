"""The server half of the cross-language protocol contract.

`tests/data/protocol_vectors.json` is the one file both platforms are tested
against: the Kotlin encoder must produce the `valid` messages
(`android/.../ProtocolTest.kt`), and this module asserts the server-side parser
accepts every one of them and refuses every `invalid` one.

Regenerate with `python scripts/gen_protocol_vectors.py` — deliberately, since
a diff here is a change to what a phone is allowed to say.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.ingest.protocol import ProtocolError, parse_hello, parse_observation

VECTORS_PATH = Path(__file__).parent / "data" / "protocol_vectors.json"


def _document() -> dict:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def document() -> dict:
    return _document()


@pytest.fixture(scope="module")
def vocab(default_config) -> dict:
    return default_config.scoring.form_error_vocab


def test_the_fixture_pins_the_live_protocol_version(document, default_config):
    assert document["protocol_version"] == default_config.ingest.protocol_version


def test_the_fixture_vocab_matches_the_live_config(document, vocab):
    """A vocabulary edit must regenerate the fixture, not silently outdate it."""
    assert document["form_error_vocab_keys"] == sorted(vocab)


@pytest.mark.parametrize("case", _document()["valid"], ids=lambda c: c["name"])
def test_every_valid_message_parses(case, vocab, default_config):
    msg = case["message"]
    if msg["type"] == "hello":
        hello = parse_hello(
            msg,
            default_config.ingest.protocol_version,
            # A vector may name the session use case it expects to join; absent,
            # it is fitness, as every vector predating the field is.
            session_use_case=case.get("session_use_case", "fitness"),
        )
        assert hello.station_id == msg["station_id"]
        assert hello.trainee_id == msg["trainee_id"]
        assert hello.use_case == msg.get("use_case", "fitness")
    else:
        obs = parse_observation(msg, vocab)
        assert obs.ts == msg["ts"]
        assert list(obs.bbox_xyxy) == msg["bbox_xyxy"]
        assert len(obs.keypoints_xy) == 17
        assert list(obs.form_reason_codes) == msg.get("form_reason_codes", [])
        assert obs.use_case == msg.get("use_case", "fitness")
        assert obs.procedure == msg.get("procedure")


@pytest.mark.parametrize("case", _document()["invalid"], ids=lambda c: c["name"])
def test_every_invalid_message_is_refused(case, vocab, default_config):
    msg = case["message"]
    with pytest.raises(ProtocolError):
        if msg.get("type") == "hello":
            parse_hello(msg, default_config.ingest.protocol_version)
        else:
            parse_observation(msg, vocab)
