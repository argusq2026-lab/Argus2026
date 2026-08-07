"""Generate the cross-language protocol vectors for docs/PROTOCOL.md.

The laptop validates (`argus.ingest.protocol`); the phone encodes (Kotlin,
`android/.../Protocol.kt`). This freezes the contract into
`tests/data/protocol_vectors.json`, one file both sides are tested against:

* every `valid` message here is passed through the real server-side parser *at
  generation time* — the fixture cannot contain a message the server would
  reject;
* the Python test re-asserts that and pins the parsed field values;
* the Kotlin test asserts its encoder produces messages that value-equal these
  (key set and values — not bytes, since the two languages format doubles
  differently and the server parses rather than compares).

The `invalid` list is as binding as `valid`: those are the messages a
conforming phone must never produce, each with the reason the server refuses
it. They double as regression cases for the server parser itself.

Regenerate with:  python scripts/gen_protocol_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argus.config import load_config  # noqa: E402
from argus.ingest.protocol import parse_hello, parse_observation  # noqa: E402

OUT_PATH = REPO_ROOT / "tests" / "data" / "protocol_vectors.json"

ZERO_KP = [[0.0, 0.0]] * 17
ZERO_CONF = [0.0] * 17


def main() -> int:
    cfg = load_config()
    vocab = cfg.scoring.form_error_vocab
    version = cfg.ingest.protocol_version

    valid = [
        {
            "name": "hello_minimal",
            "message": {
                "type": "hello", "protocol_version": version,
                "station_id": "station-3", "trainee_id": "phoneA1B2",
            },
        },
        {
            "name": "hello_with_plan",
            "message": {
                "type": "hello", "protocol_version": version,
                "station_id": "squat-rack-1", "trainee_id": "phoneA1B2",
                "exercise_plan": "squat",
            },
        },
        {
            "name": "observation_detection_only",
            "why": "a phone with no pose model yet reports all 17 keypoints at "
                   "zero confidence, exactly as PROTOCOL.md instructs — never "
                   "omitted, never invented",
            "message": {
                "type": "observation", "ts": 1730649600.125,
                "bbox_xyxy": [0.12, 0.08, 0.61, 0.97],
                "keypoints_xy": ZERO_KP, "keypoints_conf": ZERO_CONF,
            },
        },
        {
            "name": "observation_full",
            "message": {
                "type": "observation", "ts": 1730649601.0,
                "bbox_xyxy": [0.1, 0.05, 0.6, 0.95],
                "keypoints_xy": [[0.3 + 0.01 * i, 0.1 + 0.04 * i] for i in range(17)],
                "keypoints_conf": [0.9] * 17,
                "exercise": "squat", "rep_count": 12, "form_ok": False,
                "form_reason_codes": ["insufficient_depth", "knee_valgus"],
            },
        },
        {
            "name": "observation_boundary_coords",
            "why": "exactly 0.0 and 1.0 are inside the normalized contract",
            "message": {
                "type": "observation", "ts": 0.0,
                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                "keypoints_xy": ZERO_KP, "keypoints_conf": ZERO_CONF,
                "form_reason_codes": [],
            },
        },
        {
            "name": "observation_plank_sagging",
            "why": "the plank classifier's output: `exercise` selects the scoring "
                   "weight profile, so it is load-bearing here in a way it is not "
                   "for exercises with no profile — a phone that omits it gets a "
                   "correct plank scored as a fall",
            "message": {
                "type": "observation", "ts": 1730649602.5,
                # Wide and short: a plank's bounding box.
                "bbox_xyxy": [0.18, 0.4, 0.88, 0.62],
                "keypoints_xy": [[0.2 + 0.04 * i, 0.45 + 0.005 * i] for i in range(17)],
                "keypoints_conf": [0.88] * 17,
                "exercise": "plank", "form_ok": False,
                "form_reason_codes": ["hips_sagging"],
            },
        },
        {
            "name": "hello_nursing",
            "why": "a nursing station must declare its use case, because the "
                   "laptop refuses a hello that does not match [session] "
                   "use_case — the check that stops a phone streaming to a "
                   "scorer that will never fire",
            "session_use_case": "nursing",
            "message": {
                "type": "hello", "protocol_version": version,
                "station_id": "bay-2", "trainee_id": "phoneC3D4",
                "use_case": "nursing",
            },
        },
        {
            "name": "observation_nursing_cpr",
            "why": "nursing carries pose plus a procedure, and none of "
                   "fitness's exercise/rep/form fields: its faults are derived "
                   "on the laptop from the movement, not classified on the phone",
            "message": {
                "type": "observation", "use_case": "nursing",
                "procedure": "cpr", "ts": 1730649602.5,
                "bbox_xyxy": [0.35, 0.25, 0.70, 0.95],
                "keypoints_xy": [[0.5, 0.3 + 0.02 * i] for i in range(17)],
                "keypoints_conf": [0.9] * 17,
            },
        },
    ]

    # Prove every valid message parses, right here at generation time. A vector
    # may name the session use case its message expects to be admitted to;
    # absent, it is fitness, which is what every vector predating the field is.
    for case in valid:
        msg = case["message"]
        if msg["type"] == "hello":
            parse_hello(msg, version, session_use_case=case.get("session_use_case", "fitness"))
        else:
            parse_observation(msg, vocab)

    invalid = [
        {
            "name": "hello_wrong_protocol_version",
            "why": "no negotiation: a mismatch is rejected outright, not downgraded",
            "message": {"type": "hello", "protocol_version": version + 1,
                        "station_id": "s", "trainee_id": "t"},
        },
        {
            "name": "hello_empty_trainee_id",
            "why": "trainee_id is the triage key; an empty one cannot be dispatched against",
            "message": {"type": "hello", "protocol_version": version,
                        "station_id": "s", "trainee_id": ""},
        },
        {
            "name": "observation_free_text_form_code",
            "why": "the closed vocabulary is the privacy boundary: free text in a "
                   "reason code is rejected, not scored",
            "message": {"type": "observation", "ts": 0.0,
                        "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                        "keypoints_xy": ZERO_KP, "keypoints_conf": ZERO_CONF,
                        "form_reason_codes": ["trainee looks unwell"]},
        },
        {
            "name": "observation_sixteen_keypoints",
            "why": "all 17 slots are always present; a missing keypoint is "
                   "low-confidence, not absent",
            "message": {"type": "observation", "ts": 0.0,
                        "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                        "keypoints_xy": ZERO_KP[:16], "keypoints_conf": ZERO_CONF[:16]},
        },
        {
            "name": "observation_missing_bbox",
            "message": {"type": "observation", "ts": 0.0,
                        "keypoints_xy": ZERO_KP, "keypoints_conf": ZERO_CONF},
        },
    ]

    document = {
        "vectors_version": 1,
        "protocol_version": version,
        "generated_by": "scripts/gen_protocol_vectors.py",
        "form_error_vocab_keys": sorted(vocab),
        "valid": valid,
        "invalid": invalid,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(valid)} valid + {len(invalid)} invalid to {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
