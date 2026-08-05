"""Every form artifact, its fixture, and the arithmetic both ends must agree on.

`FormClassifier.kt` reimplements a scikit-learn multinomial logistic
regression in Kotlin, and `FormClassifierTest` pins that to
`tests/data/<exercise>_vectors.json`. But that test needs a JDK and the
Android SDK, so it does not run in the `test`, `determinism`, `privacy`, or
`lint` jobs -- only in `android`. This file closes that gap from the Python
side, for every exercise this build ships an artifact for
(docs/ADDING_AN_EXERCISE.md §3), and serves two purposes beyond duplication:

1. **It is the reference implementation of the scoring arithmetic**, in
   stdlib Python, readable next to the Kotlin it constrains. If the two
   disagree about feature order, standardization, or the softmax, one of them
   fails here first.
2. **It catches a regenerated artifact paired with a stale fixture.** Both
   files of a pair come out of one `scripts/train_form_model.py` run;
   committing only one is the mistake this notices.

Deliberately stdlib only -- no numpy, no sklearn. Those are offline training
dependencies (`requirements-train.txt`); nothing in the shipped system needs
them, and this test would quietly make them a test-time requirement if it
reached for them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argus.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets"
FIXTURES_DIR = REPO_ROOT / "tests" / "data"


def discover_exercises() -> list[str]:
    return sorted(p.name.removesuffix("_lr.json") for p in ASSETS_DIR.glob("*_lr.json"))


EXERCISES = discover_exercises()

pytestmark = pytest.mark.skipif(
    not EXERCISES,
    reason="no form artifacts generated; run scripts/train_form_model.py",
)


@pytest.fixture(scope="module", params=EXERCISES)
def exercise(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def artifact(exercise: str) -> dict:
    return json.loads((ASSETS_DIR / f"{exercise}_lr.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vectors(exercise: str) -> dict:
    return json.loads((FIXTURES_DIR / f"{exercise}_vectors.json").read_text(encoding="utf-8"))


def predict(artifact: dict, features: list[float]) -> list[float]:
    """The reference scoring path: standardize, project, softmax.

    This is `FormClassifier.logits` + `FormClassifier.softmax` line for line,
    including the max-shift -- without it, `exp` of a large logit overflows to
    infinity and every probability comes back NaN, which compares false
    against the threshold and would silently mean "never report a form error".
    """
    mean, scale = artifact["scaler_mean"], artifact["scaler_scale"]
    standardized = [(f - m) / s for f, m, s in zip(features, mean, scale)]
    logits = [
        b + sum(w * x for w, x in zip(row, standardized))
        for row, b in zip(artifact["coef"], artifact["intercept"])
    ]
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def test_at_least_one_artifact_was_generated():
    assert EXERCISES, "run scripts/train_form_model.py before this suite means anything"


# -- the artifact is internally consistent ----------------------------------


def test_shapes_agree_across_the_artifact(exercise, artifact):
    width = len(artifact["landmarks"]) * len(artifact["feature_kinds"])
    assert len(artifact["feature_names"]) == width
    assert len(artifact["scaler_mean"]) == width
    assert len(artifact["scaler_scale"]) == width
    assert len(artifact["coef"]) == len(artifact["classes"])
    assert len(artifact["intercept"]) == len(artifact["classes"])
    assert all(len(row) == width for row in artifact["coef"])


def test_no_zero_scale(exercise, artifact):
    """A zero scale would divide by zero at standardization time."""
    assert all(s != 0.0 for s in artifact["scaler_scale"])


def test_feature_names_are_grouped_per_landmark(exercise, artifact):
    """Feature order is the whole contract: a transposed layout still scores."""
    expected = [
        f"{lm}_{s}"
        for lm in artifact["landmarks"]
        for s in artifact["feature_kinds"]
    ]
    assert artifact["feature_names"] == expected


def test_artifact_names_its_own_exercise(exercise, artifact):
    assert artifact["exercise"] == exercise


# -- the two guards that exist because of a real failure --------------------


def test_only_coordinates_are_features(exercise, artifact):
    """No confidence-valued feature may enter the model.

    Plank's first revision used MediaPipe's per-landmark *visibility* as a
    feature, on the assumption it was the same kind of number as YOLO26's
    keypoint confidence. It is not, and the model consequently reported a
    form error for every plank it was shown -- see
    `test_no_feature_has_a_saturating_scale` for the mechanism. Coordinates
    transfer between pose models because both ends mean the same thing by
    them; confidence does not.
    """
    assert artifact["feature_kinds"] == ["x", "y"]
    assert not [n for n in artifact["feature_names"] if n.endswith(("_v", "_z"))]


def test_no_feature_has_a_saturating_scale(exercise, artifact):
    """A vanishing training spread turns a normal input into hundreds of sigma.

    `nose_v` in plank's withdrawn revision had mean 0.9993 and sd 0.0013, so an
    ordinary confidence of 0.85 standardized to -116. Bicep hit the same class
    of failure from geometry rather than confidence: under bbox encoding the
    head sits at the top of the landmark box on nearly every frame of a curl,
    so `nose_y`'s spread came in under this floor too -- `nose` is excluded
    from bicep's landmark set for exactly that reason
    (`scripts/train_form_model.py`). Either way, a feature below this floor
    saturates the softmax before the pose is consulted, which is why a
    correct plank once came back as `hips_sagging` at 100%.

    0.01 is a floor, not a fitted value: it bounds a [0, 1]-ranged feature's
    worst-case standardized magnitude at ~100 sigma. Any feature below it is
    effectively constant in training and cannot survive a change of upstream
    model, whatever it is measuring.
    """
    offenders = [
        (n, s)
        for n, s in zip(artifact["feature_names"], artifact["scaler_scale"])
        if s < 0.01
    ]
    assert not offenders, (
        f"{exercise}: features with a near-constant training spread: {offenders}. "
        "An ordinary input becomes hundreds of sigma and saturates the softmax."
    )


def test_every_landmark_exists_in_coco_17(exercise, artifact):
    """The phone can only supply COCO-17; a landmark outside it is unfillable."""
    coco = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    }
    assert set(artifact["landmarks"]) <= coco


# -- the evidence gates ------------------------------------------------------


def test_min_visible_landmarks_is_in_range(exercise, artifact):
    assert 1 <= artifact["min_visible_landmarks"] <= len(artifact["landmarks"])


def test_depth_gate_bounds_come_from_training_data_not_a_guess(exercise, artifact):
    """Where present, the depth gate's [min, max] must be a real observed range.

    Lunge's `knee_over_toe` label was only ever collected -- and, per
    upstream's own detection code, only ever evaluated -- at the bottom of a
    lunge (docs/VALIDATION.md). This is the second evidence gate that fact
    implies: `min < max`, both within the [0, 1] normalized-coordinate range
    the gap is measured in, and both landmark lists are declared.
    """
    gate = artifact["depth_gate"]
    if gate is None:
        return
    assert gate["numerator_landmarks"] and gate["reference_landmarks"]
    assert set(gate["numerator_landmarks"]) <= set(artifact["landmarks"])
    assert set(gate["reference_landmarks"]) <= set(artifact["landmarks"])
    assert 0.0 <= gate["min"] < gate["max"] <= 1.0


# -- the boundary with the rest of the system -------------------------------


def test_emitted_codes_are_in_the_servers_vocabulary(exercise, artifact):
    """An unrecognised code closes the connection, so this is a broken station.

    `argus.ingest.protocol.parse_observation` rejects a `form_reason_codes`
    entry outside `[scoring.form_error_vocab]`. A classifier able to emit one
    would take the phone offline mid-class rather than mis-score a rep.
    """
    vocab = load_config().scoring.form_error_vocab
    for label, code in artifact["class_to_code"].items():
        if code is None:
            continue
        assert code in vocab, f"{exercise}: class {label!r} emits {code!r}, absent from the vocabulary"


def test_the_correct_class_maps_to_no_code(exercise, artifact):
    """A correct classification reports an empty list, never a 'correct' code.

    `form_reason_codes` is a list of things that are *wrong*; the scorer takes
    its max weight. A code meaning 'correct' would have to be weighted 0 and
    would still show up in the wire message as though something were flagged.
    """
    assert artifact["class_to_code"]["C"] is None


def test_the_exercise_profile_exists_for_this_classifier(exercise, artifact):
    """Shipping a classifier without its weight profile is the failure mode.

    Without `[scoring.exercise_weights.<exercise>]`, a correct rep can score
    most of the way to `alert_threshold` on `fall`/`stillness`/`off_task`
    alone -- the classifier would be working perfectly and the dashboard
    would still be wrong. See docs/VALIDATION.md for why each shipped profile
    zeroes what it zeroes.
    """
    scoring = load_config().scoring
    profile = scoring.weights_for(exercise)
    assert profile is not scoring.weights, f"no {exercise} profile configured"
    assert profile["form_error"] > scoring.weights["form_error"]


# -- the arithmetic Kotlin must reproduce -----------------------------------


def test_reference_arithmetic_reproduces_the_fixture(exercise, artifact, vectors):
    """If this fails, the artifact and the fixture came from different runs."""
    tolerance = vectors["tolerance"]
    assert vectors["cases"], f"{exercise} fixture has no cases"
    for i, case in enumerate(vectors["cases"]):
        actual = predict(artifact, case["features"])
        expected = case["probabilities"]
        assert len(actual) == len(expected)
        for c, (a, e) in enumerate(zip(actual, expected)):
            assert a == pytest.approx(e, abs=tolerance), f"{exercise} case {i} class {c}"


def bbox_normalize(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """`train_form_model.bbox_normalize`, verbatim, for the check below."""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    return [(x - x0) / w for x in xs], [(y - y0) / h for y in ys]


def test_raw_features_encode_to_the_shipped_features(exercise, artifact, vectors):
    """`raw_features` and `features` must be the same pose, two encodings.

    `FormClassifierTest.kt` classifies from `raw_features` (pre-encoding,
    what a pose estimator actually produces) rather than `features` (the
    model's actual input) -- `classify()` applies the artifact's encoding
    itself. If these two ever drifted apart, that test would silently start
    exercising a different pose than the one `features`/`probabilities` were
    computed from.
    """
    kinds = artifact["feature_kinds"]
    stride = len(kinds)
    n = len(artifact["landmarks"])
    for i, case in enumerate(vectors["cases"]):
        raw = case["raw_features"]
        assert len(raw) == n * stride, f"{exercise} case {i}"
        if artifact["encoding"] == "raw":
            assert raw == case["features"], f"{exercise} case {i}"
            continue
        assert artifact["encoding"] == "bbox"
        xs = [raw[j * stride + kinds.index("x")] for j in range(n)]
        ys = [raw[j * stride + kinds.index("y")] for j in range(n)]
        nx, ny = bbox_normalize(xs, ys)
        expected = []
        for j in range(n):
            expected.append(nx[j])
            expected.append(ny[j])
        for a, e in zip(case["features"], expected):
            assert a == pytest.approx(e, abs=1e-9), f"{exercise} case {i}"


def test_the_fixture_decisions_follow_from_its_probabilities(exercise, artifact, vectors):
    """Threshold and code mapping, not just the raw probabilities."""
    classes = vectors["classes"]
    threshold = vectors["probability_threshold"]
    for i, case in enumerate(vectors["cases"]):
        probabilities = case["probabilities"]
        best = max(range(len(probabilities)), key=lambda k: probabilities[k])
        assert classes[best] == case["predicted_class"], f"{exercise} case {i}"
        code = artifact["class_to_code"][classes[best]]
        expected = [code] if (probabilities[best] >= threshold and code) else []
        assert case["form_reason_codes"] == expected, f"{exercise} case {i}"


def test_probabilities_are_a_distribution(exercise, artifact, vectors):
    for i, case in enumerate(vectors["cases"]):
        probabilities = predict(artifact, case["features"])
        assert all(math.isfinite(p) for p in probabilities), f"{exercise} case {i}"
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-9), f"{exercise} case {i}"


def test_artifact_and_fixture_agree_on_encoding_and_threshold(exercise, artifact, vectors):
    """The two files of a pair are written by one run; disagreeing means one is stale."""
    assert artifact["encoding"] == vectors["encoding"]
    assert artifact["probability_threshold"] == vectors["probability_threshold"]
    assert artifact["classes"] == vectors["classes"]
    assert vectors["exercise"] == exercise


def test_an_extreme_input_does_not_produce_nan(exercise, artifact):
    """The softmax max-shift, exercised rather than assumed."""
    width = len(artifact["feature_names"])
    probabilities = predict(artifact, [1e6] * width)
    assert all(math.isfinite(p) for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-9)
