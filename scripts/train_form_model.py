"""Refit an exercise's form classifier on the features a COCO-17 phone can produce.

Generalises what was `train_plank_model.py` to any exercise in
`EXERCISE_SPECS` below. See `docs/ADDING_AN_EXERCISE.md` before adding a new
entry -- most of what looks like a modelling decision here is actually one of
four traps that plank hit first.

The labelled data comes from NgoQuocBao1010/Exercise-Correction (MIT), which
trained its classifiers on MediaPipe Pose landmarks. Those models cannot be
loaded and used here, for reasons that are structural rather than cosmetic:

* **z is not available.** MediaPipe emits a depth estimate per landmark;
  YOLO26-pose emits x, y and a confidence. A chunk of their feature vector
  does not exist on our wire.
* **Some landmarks are not available.** Their feature sets can include
  `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`. COCO-17
  stops at the ankles.
* **Visibility is not confidence.** Their feature set includes a MediaPipe
  *visibility* per landmark. It looks like YOLO26's keypoint confidence and is
  not the same quantity: MediaPipe visibility saturates at ~0.999 with a
  standard deviation of ~0.001 on core joints, so a scaler fitted on it turns
  a perfectly ordinary YOLO26 confidence of 0.85 into a feature hundreds of
  sigma from the mean. An earlier revision of the plank script kept those
  columns, and the resulting model reported 100% confidence on whatever it was
  pointed at. They are dropped, for every exercise, always.

The general rule this cost a rebuild to learn: a feature is only transferable
between two pose models if both models mean the same thing by it. Coordinates
do. Confidence does not.

Two feature encodings are trained and compared:

* **raw** — their encoding: normalized image coordinates, as-is. A model
  trained this way partly learns *where in the frame* a person is.
* **bbox** — coordinates re-expressed relative to the person's bounding box
  (translate to the box origin, divide by box size), which is what makes the
  classifier indifferent to where the trainee stands and how close the phone
  is. `docs/PROTOCOL.md` already carries `bbox_xyxy` per observation, so the
  phone can reproduce this exactly.

The chosen model is written as plain coefficients, not a pickle. A multinomial
logistic regression is a matrix multiply and a softmax, so the phone evaluates
it in Kotlin arithmetic with no ML runtime, no ONNX export, and no second NPU
dispatch -- see android/.../FormClassifier.kt, which is pinned to the
reference vectors this script also emits.

Usage:
    pip install -r requirements-train.txt
    python scripts/train_form_model.py --exercise plank --data-dir scratch/
    python scripts/train_form_model.py --exercise bicep --data-dir scratch/
    python scripts/train_form_model.py --exercise lunge --data-dir scratch/
    python scripts/train_form_model.py --exercise all --data-dir scratch/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Their deployed threshold for plank (web/server/detection/plank.py). Below
#: it they return "unknown"; we emit no code, which is the same decision -- a
#: low-confidence guess is not worth an instructor's walk across the floor.
#: Upstream does not publish a threshold for bicep's or lunge's error models
#: (their detection notebooks report the raw predicted class), so the same
#: conservative value is applied there too, for the same reason. See
#: docs/VALIDATION.md.
PREDICTION_PROBABILITY_THRESHOLD = 0.6

#: Per-landmark contribution -- coordinates only. See the module docstring for
#: why visibility is excluded. The artifact declares this so `FormClassifier.kt`
#: builds its feature vector from the file rather than a hardcoded stride.
FEATURE_KINDS = ("x", "y")

BASE_URL = "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/"


class ExerciseSpec:
    def __init__(
        self,
        name: str,
        landmarks: list[str],
        upstream_dir: str,
        train_file: str,
        test_file: str,
        label_to_code: dict[str, str | None],
        min_visible_landmarks: int,
        source_note: str,
        depth_gate: dict[str, list[str]] | None = None,
    ):
        self.name = name
        self.landmarks = landmarks
        self.data_base = BASE_URL + upstream_dir + "/"
        self.train_file = train_file
        self.test_file = test_file
        self.label_to_code = label_to_code
        self.min_visible_landmarks = min_visible_landmarks
        self.source_note = source_note
        #: Optional second evidence gate, alongside landmark visibility. See
        #: `depth_gate_bounds` below -- lunge's knee-over-toe label was only
        #: ever collected (and, per upstream's own detection code, only ever
        #: evaluated) at the bottom of a lunge. A classifier fitted on that
        #: has no idea what a standing or ascending trainee looks like, and a
        #: softmax will still confidently name one of its two classes for
        #: one -- the same failure Trap 2 describes for plank, caused by pose
        #: instead of occlusion. `numerator`/`reference` landmarks are
        #: averaged and differenced (numerator_y - reference_y) in the same
        #: raw normalized-image coordinates upstream measured depth in; the
        #: bounds are the training data's own observed range, not a guess.
        self.depth_gate = depth_gate


#: The 13 landmarks the plank model was fit on -- MediaPipe's set intersected
#: with COCO-17, in COCO index order. Heels and foot indices have no COCO
#: equivalent and are dropped, along with every `_z` and `_v` column.
PLANK_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

#: 8 of bicep's 9 upstream landmarks are used. All 9 are in COCO-17 -- no
#: heels, no foot indices, no z/v loss beyond the usual drop -- but `nose` is
#: excluded anyway: under bbox encoding the head sits at the top of the
#: landmark box on nearly every frame of a curl (only the arms move), so
#: `nose_y`'s training spread comes in under the 0.01 saturating-scale floor
#: (`test_no_feature_has_a_saturating_scale`) -- the same failure Trap 1
#: describes, this time from geometry rather than a confidence column. See
#: docs/VALIDATION.md. Shoulders/elbows/wrists/hips already carry the
#: lean-back signal without it.
BICEP_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
]

#: 9 of lunge's 13 upstream landmarks survive; heels and foot indices do not.
LUNGE_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

EXERCISE_SPECS: dict[str, ExerciseSpec] = {
    "plank": ExerciseSpec(
        name="plank",
        landmarks=PLANK_LANDMARKS,
        upstream_dir="plank_model",
        train_file="train.csv",
        test_file="test.csv",
        # "C" maps to no code at all: a correct plank reports an empty
        # `form_reason_codes`, it does not report "correct". See
        # [scoring.form_error_vocab] in configs/argus.default.toml.
        label_to_code={"C": None, "L": "hips_sagging", "H": "hips_piked"},
        min_visible_landmarks=10,
        source_note="NgoQuocBao1010/Exercise-Correction (MIT), core/plank_model",
    ),
    "bicep": ExerciseSpec(
        name="bicep",
        landmarks=BICEP_LANDMARKS,
        upstream_dir="bicep_model",
        train_file="train.csv",
        test_file="test.csv",
        # Upstream's binary label is their "lean too far back" error -- the
        # one of their three bicep-curl faults picked out for ML detection;
        # loose-upper-arm and weak-peak-contraction are threshold checks in
        # their code, not part of this dataset. See core/bicep_model/README.md.
        label_to_code={"C": None, "L": "lean_back_error"},
        min_visible_landmarks=7,
        source_note="NgoQuocBao1010/Exercise-Correction (MIT), core/bicep_model",
    ),
    "lunge": ExerciseSpec(
        name="lunge",
        landmarks=LUNGE_LANDMARKS,
        upstream_dir="lunge_model",
        train_file="err.train.csv",
        test_file="err.test.csv",
        # `err.*`, not `stage.*`: Argus has no rep-phase concept and does not
        # need one. See the depth_gate note below for what this label being
        # scoped to the bottom of a rep implies.
        label_to_code={"C": None, "L": "knee_over_toe"},
        min_visible_landmarks=7,
        source_note="NgoQuocBao1010/Exercise-Correction (MIT), core/lunge_model (err.* split)",
        depth_gate={
            "numerator": ["left_ankle", "right_ankle"],
            "reference": ["left_hip", "right_hip"],
        },
    ),
}


def feature_names(spec: ExerciseSpec) -> list[str]:
    return [f"{lm}_{s}" for lm in spec.landmarks for s in FEATURE_KINDS]


def fetch(data_dir: Path, spec: ExerciseSpec, name: str) -> Path:
    path = data_dir / spec.name / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec.name}/{name} ...")
        urllib.request.urlretrieve(spec.data_base + name, path)
    return path


def bbox_normalize(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Re-express coordinates relative to the box the landmarks span.

    Deliberately *not* the observation's `bbox_xyxy`: the training CSV has no
    detector box, and even if it did, MediaPipe's person box and YOLO26's are
    not the same convention -- normalising by them would train on one
    definition and infer on another. The landmark extent is computed the same
    way from the same joints on both sides, which is what makes the phone
    able to reproduce this exactly. FormClassifier.kt must derive it
    identically.

    Degenerate spans (a collapsed detection) divide by a floor rather than
    exploding -- the resulting features are meaningless either way, but they
    stay finite so one bad frame cannot poison a batch.
    """
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    return [(x - x0) / w for x in xs], [(y - y0) / h for y in ys]


def load_xy(path: Path, spec: ExerciseSpec, encoding: str):
    """Read a labelled CSV into (features, labels, raw_depth_gap) using `encoding`.

    `raw_depth_gap` is None unless `spec.depth_gate` is set; when it is, it is
    computed in raw (pre-encoding) normalized-image coordinates regardless of
    `encoding`, because the gate exists to bound what pose the model was
    fitted on, not to be a feature itself.
    """
    import numpy as np

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        missing = [
            f"{lm}_{s}" for lm in spec.landmarks for s in FEATURE_KINDS
            if f"{lm}_{s}" not in index
        ]
        if missing:
            raise SystemExit(f"{path.name}: missing expected columns {missing[:5]}")

        gate = spec.depth_gate
        gate_num = [f"{lm}_y" for lm in gate["numerator"]] if gate else []
        gate_ref = [f"{lm}_y" for lm in gate["reference"]] if gate else []

        rows, labels, gaps = [], [], []
        for row in reader:
            xs = [float(row[index[f"{lm}_x"]]) for lm in spec.landmarks]
            ys = [float(row[index[f"{lm}_y"]]) for lm in spec.landmarks]
            if gate:
                num = sum(float(row[index[c]]) for c in gate_num) / len(gate_num)
                ref = sum(float(row[index[c]]) for c in gate_ref) / len(gate_ref)
                gaps.append(num - ref)
            if encoding == "bbox":
                xs, ys = bbox_normalize(xs, ys)
            feats = []
            for x, y in zip(xs, ys):
                feats.extend((x, y))
            rows.append(feats)
            labels.append(row[index["label"]])
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels),
        np.asarray(gaps, dtype=np.float64) if gate else None,
    )


def train(spec: ExerciseSpec, encoding: str, train_csv: Path, test_csv: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    x_train, y_train, gaps_train = load_xy(train_csv, spec, encoding)
    x_test, y_test, _ = load_xy(test_csv, spec, encoding)

    scaler = StandardScaler().fit(x_train)
    # Multinomial (softmax) is the default for multi-class in modern sklearn;
    # the explicit `multi_class=` argument was removed in 1.7. This matters for
    # the Kotlin port, which implements softmax over all classes rather than
    # one-vs-rest sigmoids.
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler.transform(x_train), y_train)

    predicted = model.predict(scaler.transform(x_test))
    accuracy = accuracy_score(y_test, predicted)

    print(f"\n=== {spec.name} encoding={encoding} ===")
    print(f"train rows {len(y_train)}, test rows {len(y_test)}, features {x_train.shape[1]}")
    print(f"test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predicted, digits=4))
    print("confusion matrix (rows=true, cols=pred), classes:", list(model.classes_))
    print(confusion_matrix(y_test, predicted, labels=model.classes_))

    return {
        "encoding": encoding,
        "accuracy": float(accuracy),
        "model": model,
        "scaler": scaler,
        "x_test": x_test,
        "y_test": y_test,
        "gaps_train": gaps_train,
    }


def coef_and_intercept(model) -> dict:
    """`coef`/`intercept`, always one row per class -- softmax-shaped.

    Binary classification is where this bites: sklearn's `LogisticRegression`
    fits a single sigmoid for two classes, so `model.coef_` and
    `model.intercept_` come back with exactly *one* row (log-odds of
    `classes_[1]` vs `classes_[0]`), not one per class. Both `FormClassifier.kt`
    and the Python reference implementation in `tests/test_form_artifacts.py`
    are written for a generic C-row softmax -- correct for plank's 3 classes,
    silently wrong for bicep/lunge's 2 if the single sklearn row were shipped
    as-is (a 1x2 "coef" that both sides would misread as one degenerate class).

    The fix ships a genuine second row: `classes_[0]` gets the zero vector,
    `classes_[1]` gets sklearn's fitted row. Softmax over `[0, z]` is
    `[sigmoid(-z), sigmoid(z)]`, which is exactly binary logistic regression's
    `predict_proba` -- so this is a re-expression of the identical decision
    boundary, not an approximation of it, and both sides only ever need to
    implement one kind of arithmetic.
    """
    coef, intercept = model.coef_, model.intercept_
    if coef.shape[0] == 1:
        coef = [[0.0] * coef.shape[1], list(coef[0])]
        intercept = [0.0, float(intercept[0])]
    return {
        "coef": [[float(v) for v in row] for row in coef],
        "intercept": [float(v) for v in intercept],
    }


def write_artifact(spec: ExerciseSpec, result, path: Path) -> dict:
    model, scaler = result["model"], result["scaler"]
    artifact = {
        "format": "multinomial_logistic_regression",
        "artifact_version": 1,
        "exercise": spec.name,
        "encoding": result["encoding"],
        "source_dataset": spec.source_note,
        "generated_by": "scripts/train_form_model.py",
        "test_accuracy": round(result["accuracy"], 6),
        "test_accuracy_caveat": (
            "Held-out frames from the source dataset's own recordings, which "
            "feature a small number of people. This is not a per-subject "
            "generalisation estimate and must not be quoted as accuracy on a "
            "new trainee. See docs/VALIDATION.md."
        ),
        "probability_threshold": PREDICTION_PROBABILITY_THRESHOLD,
        "feature_names": feature_names(spec),
        # Consumed by FormClassifier.kt so the phone builds its feature vector
        # from the artifact rather than a hardcoded stride -- dropping a kind
        # here must not need a matching Kotlin edit to stay correct.
        "feature_kinds": list(FEATURE_KINDS),
        "landmarks": spec.landmarks,
        "classes": [str(c) for c in model.classes_],
        "class_to_code": {str(c): spec.label_to_code[str(c)] for c in model.classes_},
        "min_visible_landmarks": spec.min_visible_landmarks,
        "scaler_mean": [float(v) for v in scaler.mean_],
        # Guard against the failure that made the plank rewrite necessary: a
        # feature whose training spread is vanishing turns any ordinary
        # deviation into hundreds of sigma and saturates the softmax.
        # Coordinates never do this; the dropped visibility columns did.
        "scaler_scale": [float(v) for v in scaler.scale_],
        "min_scaler_scale": float(min(scaler.scale_)),
        **coef_and_intercept(model),
        "depth_gate": None,
    }
    if spec.depth_gate is not None:
        gaps = result["gaps_train"]
        artifact["depth_gate"] = {
            "numerator_landmarks": spec.depth_gate["numerator"],
            "reference_landmarks": spec.depth_gate["reference"],
            # The training data's own observed range -- not a guessed
            # threshold. Outside it, the pose is unlike anything the label was
            # ever fit against; see docs/VALIDATION.md for why this exists
            # specifically for lunge.
            "min": float(gaps.min()),
            "max": float(gaps.max()),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}  ({len(artifact['coef'][0])} features)")
    return artifact


def write_fixture(spec: ExerciseSpec, result, artifact: dict, raw_x_test, path: Path) -> None:
    """Reference (input -> probabilities -> code) triples for the Kotlin test.

    The Kotlin classifier is a reimplementation, so it needs the same pinning
    the decode path already has: cases drawn from real test rows, scored by
    the actual fitted model here, that `FormClassifierTest` must reproduce.

    Each case carries both `features` (in the artifact's shipped encoding --
    what the model was fit on, what the Python reference in
    `tests/test_form_artifacts.py` scores directly) and `raw_features`
    (pre-encoding, in the same normalized-image space `classify()` actually
    receives from a pose estimator). They are *not* interchangeable inputs to
    `FormClassifier.classify()`: it does its own encoding internally, so
    feeding it `features` for a `bbox` artifact double-applies the transform.
    That happens to be harmless for the softmax (re-normalizing an
    already-normalized box is an identity), which is exactly how this went
    unnoticed for plank -- but lunge's `depth_gate` runs on the pre-encoding
    coordinates specifically, and silently saw the wrong ones. `raw_features`
    exists so `FormClassifierTest` can call `classify()` the way the phone
    actually does.
    """
    import numpy as np

    model, scaler = result["model"], result["scaler"]
    x_test, y_test = result["x_test"], result["y_test"]
    classes = [str(c) for c in model.classes_]

    # A few rows of each true class, so every branch of the mapping is covered.
    picked: list[int] = []
    for label in sorted(set(str(v) for v in y_test)):
        idx = [i for i, v in enumerate(y_test) if str(v) == label][:3]
        picked.extend(idx)

    cases = []
    for i in picked:
        features = x_test[i]
        probabilities = model.predict_proba(scaler.transform(features.reshape(1, -1)))[0]
        best = int(np.argmax(probabilities))
        confident = probabilities[best] >= PREDICTION_PROBABILITY_THRESHOLD
        code = artifact["class_to_code"][classes[best]] if confident else None
        cases.append({
            "true_label": str(y_test[i]),
            "features": [float(v) for v in features],
            "raw_features": [float(v) for v in raw_x_test[i]],
            "probabilities": [float(p) for p in probabilities],
            "predicted_class": classes[best],
            "form_reason_codes": [code] if code else [],
        })

    document = {
        "vectors_version": 1,
        "exercise": spec.name,
        "generated_by": "scripts/train_form_model.py",
        "why": (
            "FormClassifier.kt reimplements this model's arithmetic in Kotlin. "
            "These cases pin that reimplementation to the fitted model, the same "
            "way protocol_vectors.json pins the encoder to the server's parser."
        ),
        "encoding": artifact["encoding"],
        "classes": classes,
        "probability_threshold": PREDICTION_PROBABILITY_THRESHOLD,
        "tolerance": 1e-6,
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(cases)} reference cases)")


def run(spec: ExerciseSpec, data_dir: Path, encoding_arg: str) -> int:
    train_csv = fetch(data_dir, spec, spec.train_file)
    test_csv = fetch(data_dir, spec, spec.test_file)

    encodings = ["raw", "bbox"] if encoding_arg == "both" else [encoding_arg]
    results = [train(spec, e, train_csv, test_csv) for e in encodings]

    # `bbox` ships whenever it was trained, even when `raw` measures higher.
    #
    # This is not a tie-break, it is a refusal to be guided by this metric.
    # `raw` features are absolute image coordinates, so a model fitted on them
    # partly learns where in the *source recordings'* frame a person stood.
    # The test split comes from those same recordings, so it rewards exactly
    # the memorisation that makes the model useless on a phone whose framing,
    # distance, and mounting height are all different. A percentage point
    # measured on data that cannot expose the flaw is not evidence.
    best = next((r for r in results if r["encoding"] == "bbox"), results[0])
    if len(results) > 1:
        print(f"\n=== {spec.name} comparison ===")
        for r in results:
            print(f"  {r['encoding']:<5} test accuracy {r['accuracy']:.4f}")
        print(f"shipping: {best['encoding']} (position-invariant; see the note in run())")

    # `write_fixture` needs the *raw* (pre-encoding) coordinates for the same
    # test rows too -- see its docstring. `load_xy` is deterministic and
    # unshuffled, so re-reading `test_csv` with encoding="raw" lines up
    # index-for-index with `best["x_test"]`/`best["y_test"]` regardless of
    # which encoding was shipped.
    raw_x_test, _, _ = load_xy(test_csv, spec, "raw")

    out_artifact = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / f"{spec.name}_lr.json"
    out_fixture = REPO_ROOT / "tests" / "data" / f"{spec.name}_vectors.json"
    artifact = write_artifact(spec, best, out_artifact)
    write_fixture(spec, best, artifact, raw_x_test, out_fixture)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", choices=[*EXERCISE_SPECS, "all"], default="all",
                        help="which exercise's classifier to (re)fit")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "scratch",
                        help="where each exercise's CSVs live (downloaded if absent)")
    parser.add_argument("--encoding", choices=["raw", "bbox", "both"], default="both",
                        help="feature encoding to train; 'both' compares and ships the better")
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "scikit-learn is not installed. This script is offline tooling:\n"
            "    pip install -r requirements-train.txt"
        )

    names = list(EXERCISE_SPECS) if args.exercise == "all" else [args.exercise]
    for name in names:
        run(EXERCISE_SPECS[name], args.data_dir, args.encoding)
    return 0


if __name__ == "__main__":
    sys.exit(main())
