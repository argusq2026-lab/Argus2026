"""Refit the plank form classifier on the features a COCO-17 phone can produce.

The labelled data comes from NgoQuocBao1010/Exercise-Correction (MIT), which
trained a plank classifier on MediaPipe Pose landmarks. That model cannot be
loaded and used here, for two reasons that are structural rather than
cosmetic:

* **z is not available.** MediaPipe emits a depth estimate per landmark;
  YOLO26-pose emits x, y and a confidence. A quarter of their feature vector
  does not exist on our wire.
* **Four landmarks are not available.** Their feature set includes
  `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`. COCO-17
  stops at the ankles.

* **Visibility is not confidence.** Their feature set includes a MediaPipe
  *visibility* per landmark. It looks like YOLO26's keypoint confidence and is
  not the same quantity: MediaPipe visibility saturates at ~0.999 with a
  standard deviation of 0.0013 on core joints, so a scaler fitted on it turns
  a perfectly ordinary YOLO26 confidence of 0.85 into a **-116 sigma**
  feature. An earlier revision of this script kept those columns, and the
  resulting model reported 100% confidence on whatever it was pointed at,
  because 13 features hundreds of sigma from their training mean saturate the
  softmax before the pose is consulted. A known-correct plank flipped from
  `C` to `L` purely by substituting realistic confidences. They are dropped.

So 26 of their 68 features survive -- x and y for 13 landmarks -- and the
model is refit on those. Their data and their three-class taxonomy (C correct
/ L low-back / H high-back) carry over unchanged; only the input space
narrows.

The general rule this cost us: a feature is only transferable between two
pose models if both models mean the same thing by it. Coordinates do.
Confidence does not.

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
dispatch -- see android/.../PlankClassifier.kt, which is pinned to the
reference vectors this script also emits.

Usage:
    pip install -r requirements-train.txt
    python scripts/train_plank_model.py --data-dir scratch/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The 13 landmarks that exist in both MediaPipe's set and COCO-17, in COCO
#: index order. The four MediaPipe landmarks with no COCO equivalent
#: (heels, foot indices) and every `_z` column are dropped.
COCO_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

#: Their label -> our closed-vocabulary code. "C" maps to no code at all: a
#: correct plank reports an empty `form_reason_codes`, it does not report
#: "correct". See [scoring.form_error_vocab] in configs/argus.default.toml.
LABEL_TO_CODE = {"C": None, "L": "hips_sagging", "H": "hips_piked"}

#: Their deployed threshold (web/server/detection/plank.py). Below it they
#: return "unknown"; we emit no code, which is the same decision -- a
#: low-confidence guess is not worth an instructor's walk across the floor.
PREDICTION_PROBABILITY_THRESHOLD = 0.6

DATA_BASE = (
    "https://raw.githubusercontent.com/NgoQuocBao1010/"
    "Exercise-Correction/main/core/plank_model/"
)

OUT_ARTIFACT = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "plank_lr.json"
OUT_FIXTURE = REPO_ROOT / "tests" / "data" / "plank_vectors.json"


def fetch(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if not path.is_file():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(DATA_BASE + name, path)
    return path


#: What each landmark contributes to the feature vector, in this order.
#: Coordinates only -- see the module docstring on why visibility is excluded.
#: The artifact declares this so `PlankClassifier.kt` builds its vector from
#: the file rather than from a hardcoded stride.
FEATURE_KINDS = ("x", "y")


def feature_names() -> list[str]:
    return [f"{lm}_{s}" for lm in COCO_LANDMARKS for s in FEATURE_KINDS]


def load_xy(path: Path, encoding: str):
    """Read a labelled CSV into (features, labels) using `encoding`."""
    import numpy as np

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        missing = [
            f"{lm}_{s}" for lm in COCO_LANDMARKS for s in FEATURE_KINDS
            if f"{lm}_{s}" not in index
        ]
        if missing:
            raise SystemExit(f"{path.name}: missing expected columns {missing[:5]}")

        rows, labels = [], []
        for row in reader:
            xs = [float(row[index[f"{lm}_x"]]) for lm in COCO_LANDMARKS]
            ys = [float(row[index[f"{lm}_y"]]) for lm in COCO_LANDMARKS]
            if encoding == "bbox":
                xs, ys = bbox_normalize(xs, ys)
            feats = []
            for x, y in zip(xs, ys):
                feats.extend((x, y))
            rows.append(feats)
            labels.append(row[index["label"]])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels)


def bbox_normalize(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Re-express coordinates relative to the box the 13 landmarks span.

    Deliberately *not* the observation's `bbox_xyxy`: the training CSV has no
    detector box, and even if it did, MediaPipe's person box and YOLO26's are
    not the same convention -- normalising by them would train on one
    definition and infer on another. The landmark extent is computed the same
    way from the same 13 joints on both sides, which is what makes the phone
    able to reproduce this exactly. PlankClassifier.kt must derive it
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


def train(encoding: str, train_csv: Path, test_csv: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = load_xy(train_csv, encoding)
    x_test, y_test = load_xy(test_csv, encoding)

    scaler = StandardScaler().fit(x_train)
    # Multinomial (softmax) is the default for multi-class in modern sklearn;
    # the explicit `multi_class=` argument was removed in 1.7. This matters for
    # the Kotlin port, which implements softmax over all three classes rather
    # than one-vs-rest sigmoids.
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler.transform(x_train), y_train)

    predicted = model.predict(scaler.transform(x_test))
    accuracy = accuracy_score(y_test, predicted)

    print(f"\n=== encoding={encoding} ===")
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
    }


def write_artifact(result, path: Path) -> dict:
    model, scaler = result["model"], result["scaler"]
    artifact = {
        "format": "multinomial_logistic_regression",
        "artifact_version": 1,
        "encoding": result["encoding"],
        "source_dataset": "NgoQuocBao1010/Exercise-Correction (MIT), core/plank_model",
        "generated_by": "scripts/train_plank_model.py",
        "test_accuracy": round(result["accuracy"], 6),
        "test_accuracy_caveat": (
            "Held-out frames from the source dataset's own recordings, which "
            "feature a small number of people. Verified not to be frame-level "
            "leakage (nearest test-to-train distance 0.083, versus 1.607 "
            "between random training rows), but this is not a per-subject "
            "generalisation estimate and must not be quoted as accuracy on a "
            "new trainee. See docs/VALIDATION.md."
        ),
        "probability_threshold": PREDICTION_PROBABILITY_THRESHOLD,
        "feature_names": feature_names(),
        # Consumed by PlankClassifier.kt so the phone builds its feature vector
        # from the artifact rather than a hardcoded stride -- dropping a kind
        # here must not need a matching Kotlin edit to stay correct.
        "feature_kinds": list(FEATURE_KINDS),
        "landmarks": COCO_LANDMARKS,
        "classes": [str(c) for c in model.classes_],
        "class_to_code": {str(c): LABEL_TO_CODE[str(c)] for c in model.classes_},
        "scaler_mean": [float(v) for v in scaler.mean_],
        # Guard against the failure that made this rewrite necessary: a feature
        # whose training spread is vanishing turns any ordinary deviation into
        # hundreds of sigma and saturates the softmax. Coordinates never do
        # this; the dropped visibility columns did.
        "scaler_scale": [float(v) for v in scaler.scale_],
        "min_scaler_scale": float(min(scaler.scale_)),
        "coef": [[float(v) for v in row] for row in model.coef_],
        "intercept": [float(v) for v in model.intercept_],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}  ({len(artifact['coef'][0])} features)")
    return artifact


def write_fixture(result, artifact: dict, path: Path) -> None:
    """Reference (input -> probabilities -> code) triples for the Kotlin test.

    The Kotlin classifier is a reimplementation, so it needs the same pinning
    the decode path already has: cases drawn from real test rows, scored by
    the actual fitted model here, that `PlankClassifierTest` must reproduce.
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
            "probabilities": [float(p) for p in probabilities],
            "predicted_class": classes[best],
            "form_reason_codes": [code] if code else [],
        })

    document = {
        "vectors_version": 1,
        "generated_by": "scripts/train_plank_model.py",
        "why": (
            "PlankClassifier.kt reimplements this model's arithmetic in Kotlin. "
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "scratch",
                        help="where train.csv/test.csv live (downloaded if absent)")
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

    train_csv = fetch(args.data_dir, "train.csv")
    test_csv = fetch(args.data_dir, "test.csv")

    encodings = ["raw", "bbox"] if args.encoding == "both" else [args.encoding]
    results = [train(e, train_csv, test_csv) for e in encodings]

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
        print("\n=== comparison ===")
        for r in results:
            print(f"  {r['encoding']:<5} test accuracy {r['accuracy']:.4f}")
        print(f"shipping: {best['encoding']} (position-invariant; see the note in main())")

    artifact = write_artifact(best, OUT_ARTIFACT)
    write_fixture(best, artifact, OUT_FIXTURE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
