# Adding an exercise

How the plank form classifier was built, end to end, written so the next one
does not have to rediscover it. Read this before adding squat, lunge, or bicep
curl.

It is deliberately blunt about what went wrong, because two of the four
mistakes below produced a system that *looked* like it worked — high test
accuracy, confident probabilities, a clean dashboard — and were only caught by
pointing a real phone at a real person.

Reference implementation: commits `5fd7525` (server) and `01ce651` (phone) on
`plank-form-classification`.

---

## 1. The shape of the thing

Form judgement happens **on the phone**. The laptop never sees a pose well
enough to judge form — it scores fall, stillness, occlusion and orientation
from keypoint *history*, and takes form as a closed-vocabulary verdict the
phone hands it. So an exercise needs work in two places, and they are joined
only by two strings: the `exercise` label and the `form_reason_codes`
vocabulary.

```
  labelled CSV (upstream)                     [offline, once]
        │
        │  scripts/train_plank_model.py
        ▼
  android/app/src/main/assets/plank_lr.json   ← coefficients, not a pickle
  tests/data/plank_vectors.json               ← pins the Kotlin reimplementation
        │
        │  PlankClassifier.fromJson()          [phone, every frame]
        ▼
  MainActivity.classifyForm() ──► Observation{exercise, form_reason_codes}
        │
        │  WebSocket                           [wire]
        ▼
  argus.ingest.protocol.parse_observation     ← rejects codes outside the vocab
        │
        ▼
  argus.triage.compute_triage                 [laptop, every rank tick]
        └─ cfg.weights_for(exercise) picks the weight vector
```

| Concern | File |
|---|---|
| Fit the model, emit artifact + fixture | `scripts/train_plank_model.py` |
| Evaluate it on-device | `android/.../PlankClassifier.kt` |
| Wire it into the frame loop | `android/.../MainActivity.kt` (`classifyForm`, `openPlankClassifier`) |
| Codes the server will accept | `[scoring.form_error_vocab]` in `configs/argus.default.toml` |
| Which features count for this exercise | `[scoring.exercise_weights.<name>]` in the same file |
| Pin the Kotlin arithmetic | `android/.../PlankClassifierTest.kt` |
| Pin the artifact in CI without a JDK | `tests/test_plank_artifact.py` |
| Prove the profile fixes the misfire | `tests/test_exercise_profiles.py` |

---

## 2. The recipe

### Step 1 — Check the upstream data actually contains form errors

Not all of it does. See §5: their **squat** model classifies `up`/`down`, which
is a rep-phase classifier, not a form classifier. Doing the plank recipe on it
would produce a model that confidently reports which half of a squat someone is
in, mapped to form codes that mean nothing.

Read `core/<exercise>_model/README.md` upstream and check the label column
before writing any code:

```python
import csv, io, urllib.request, collections
B = "https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/"
rows = list(csv.reader(io.StringIO(
    urllib.request.urlopen(B + "<exercise>_model/test.csv").read().decode())))
print(rows[0][:5], collections.Counter(r[0] for r in rows[1:]))
```

### Step 2 — Work out which of their features survive

Their CSVs carry `x, y, z, v` per landmark. **Keep `x` and `y`. Drop `z` and
`v`, always** — see §4. Then drop every landmark COCO-17 does not have:
`left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`.

Surviving feature count is `2 × (landmarks ∩ COCO-17)`. For plank that was 26
of their 68. If fewer than ~6 landmarks survive, stop and reconsider — the
model may not have enough of the body left to separate the classes.

### Step 3 — Choose codes and add them to the vocabulary

`form_reason_codes` is a **closed vocabulary**. The server closes the
connection on an unrecognised code, so a classifier that can emit one is a
broken station, not a bad guess. Add entries to `[scoring.form_error_vocab]`
with a severity weight, and bump `config_version`.

Two rules learned on the plank:

- **The correct class maps to no code**, not to a `"correct"` code. The scorer
  takes the *max* weight over the codes; a code meaning "fine" would have to be
  weighted 0 and would still appear on the dashboard as though something were
  flagged.
- **Severity is a claim about injury, not about model confidence.**
  `hips_sagging` (0.8) outranks `hips_piked` (0.6) because lumbar
  hyperextension under load is the injury mechanism and piking is merely an
  ineffective hold.

### Step 4 — Fit, and ship coefficients rather than a pickle

Copy `scripts/train_plank_model.py`. A multinomial logistic regression is a
standardise, a matrix multiply, and a softmax, so the phone evaluates it as
Kotlin arithmetic — no ONNX export, no ML runtime, no second NPU dispatch on a
path where dispatch alone costs ~500 µs, and no third artifact to stage on the
device. Keep that property. If an exercise genuinely needs a non-linear model,
that is a real decision with real cost, not a drop-in.

Emit both files from **one run**: the artifact (`assets/<exercise>_lr.json`)
and the reference fixture (`tests/data/<exercise>_vectors.json`). They cannot
then drift apart, and committing only one is a mistake CI notices.

The training run must be **byte-reproducible** — re-running the script on the
same data must produce identical files. Verify it; the plank's is.

### Step 5 — Port the arithmetic and pin it

`PlankClassifier.kt` is a reimplementation of a model fitted in Python, which
is exactly the situation that produces plausible-looking wrong answers: a
transposed feature order, a skipped standardisation, or one-vs-rest sigmoids
instead of a softmax all still yield numbers in [0, 1] that sum to 1. Only
comparison against the fitted model catches it, which is what
`plank_vectors.json` and `PlankClassifierTest` are for. Reproduce to 1e-6.

Do not forget the **max-shift in the softmax**. Without it, `exp` of a large
logit overflows to infinity on a degenerate pose, every probability comes back
NaN, NaN compares false against the threshold, and the classifier silently
means "never report a form error". There is a test for this.

### Step 6 — Give the exercise a weight profile

This is the step that has nothing to do with the model and is the one most
likely to be skipped. See §4, third trap.

### Step 7 — Verify on a device, not in a test

The plank passed every test it had while being completely broken on a phone.
See the checklist in §7.

---

## 3. What has to be generalised first

The current code is **hardcoded to plank in four places**. Someone adding a
second exercise has to decide whether to generalise or to copy-paste, and the
honest answer depends on how much time is left:

| Hardcoded | Where |
|---|---|
| `EXERCISE = "plank"`, `describe()` prefixes `"plank:"` | `PlankClassifier.kt` companion |
| `classifyForm` returns empty unless `exercise == PlankClassifier.EXERCISE` | `MainActivity.kt:575` |
| Asset name `"plank_lr.json"` | `MainActivity.openPlankClassifier()` |
| `COCO_LANDMARKS`, `LABEL_TO_CODE`, `DATA_BASE`, output paths as module constants | `train_plank_model.py` |

**The clean version** is a `FormClassifier` (rename; the arithmetic is not
plank-specific — it already reads landmarks, feature kinds, classes, and the
code mapping from the artifact) plus a registry mapping `exercise` → asset
name, loaded lazily. The training script becomes one exercise-spec table and a
`--exercise` flag. The two test files iterate over artifacts rather than naming
one. This is maybe half a day and it is the right shape.

**The fast version** is a second classifier class and a second training script,
copied. It works, and it makes the third exercise worse. If you take it, take
it knowingly — and note that `MainActivity` will need a real dispatch on
`exercise` either way, so you pay part of the cost regardless.

`PlankClassifier`'s *loading and validation* logic is already exercise-agnostic
and worth keeping verbatim: the artifact declares its own landmarks, feature
kinds, classes, and code mapping, so the Kotlin builds its feature vector from
the file rather than from a hardcoded stride. Dropping a feature kind in Python
does not require a matching Kotlin edit to stay correct. Preserve that.

---

## 4. The four traps

### Trap 1 — Visibility is not confidence *(cost a full rebuild)*

Their feature set has a MediaPipe **visibility** per landmark. It looks exactly
like YOLO26's keypoint confidence: same name-ish, same [0, 1] range, same
apparent meaning of "did we see it". It is not the same quantity.

Visibility saturates. On this dataset `nose_v` has mean **0.9993** and standard
deviation **0.0013**. A `StandardScaler` fitted on that turns a perfectly
ordinary YOLO26 confidence of 0.85 into a **−116 sigma** feature (worst case
−546). Thirteen such features saturate the softmax before the pose is consulted
at all.

The symptom was a form error on every plank, at 100% confidence. Proven
offline: taking a known-correct row from the fixture and substituting realistic
confidences, **coordinates untouched**, flips the prediction from correct at
100% to `hips_sagging` at 100%.

No accuracy metric on the source dataset could have caught this. The model was
99.6% correct on data where the feature behaved as trained.

> **The general rule:** a feature transfers between two pose models only if both
> models mean the same thing by it. Coordinates do. Confidence does not, and its
> *units* agreeing is not evidence that its *distribution* does.

Guarded by `test_no_feature_has_a_saturating_scale`, which rejects any feature
with training spread under 0.01 — a floor that bounds a [0, 1] feature's worst
case at ~100 sigma. **Keep this test when you copy the script.**

### Trap 2 — A softmax has no "none of the above"

Three-class softmax over correct/piked/sagging always returns one of the three,
and on input unlike anything it was fitted on it can be arbitrarily certain.
Observed on a real device: a photograph of a single leg, 8 of 17 keypoints,
scored `hips_piked` at 100%.

**The probability threshold does not help.** It separates ambiguity *between the
classes*; it says nothing about whether the input is the exercise at all. Those
are different questions and only one of them is in the model.

So gate on **evidence** instead: `MIN_VISIBLE_LANDMARKS = 10` of 13 above
`MIN_LANDMARK_CONFIDENCE = 0.3`, or the classifier declines to answer. The
threshold matches the server's own `keypoint_conf_threshold` so both ends agree
what "visible" means. Tune the count per exercise — it should keep a normal
side-on view with occluded far-side limbs, and reject a partial body.

With no footage to tune against, err conservative: a missed form error costs
less than a fabricated one, because a fabricated one outranks a trainee who
genuinely needs help.

### Trap 3 — The triage features may be wrong for your exercise

**This is not a model problem and it will bite you separately.**

The five scoring features were written for standing HIIT, where "horizontal"
and "not moving" are good evidence of trouble. A correct plank is horizontal
*and* motionless *and* oriented away from the station-facing reference — so
`fall` (bbox wider than tall), `stillness` (centroid not moving), and
`off_task` (shoulder line off the reference angle) all read a textbook rep as
an emergency.

Measured on the default weights, a correct plank scored **0.42** against a 0.5
threshold and displayed `prolonged_stillness, off_task_orientation`, with the
real form signal worth 0.12 against 0.42 of noise. The classifier was working
perfectly and the dashboard was still wrong.

Fix is `[scoring.exercise_weights.<name>]` — a *complete* weight vector, same
contract as the default: every feature named, non-negative, summing to 1.0. A
zero weight both removes the contribution **and** suppresses the reason code,
because a reason that explains no part of the score is worse than no reason.

Before writing a profile, ask of each of the five features: *is this measuring
what it thinks it is, during this movement?* For plank the answer was no for
three of five. Occlusion survived at its default weight — a trainee the phone
cannot see is worth flagging whatever they are doing.

Guarded by `tests/test_exercise_profiles.py`. Copy its structure; it pins the
actual misfire, not just the config plumbing.

### Trap 4 — Watch what the threshold arithmetic implies

The plank profile gives `form_error` 0.85, which puts `hips_sagging` at 0.68
(alerts) and `hips_piked` at 0.51 (alerts, by 0.01). That margin is arithmetic,
not a judgement that piked planks are worth interrupting a class for. Check
where each new code lands relative to `alert_threshold` and decide deliberately
whether that is what you meant. Record it in `docs/VALIDATION.md`; §2 there
already carries the plank's.

---

## 5. What is actually in the upstream data

Measured 2026-08-04 from `NgoQuocBao1010/Exercise-Correction@main` (MIT). All
CSVs carry `x, y, z, v` per landmark; the "survives" column is landmarks in
COCO-17, so features = 2 × that.

| Exercise | CSVs | Label column | Landmarks | Survives | Upstream calls it |
|---|---|---|---|---|---|
| **plank** | `train.csv` / `test.csv` | `C` / `L` / `H` | 17 | **13** → 26 feats | *all errors* — **done** |
| **bicep** | `train.csv` / `test.csv` | `C` / `L` | 9 | **9** → 18 feats | *lean back error* |
| **lunge** | `err.train.csv` / `err.test.csv` | `C` / `L` | 13 | **9** → 18 feats | *knee over toe error* |
| **lunge** | `stage.train.csv` / `stage.test.csv` | `I` / `M` / `D` | 13 | **9** → 18 feats | *stage* — not a form model |
| **squat** | `train.csv` / `test.csv` | `up` / `down` | 9 | **9** → 18 feats | *stage* — **not a form model** |

Base URL: `https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/main/core/<exercise>_model/`

**Recommended order:**

1. **Bicep curl** is the closest analogue to plank — a genuine binary form
   classifier, all 9 landmarks survive intact, no `z`/heel losses at all. If
   you are generalising the code (§3), do it here where the second case is
   easy.
2. **Lunge** works too, via `err.*`, losing heels and foot indices. Note it has
   a *second* model for rep stage; Argus has no rep-phase concept and does not
   need one — take `err.*` only, unless the error turns out to be
   stage-dependent, which is worth checking before fitting.
3. **Squat needs a different approach entirely.** Their squat *model* is a
   stage classifier. Their squat *errors* (foot placement, knee placement) are
   detected geometrically — distance ratios between joints against thresholds,
   documented in `core/README.md` §1 — not by a fitted model. That is a
   perfectly good approach and it is not this pipeline. It would be a
   `SquatGeometry.kt` computing ratios, with no artifact, no scaler, and no
   fixture, and thresholds that are *unvalidated priors* and must be recorded
   as such in `docs/VALIDATION.md`. Do not force it into the LR shape.

Also true of the whole corpus, and inherited by anything built on it: their
accuracy figures are on held-out **frames** from the same few recordings as the
training data. Frame-level leakage was ruled out for plank (nearest test row
0.083 from any training row, versus 1.607 between random training rows), but
that is not per-subject generalisation. Nothing here establishes any of these
models work on a trainee they have never seen. See `docs/VALIDATION.md` §1b —
write the equivalent section for each new exercise, and do not quote the
number without the caveat.

---

## 6. Conventions that are easy to get wrong

- **COCO-17 left/right is the subject's own**, so a camera-facing trainee's
  *left* shoulder is at the *larger* image x. The prototype had this backwards
  and its `off_task_reference_angle_deg` default of 0.0 looked correct in tests
  while flagging every attentive trainee.
- **`bbox` normalisation uses the landmark extent, not `bbox_xyxy`.** The
  training CSVs have no detector box, and MediaPipe's person box and YOLO26's
  are not the same convention — normalising by them would train on one
  definition and infer on another. Both sides compute the extent from the same
  joints. `PlankClassifier.normalizeToLandmarkBox` and
  `train_plank_model.bbox_normalize` must stay identical.
- **`bbox` ships even though `raw` measures higher** (0.9930 vs 0.9901). The
  extra point is earned by learning where in the source recordings a person
  stood, scored on a test split drawn from those same recordings. A metric
  measured on data that cannot expose the flaw is not evidence.
- **Send `exercise` on every observation**, not only when it changes. The
  server tracks the most recent value and cannot distinguish "still planking"
  from "stopped reporting".
- **`exercise` is open, `form_reason_codes` is closed.** An unconfigured
  exercise scores on the default weights; an unrecognised code closes the
  connection. Version skew is what the closed vocabulary exists to catch; a
  free-form label cannot express it.
- **Clear the verdict when nobody is in frame.** `lastVerdict = null` on an
  absent subject, or the UI shows the last person's form for the next one.

---

## 7. Verification checklist

Tests passing is necessary and was not sufficient — the visibility bug passed
everything.

1. `pytest tests/ -q` — currently 162.
2. `cd android && ./gradlew testDebugUnitTest` — currently 37.
3. Re-run the training script; artifact and fixture must be **byte-identical**.
4. Check `min_scaler_scale` in the artifact. Under 0.01 means a near-constant
   feature and the model will not survive the change of pose model.
5. **Point a phone at a real person doing the exercise correctly.** Confirm the
   on-device verdict, then confirm `/triage` scores it **0.0 with no reason
   codes**. This is the step that caught both device-level bugs.
6. Do the same for each error class, and confirm the score matches
   `vocab_weight × form_error_weight`.
7. Point it at something that is *not* the exercise — a partial body, a chair —
   and confirm it declines rather than guessing.

---

## 8. Known open issue

**A single-frame misclassification propagates straight to an alert.** Sampling
the live rank over a correct plank, one tick in five came back **0.51 with
`form_error`** while the rest were 0.0. `TrackState` keeps only the *latest*
form verdict, where `fall` and `stillness` are smoothed over the history
window. For a demo this means an occasional false red on a trainee holding good
form.

The fix is server-side in `argus.triage` — require a code to persist across
several observations before it scores — and it is **not exercise-specific**.
Whoever gets there first should do it once, for all exercises. It changes
scoring semantics, so it needs a decision on how many consecutive observations
count and whether the window is time- or count-based.
