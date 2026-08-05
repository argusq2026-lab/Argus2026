# Validation gaps

What Argus has **not** been shown to do. Each entry says what is unverified,
why it matters, and what closing it would take. Nothing here is a known bug
— these are claims the product cannot currently make.

Ordered by how much a wrong assumption would cost.

---

## 1. The phone app does not exist yet

**Status:** blocking for everything else on this list.

This repository is the laptop side only. [`docs/PROTOCOL.md`](PROTOCOL.md)
specifies exactly what a phone app must send, and
[`demo/replay_client.py`](../demo/replay_client.py) is a reference
implementation of the client half of that protocol — but no on-device pose
model, no form/exercise classifier, and no real camera has ever driven this
server. Every test in this repository exercises the ingest -> triage -> alert
path against a synthetic observation fixture (`argus.synthetic`), not a real
trainee.

**To close:** build the phone app against `docs/PROTOCOL.md`; pick and
validate an on-device pose model and a form/exercise classifier; run it
against real trainees and confirm the wire contract holds up (frame rate,
keypoint layout, timestamp behaviour) under a real Wi-Fi network, not a
loopback socket in a test.

**Owner decision needed:** which on-device pose model and form-classifier
architecture, and which specific form errors the closed vocabulary in
`[scoring.form_error_vocab]` should actually cover for the exercises this
floor runs — the nine entries shipped in the default config are
illustrative, not the result of any domain review.

---

## 1b. The plank classifier's 99.0% is not an accuracy claim about a trainee

**Status:** blocking for any claim about form detection.

`scripts/train_form_model.py --exercise plank` refits the plank classifier from
[NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)
(MIT) on the 26 of its 68 features a COCO-17 phone can reproduce — x and y for
13 landmarks. Their set also includes heels and foot indices COCO does not
have, a `z` depth estimate YOLO26-pose does not emit, and a MediaPipe
*visibility* per landmark that is excluded deliberately (see "the feature that
cost a rebuild" below). It reports **0.9901** on their held-out split.

That number is real and it is nearly meaningless for this product. It was
measured on held-out *frames* from the same small set of recordings the
training data came from. Frame-level leakage was checked and ruled out (the
nearest test row sits 0.083 from any training row, against 1.607 between two
random training rows), so the model is not simply memorising — but "a frame of
a person the model has seen in other frames" is not "a trainee it has never
seen". Nothing here establishes it generalises across people, body types,
camera heights, or clothing.

Two narrower gaps sit underneath it:

* **The three-class taxonomy is theirs, not a domain review.** Correct /
  low-back / high-back is what their data was labelled with. Whether those are
  the plank faults worth interrupting a class for is an unexamined
  inheritance. Worse, a three-class softmax has no fourth answer: shown
  something that is not a plank at all it must still name one of the three.
  `FormClassifier` therefore gates on landmark visibility rather than trusting
  the probability, which bounds the failure without measuring it.
* **`bbox` encoding ships despite measuring *lower*.** Raw absolute
  coordinates score 0.9930 against landmark-box-relative features' 0.9901, and
  the higher number is the one to distrust: a model fitted on absolute
  coordinates partly learns where in the source recordings' frame a person
  stood, and the test split comes from those same recordings, so it rewards
  exactly the memorisation that fails on a phone at a different height and
  distance. The reasoning is sound and unmeasured; nothing here demonstrates
  `bbox` is better where it matters.

**To close:** record real trainees planking, from real station placements, and
measure per-subject held-out accuracy — the split has to be by person, not by
frame. Until then the classifier is wiring that works, not a verified signal.

### The feature that cost a rebuild

The first revision kept the visibility columns, on the assumption that
MediaPipe visibility and YOLO26 keypoint confidence are the same kind of
number. They are not. Visibility saturates: `nose_v` has a training mean of
0.9993 and a standard deviation of **0.0013**, so a perfectly ordinary
YOLO26 confidence of 0.85 standardises to −116 sigma, and thirteen such
features saturate the softmax before the pose is consulted at all. On device
this reported a form error for every plank it was shown. Substituting
realistic confidences into a known-correct fixture row, coordinates untouched,
flips the prediction from correct at 100% to `hips_sagging` at 100%.

The general form of the mistake is worth recording, because no accuracy metric
on the source dataset could have caught it: a feature only transfers between
two pose models if both mean the same thing by it. Coordinates do. Confidence
does not, and its *units* agreeing is not evidence that its distribution does.
`tests/test_plank_artifact.py` now rejects any feature whose training spread is
under 0.01 — a floor that bounds a [0, 1] feature's worst case at ~100 sigma —
so a near-constant column cannot enter the model again unnoticed.

---

## 1c. The bicep curl classifier: same caveats as plank, plus a weak model

**Status:** blocking for any claim about form detection; worse than plank's.

`scripts/train_form_model.py --exercise bicep` refits on
[NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)'s
`core/bicep_model`, their binary label for "lean too far back" (their other
two bicep faults, loose upper arm and weak peak contraction, are threshold
checks in their own code, not this dataset). All 9 of their landmarks exist
in COCO-17, so no landmark loss beyond the structural `z`/visibility drop.

It reports **0.629** test accuracy -- barely past a coin flip for a binary
label, and well below plank's 0.99. Two things compound here, beyond the
frame-vs-subject caveat in §1b (which applies unchanged: this is held-out
frames from the same few recordings, not a new trainee):

* **`nose` had to be dropped.** Under `bbox` encoding, `nose_y` came in with a
  training spread of 0.0083 -- under the 0.01 floor
  `test_no_feature_has_a_saturating_scale` enforces, and the same failure
  class as plank's withdrawn visibility feature, from geometry this time
  instead of confidence: the head sits at the top of the landmark box on
  nearly every frame of a curl, so bbox-normalized `nose_y` is almost
  constant. With it, accuracy measured 0.841; without it, 0.629. That drop is
  the honest cost of the guard doing its job, not a bug to chase away.
* **`raw` measures far worse than `bbox` here** (0.555 vs 0.629), a wider gap
  than plank's, plausibly because a curl's genuinely diagnostic cue (torso
  lean) is a small angle change that raw absolute coordinates drown in
  frame-position noise the way plank's did -- unconfirmed, since there is no
  real footage to check it against.

**To close:** same as §1b, plus specifically: this model is not accurate
enough to trust today even on its own dataset's terms. Before shipping it
against real trainees, either find a cleaner feature set (upstream's own
"loose upper arm"/"weak peak contraction" checks are threshold-based, not
ML -- a geometric approach in the spirit of §5's squat recommendation may
simply be the better fit for bicep's diagnostic cue too) or collect real
data and refit.

---

## 1d. The lunge classifier: a label that only ever meant "at the bottom of a rep"

**Status:** blocking for any claim about form detection.

`scripts/train_form_model.py --exercise lunge` refits on
[NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)'s
`core/lunge_model`, the `err.*` split (knee-over-toe), not `stage.*` (init /
mid / down) -- Argus has no rep-phase concept and does not need one. 9 of
their 13 landmarks survive into COCO-17 (heels and foot indices do not). It
reports **0.999** test accuracy, but that number is close to meaningless on
its own, for a reason worth recording precisely because it was checked before
fitting rather than discovered after:

**The knee-over-toe label was only ever collected, and is only ever
evaluated upstream, at the bottom of a lunge.** Their own detection notebook
(`core/lunge_model/9.err.detection.ipynb`) only calls the error model when a
separate stage classifier reports `current_stage == "down"`. Measuring
`mean(ankle_y) - mean(hip_y)` (a depth proxy, in raw normalized-image
coordinates) confirms this empirically: `err.train.csv` has mean 0.144
(sd 0.027), matching `stage.train.csv`'s own `D` rows (mean 0.158, sd 0.029)
almost exactly, versus 0.376 for `I` (init) and 0.285 for `M` (mid). The
classifier has never seen a standing or ascending trainee under either label.

A softmax has no "none of the above" (Trap 2 in
`docs/ADDING_AN_EXERCISE.md`), so naively wiring this model in would
confidently name `C` or `L` for every frame of a lunge, including the 2/3 of
each rep the label was never fit against. `FormClassifier.kt` gates on this
with a `depth_gate`: the same `ankle_y - hip_y` gap, bounded to the training
data's own observed range (**[0.0951, 0.3014]**, not a guessed threshold),
checked before classification alongside the landmark-visibility gate. Outside
it, the classifier declines rather than guessing -- see
`test_a_pose_outside_the_depth_gate_is_refused` in `FormClassifierTest.kt`
and `test_depth_gate_bounds_come_from_training_data_not_a_guess` in
`tests/test_form_artifacts.py`.

**What this does not close:** the gate bounds the *input pose*, not the
*label's correctness*. Nothing here establishes that "at the bottom of a
lunge, in these recordings" generalises the way plank's per-subject gap does
in §1b -- that caveat applies here too, unchanged, and is probably worse
given upstream's own data collection was itself scoped to one moment of the
rep by design.

**To close:** same as §1b (real trainee footage, split by person). Additionally,
confirm on-device that the depth gate's bounds -- fit on MediaPipe recordings,
applied to YOLO26 keypoints -- actually track "bottom of a lunge" and not some
artifact of the source dataset's camera placement; nothing here checked that
beyond the internal consistency of upstream's own two datasets.

---

## 2. The five scoring weights have never been fitted to an incident

**Status:** blocking for operational trust in the rank.

`fall 0.40 / stillness 0.20 / occlusion 0.15 / form_error 0.15 / off_task
0.10` are the prototype author's priors, carried forward unchanged through
the move to phone-based ingestion. No incident has ever been scored with
them. `alert_threshold = 0.5` is equally unvalidated: nobody has measured how
many alerts an instructor would receive per hour, or what fraction would be
real.

`[scoring.exercise_weights.plank]` is newer and no better evidenced. Zeroing
`fall`, `stillness`, and `off_task` for a plank follows from geometry — those
features describe a correct plank, which was measured — but the 0.85 given to
`form_error` was picked so that the two plank codes land either side of the
threshold (`hips_sagging` 0.68, `hips_piked` 0.51), not from any observation
of what an instructor would want to be called over for. `hips_piked` clearing
the threshold by 0.01 is arithmetic, not a judgement about piked planks.

`[scoring.exercise_weights.bicep]` and `.lunge]` are weaker still: rather than
independently reasoning through fall/stillness/off_task the way plank's
profile did (see `docs/ADDING_AN_EXERCISE.md` §4 Trap 3), both simply copy
plank's numbers — occlusion 0.15, form_error 0.85, everything else zeroed.
Bicep has a real case for keeping `fall`/`off_task` at their defaults (a curl
is close to the "standing HIIT" case those features were written for) and
lunge has a real but different case for zeroing only `fall` (a full-depth
lunge viewed side-on plausibly reads bbox-wider-than-tall). Both were
overridden by the same practical constraint: the vocabulary weight the code
carries (`lean_back_error` 0.7, `knee_over_toe` 0.6) does not clear
`alert_threshold` at a partial profile's lower `form_error` share, so a
correctly-flagged rep would silently never alert. Copying plank's numbers is
the same class of arithmetic-driven choice as `hips_piked`'s 0.01 margin
above, made explicit here rather than discovered later:
`test_the_vocab_weight_and_profile_together_clear_the_alert_threshold` in
`tests/test_exercise_profiles.py` pins that the chosen numbers actually clear
the bar; it does not claim they are the *right* numbers.

The weights live in `configs/argus.default.toml` rather than in code, so
retuning is a config edit and a `config_version` bump — the mechanism is
ready, the evidence is not.

**To close:** build a labelled-clip harness once real phone data exists: a
corpus of observation streams each labelled with what happened and whether an
instructor was in fact needed, a replay path that emits the full per-feature
breakdown (not just the final score), and a fitting step that searches the
weight simplex against a chosen operating point.

**Owner decision needed:** what counts as "needed an instructor" — the label
definition is the experiment, and it cannot be inferred from the code.

---

## 3. The WebSocket ingest server has no authentication or transport security

**Status:** open risk for any deployment beyond a trusted private LAN.

`ws://` is plaintext, and any device that can reach `ingest.ws_port` can send
a `hello` claiming *any* `trainee_id` — there is no credential binding a
specific phone to a specific trainee identity, only the "first connection to
claim an id wins, and it's exclusive while connected" rule in
`argus.ingest.session`. That rule stops two simultaneous claims from
colliding silently; it does not stop a misconfigured or malicious device on
the same network from impersonating a trainee it has no relationship to.

**To close:** at minimum, run this on a network the gym controls (not open
Wi-Fi); for anything beyond that, add `wss://` (TLS) and a per-phone
credential (e.g. a token issued at trainee check-in) checked at `hello`
time — the protocol has a natural extension point there, since `hello`
already carries identity claims.

---

## 4. No cross-phone clock synchronization

**Status:** open risk for cross-trainee timing comparisons.

Each `observation` message's `ts` is the phone's own clock
(`docs/PROTOCOL.md`); nothing enforces or measures agreement between two
phones' clocks. The merged rank's own timestamp comes from the laptop's
clock at each rank tick, so *ranking* is unaffected — but any feature that
someday compared two trainees' event timing directly (e.g. "did they start
their rep within N ms of each other") would inherit whatever clock skew
exists between their phones, unmeasured.

**To close:** if a feature ever needs cross-trainee timing precision, add an
NTP-style offset estimate to the handshake and either correct or flag skew
beyond a threshold.

---

## 5. Ingest capacity under many concurrent phones is unmeasured

**Status:** blocking for a capacity claim ("how many trainees per laptop").

The ingest server is a single asyncio event loop; `IngestServer.tick()`
recomputes the full merged rank every `ingest.rank_interval_s` over every
connected session. Nothing about this has been load-tested: not the number
of concurrent WebSocket connections one process can hold open, not the CPU
cost of `rank_trainees` at real class sizes, not behaviour when a rank tick
takes longer than `rank_interval_s` to compute (there is no back-pressure or
skip-if-still-running guard on the periodic task today).

**To close:** run a synthetic load test — many concurrent
`demo/replay_client.py`-style connections streaming at a realistic rate —
and measure rank-tick latency and memory as trainee count grows; add a
guard against overlapping ticks if it becomes an issue in practice.

---

## 6. No accuracy figure exists for pose or form/exercise classification

**Status:** inherited from §1 — there is no model to measure yet.

Whatever on-device pose model and form classifier the phone app eventually
uses will need its own accuracy validation (keypoint error, exercise
classification precision/recall, false-positive rate on
`form_reason_codes`) against real trainee footage before any claim about
correctness can be made. None of that exists today because the model choice
itself hasn't been made.

**Owner decision needed:** same as §1 — this is one gap, not two, restated
here because it is the reason §2's weight-fitting work can't start yet
either: there is no real per-feature signal to fit weights against until a
real classifier exists.
