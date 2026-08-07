# The console

The page at `GET /`. It is the supervisor's live view of the floor: who needs
attention now, what every station is doing, and — the part no other surface
in the system can show — which stations have stopped saying anything.

Everything it draws comes from one endpoint, `GET /console`, polled at
`outputs.console_poll_interval_ms`. The page itself is
[`src/argus/console.py`](../src/argus/console.py): a single static HTML string
with no build step, no framework, and no network origin other than the server
that served it.

**One page, whatever the floor is running.** Everything below is use-case
neutral except where marked: the queue, the tiers, the silence states, and the
join requests are the same for a gym, a ward, and a welding bay. What differs
is the station card's drawing (`RENDERERS`, below) and, for fitness, the
per-exercise weight profile it explains. Fitness supplies most of the worked
examples here because it is the use case built out furthest — see
[`ADDING_A_USE_CASE.md`](ADDING_A_USE_CASE.md).

---

## What it shows

**Join requests**, above everything else, when the session runs
`[session] approval = "manual"`. Each row names the phone's offered label, its
station, the `trainee_id` it is claiming, and how long it has left before it
gives up — with Approve and Decline. It is the only blue on the page: a phone
waiting at the door is a trainee nobody is watching yet, which is urgent, but
it is not the same kind of urgent as a fall and must not borrow that colour.

Decisions go to `POST /join/decide`, which is **accepted only from this
machine**. The check is on the requesting client's address rather than on what
the server was bound to, so serving the console on a second screen keeps
working without handing the rest of the LAN the power to decide who monitors a
trainee. `outputs.allow_remote_join_control` opts out of that, and `argus
doctor` warns when it is on.

Under `"auto"`, the default, this section never appears — phones are admitted
on a well-formed `hello` exactly as they were before admission existed.

**The score an instructor reads is the rolling one.** The triage score is
instantaneous by design — computed off a ~2 s window, recomputed every
`ingest.rank_interval_s` — which makes it correct and nearly unwatchable: it
moves constantly and cannot be compared against the same trainee a minute
ago. The card's headline number is that score decayed over
`scoring.rolling_half_life_s`, labelled `session avg`, with `now` and `peak`
beside it.

Two things deliberately did **not** become rolling:

- **Alerts.** A fall averaged over twenty seconds is a fall nobody was told
  about. The red bar and the top tier of the queue are still the *instant*
  score crossing `scoring.alert_threshold`.
- **The peak.** A rolling mean is supposed to forget, and "this trainee was
  briefly in real trouble" should survive being forgotten.

**Work is counted in the unit the exercise actually has** *(fitness)*. Reps for a curl or
a squat; seconds for a plank, which has no reps and whose entire quality is
how long it was held well. One `fault_rate` then means the same thing for
both — "3 of 14 reps flagged", "18s of 2m10s flagged". Below
`scoring.min_reps_for_fault_rate` / `min_hold_s_for_fault_rate` the rate is
withheld and the card says so: one bad rep out of one is not a 100% fault
rate, and should not put anyone at the top of a queue.

**A trainee who cannot hold the movement is marked for help** *(fitness)*. This is the
one case the weighted sum could not express. With `form_error` at 0.15 in the
default vector, a trainee getting *every single squat rep wrong* scored
0.15 x 0.8 = 0.12 against a 0.5 threshold: no matter how badly or how long
they struggled, nobody was ever sent. Raising the weight is not the fix
either — enough to alert on form would make one bad frame outrank a fall.

So sustained failure is scored on its own terms and joins the score as a
floor, never as a sixth weighted term. Once a trainee has been wrong for the
*majority* of a meaningful stretch (`scoring.form_persistence_threshold`, over
`form_persistence_min_s`), they are escalated at the severity of the fault
they keep making, and the card reads **"Cannot hold form — wrong most of this
set"** rather than the per-rep "Form flagged by the phone". The severity
ladder still decides who is an alert: `hips_sagging` at 0.8 summons someone,
a persistent `incomplete_lockout` at 0.4 stays a coaching note.

It decays on `form_persistence_half_life_s`, so a trainee who corrects their
form clears within a minute or two — a coaching tool has to let people
recover, or a bad first set marks someone all session and the instructor
learns to ignore the flag. And it obeys the zero-weight rule: an exercise
profile that silences `form_error` silences this too.

**The help queue**, in three tiers that are deliberately not styled alike.
Alerts are ordered by the instant score; everything below is ordered by the
rolling one, with the fault rate breaking ties — two trainees can average the
same because the same non-form feature dominates both, and only the volume
distinguishes the one doing half their reps badly from the one doing them
cleanly:

| Tier | When | Why it is separate |
|---|---|---|
| Above alert threshold | `score >= scoring.alert_threshold` | The actual alert. Red is reserved for this. |
| Flagged, below threshold | a reason code fired but the score did not cross | Worth a glance, not an alarm. Styling a 0.20 like a 0.80 teaches a trainer to discount the colour, and then the colour stops working when it matters. |
| Not reporting | stale, disconnected, or evicted | Ranked *alongside* danger rather than below it — see below. |

**Station cards**, one per station, ordered by `trainee_id` and never by
score: a grid that reordered as scores moved would be unreadable, because a
supervisor watching one station would lose it mid-glance.

Each card carries whatever its use case has to draw (see **Drawing is
dispatched on `station.use_case`** below), the score and its reason codes in
prose, and a warm-up indicator while the rolling history is still filling. For
a pose use case that is a live COCO-17 skeleton drawn from `keypoints_xy` plus
the bounding box; for fitness it also carries the phone's own `exercise` /
`rep_count` / `form_ok` and **which scoring profile is running, and what it is
not watching for**.

That last one exists because of a trade fitness's scorer makes deliberately. A
correct plank is horizontal and motionless, which `fall` and `stillness` are
built to read as an emergency, so `[scoring.exercise_weights.plank]` weights
both at zero. A zero weight suppresses the feature's contribution *and* its
reason code, which is right — a reason that explains no part of the score is
worse than no reason. The cost is that a trainee who genuinely collapses
mid-plank raises neither, and a card reading "nothing flagged" would look
exactly like one where those checks had run and found nothing.

So the page says it, in amber, on the card:

> Scored as plank · not watching for falls, stillness

The console resolves the profile the same way `ScoringConfig.weights_for`
does — an exercise with no configured profile falls back to the default
vector, not to nothing — and it reads the weights from the snapshot rather
than knowing any of them, so a retuned profile changes the sentence without a
code change.

**Drawing is dispatched on `station.use_case`.** Every `StationView` carries
which use case its station is running (see `docs/PROTOCOL.md`), and
`console.py`'s `RENDERERS` map picks that use case's drawing function:

| Use case | Renderer | Why |
|---|---|---|
| `fitness` | `renderFitness` | The skeleton-and-bbox drawing described above |
| `nursing` | `renderFitness` | Shared because a nursing station streams the *same* COCO-17 pose — genuinely the same measurement, not one shape bent to fit the other. What differs is the labels around the canvas: a `procedure` where fitness shows an `exercise` |
| `welding` | `renderPlaceholder` | Clears the canvas. Welding has no classifier and nothing numeric to draw yet (see `argus.triage.compute_triage_welding`), so its card shows an empty frame rather than a skeleton drawn from fields that were never sent |

A new use case with something else to draw adds its own renderer here rather
than teaching `renderFitness` a second shape — and one with nothing to draw
yet points at `renderPlaceholder`, which is honest and takes no work. An
unregistered use case falls back to `renderFitness`, which is safe because it
draws nothing for a station with no `keypoints_xy`.

**Reason-code prose is a lookup with a fallback.** `PROSE` maps a code to a
sentence a human standing on a floor would understand; anything absent is
prettified mechanically (`cpr_rate_slow` renders as "Cpr rate slow"). That
fallback is adequate for a code that already reads as English and inadequate
for jargon, so a new use case should add its codes to `PROSE` — nursing's
`cpr_*` codes do not have entries yet and read as the mechanical fallback
today.

---

## Silence is the point

A station that has gone quiet and a station whose trainee is calm produce the
*same empty reason list*. The console is the only place in Argus where that
difference can be seen, so it is drawn loudly rather than quietly:

| State | Meaning | On screen |
|---|---|---|
| `live` | reporting, last observation newer than `outputs.console_stale_after_s` | green, full-colour skeleton |
| `silent` | socket open, but no observation for longer than that | amber, faded pose, "No frames for N s" |
| `dropped` | socket closed, inside the `ingest.track_ttl_s` grace window | red, counting down to eviction |
| `left floor` | session evicted; no longer scored | red, held briefly then removed |
| `no link` | this console cannot reach the server at all | red, every card, plus a banner |

Three further honesty rules follow from the same reasoning:

- **A frozen score does not look live.** A silent station keeps being scored
  off its frozen history, so it can still carry a reason code — but that code
  describes the last frames that arrived, not the trainee now. It is shown as
  "Last reading before it went quiet: …", greyed, never in alert red.
- **A trainee never simply vanishes.** When a session is evicted the card is
  held, marked, and only then removed. Disappearing without a word is exactly
  the quiet degradation this page exists to prevent.
- **A stalled server is reported.** If the HTTP server answers but its
  snapshot timestamp stops advancing — a dead rank loop — every age on the
  page would freeze and read as a calm floor. The page watches for that and
  says so.

---

## What the console is allowed to see

This is the one surface in Argus that reads something wider than a
`TriageRecord`, and it is worth being exact about what that does and does not
mean. A skeleton cannot be drawn from four scalar fields, so `GET /console`
also serves [`StationView`](../src/argus/outputs.py) — the live numeric
observation.

What that is **not**:

- **Not a frame path.** No image ever exists past the phone's own camera
  pipeline, so there is nothing here to redact. `argus.outputs` still imports
  no image library, and `tests/test_privacy.py` still asserts it by AST.
- **Not free text.** `form_reason_codes` is closed-vocabulary before
  `argus.ingest.protocol` accepts it. The phone-filled labels — `exercise` for
  fitness, `procedure` for nursing, plus the optional `display_name` — are the
  only free fields; each is length-bounded on the wire, never logged, and
  rendered as text and never as markup — `tests/test_console.py` asserts
  the page contains no markup-writing call at all.
- **Not a widening of the alert boundary.** `emit_alert`, `JsonLogSink`, and
  `GET /triage` carry the same four fields they always did. A `StationView`
  cannot reach any of them, because none of them has a parameter that names
  one.

`StationView`'s field set is pinned by `tests/test_privacy.py` exactly as
`TriageRecord`'s is, so widening what a trainer's screen can see stays a
visible, reviewable change.

**One operational consequence.** `outputs.http_host` defaults to `127.0.0.1`,
and that default matters more now than it did: opening it beyond loopback
publishes a live pose stream on the LAN, not just a ranked list of names.

---

## What it deliberately does not do

- **It does not re-derive anything the scorer computed.** No score is
  recalculated client-side and no trigger condition is re-implemented in
  JavaScript, so the page cannot drift from `argus.triage`. It gates which
  keypoints it draws on `scoring.keypoint_conf_threshold` for the same
  reason: the skeleton a trainer sees is the pose the rank was computed from.
- **It does not present the rank as tuned.** The weights have never been
  fitted to a real incident and no accuracy has been measured on this task
  ([VALIDATION.md](VALIDATION.md)), so the page says so on screen rather than
  only in a doc.
- **It does not correct for camera aspect ratio.** Coordinates are normalized
  to the phone's own frame and the protocol does not carry the frame's shape,
  so the console draws the [0,1] square as a square and labels it. Stretching
  it to a guessed aspect would misrepresent the geometry the scorer read.
- **It does not assume a shared clock.** Staleness is
  `snapshot.ts - station.last_seen_ts`, both on the *server's* clock. The
  phone's own `ts` is never used for elapsed time; phone and laptop clocks
  are not synchronised and nothing here pretends otherwise.

---

## Configuration

```toml
[outputs]
http_port = 8080                 # 0 disables the console entirely
http_host = "127.0.0.1"          # see the operational note above
console_poll_interval_ms = 200
console_stale_after_s = 2.0      # keep well under ingest.track_ttl_s
allow_remote_join_control = false

[session]
name = "Coach Riley — 6pm class" # shown in the beacon and in the header
approval = "auto"                # "manual" puts joins on this page first
join_timeout_s = 120.0
use_case = "fitness"             # what this floor is running; see docs/PROTOCOL.md
```

Note that `approval = "manual"` is a commitment to watch this screen: nobody
watching means phones queue, time out, and end up unmonitored while the system
itself reports no problem at all. `argus doctor` says so rather than leaving
it to be discovered.

**What this floor is running is a dropdown in the header**, next to the
session name — not just text, because it is also the control for changing
it. Its options come from the snapshot's `config.known_use_cases`, exactly
what `POST /session/use_case` will accept, rather than a list hardcoded into
the page that could drift from what the server can actually score. Choosing
a different one calls that route; on success every *future* `hello` is
checked against the new value (see `docs/PROTOCOL.md`), and on failure
(a use case this build has no scorer for, or a network hiccup) the select
snaps back to what it was and the reason is printed beside it — the same
inline-error pattern the join-request rows use, not a popup.

**This does not touch a phone already connected.** A trainee mid-session is
not retroactively reclassified because someone changed the dropdown — the
same reasoning that keeps `ingest.protocol_version` fixed at hello for an
already-open connection. Switching a floor from fitness to welding between
classes is the intended use; switching it under trainees who are still
streaming is not something this control tries to make safe, and every card
on screen at the moment of the change keeps scoring on the use case its own
phone agreed to.

This is a **`POST` route, guarded the same way `/join/decide` is**: accepted
only from this machine unless `outputs.allow_remote_join_control` is set,
because it decides who this floor will admit from now on — exactly the kind
of decision that flag already exists to keep off the rest of the LAN.

Every phone connected here has already passed `argus.ingest.protocol`'s
hello-time check against whatever `use_case` was in effect when it
connected (see `docs/PROTOCOL.md`); the header names the *current* setting
once rather than repeating it per card, and a phone connected before the
dropdown was last changed may be running a use case the header no longer
shows.

`argus doctor` warns if `console_stale_after_s` is not shorter than
`ingest.track_ttl_s` — set the other way round, a silent station is evicted
before the console ever draws it stale, so a trainee who went quiet leaves the
grid having only ever been shown as calm.

---

## Developing against it

No phone required:

```bash
argus demo --out fixture.json --ticks 900
argus run --http-port 8080 &
argus replay --fixture fixture.json --speed 1.0
```

`--speed 1.0` streams at the fixture's own pace, which is what makes the
console behave like a real floor rather than filling instantly. The synthetic
scene is a **fitness** floor (see `argus.synthetic`) and includes one station
reporting a form-error code, so the `form_error` path stays exercised in this
fixture without needing a phone physically present to develop the console
against. A nursing or welding card is driven by a phone or a hand-written
client today — there is no synthetic fixture for either.

To see the silent and dropped states, stop the replay part-way: the stations
stay in `ingest.track_ttl_s`'s grace window before being evicted.
