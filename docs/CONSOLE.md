# The trainer console

The page at `GET /`. It is the trainer's live view of the floor: who needs
attention now, what every station is doing, and — the part no other surface
in the system can show — which stations have stopped saying anything.

Everything it draws comes from one endpoint, `GET /console`, polled at
`outputs.console_poll_interval_ms`. The page itself is
[`src/argus/console.py`](../src/argus/console.py): a single static HTML string
with no build step, no framework, and no network origin other than the server
that served it.

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

**The help queue**, in three tiers that are deliberately not styled alike:

| Tier | When | Why it is separate |
|---|---|---|
| Above alert threshold | `score >= scoring.alert_threshold` | The actual alert. Red is reserved for this. |
| Flagged, below threshold | a reason code fired but the score did not cross | Worth a glance, not an alarm. Styling a 0.20 like a 0.80 teaches a trainer to discount the colour, and then the colour stops working when it matters. |
| Not reporting | stale, disconnected, or evicted | Ranked *alongside* danger rather than below it — see below. |

**Station cards**, one per trainee, ordered by `trainee_id` and never by
score: a grid that reordered as scores moved would be unreadable, because a
trainer watching one station would lose it mid-glance.

Each card carries a live COCO-17 skeleton drawn from `keypoints_xy`, the
bounding box, the score and its reason codes in prose, the phone's own
`exercise` / `rep_count` / `form_ok`, and a warm-up indicator while the
rolling history is still filling.

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
  `argus.ingest.protocol` accepts it. `exercise` is the single field a phone
  fills freely; it is length-bounded on the wire, never scored, never logged,
  and rendered as text and never as markup — `tests/test_console.py` asserts
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
http_port = 8080                # 0 disables the console entirely
http_host = "127.0.0.1"         # see the operational note above
console_poll_interval_ms = 200
console_stale_after_s = 2.0     # keep well under ingest.track_ttl_s
allow_remote_join_control = false

[session]
name = "Coach Riley — 6pm HIIT" # shown in the beacon and in the header
approval = "auto"               # "manual" puts joins on this page first
join_timeout_s = 120.0
```

Note that `approval = "manual"` is a commitment to watch this screen: nobody
watching means phones queue, time out, and end up unmonitored while the system
itself reports no problem at all. `argus doctor` says so rather than leaving
it to be discovered.

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
scene includes one trainee reporting a form-error code (see
`argus.synthetic`), because no phone reports one yet — without it the
`form_error` path would be unexercised in the only fixture the console is
developed against.

To see the silent and dropped states, stop the replay part-way: the stations
stay in `ingest.track_ttl_s`'s grace window before being evicted.
