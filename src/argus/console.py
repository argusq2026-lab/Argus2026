"""The trainer console page, served at `GET /`.

One module-level string. It lives here rather than in `argus.outputs` so that
module stays readable, and rather than in a data file so it is impossible for
a wheel or a PyInstaller bundle to ship a working `argus` binary with a
missing console — the page is code, so it travels wherever the code does.

The page is static and reads exactly one endpoint, `GET /console`. It holds no
state the server does not give it, and it cannot ask for anything else: there
is no route that serves a frame, because no frame exists anywhere past the
phone's own camera. What it draws is the same numeric observation the scorer
reads, gated on the same `keypoint_conf_threshold` — so a trainer looking at a
skeleton is looking at the pose the rank was computed from, not a prettier one.

Three things it must never stop doing, in rough order of how badly it would
mislead someone if it did:

1. **Show silence.** A station that has gone quiet and a station whose trainee
   is calm both produce an empty reason list, and this page is the only place
   in the system where that difference can be seen. Stale, dropped, and
   vanished stations are rendered louder than healthy ones, not quieter.
2. **Not overclaim the rank.** The weights have never been fitted to a real
   incident (docs/VALIDATION.md), so the queue is a prompt to look, not a
   diagnosis, and the page says so on screen rather than in a docstring.
3. **Treat every string from a phone as text.** `trainee_id`, `station_id`,
   and `exercise` are all phone-chosen. They are written with `textContent`
   and never with `innerHTML`, so a phone cannot inject markup into the
   trainer's view.
"""

from __future__ import annotations

CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Argus — trainer console</title>
<style>
  :root {
    --bg: #0b0d10;
    --panel: #14181d;
    --panel-2: #1a1f26;
    --line: #262d36;
    --text: #e8eaed;
    --muted: #8b9096;
    --live: #35d07f;
    --stale: #f5a623;
    --down: #ff4d4f;
    --left: #4aa3ff;
    --right: #ffa24a;
    --bone: #98a2b3;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  header {
    display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
    padding: 0.9rem 1.25rem;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
    position: sticky; top: 0; z-index: 5;
  }
  header h1 { font-size: 1rem; font-weight: 700; letter-spacing: 0.12em; margin: 0; }
  header h1 span { color: var(--muted); font-weight: 500; letter-spacing: 0.02em; }
  #link { margin-left: auto; display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
  #counts { color: var(--muted); font-size: 0.85rem; font-variant-numeric: tabular-nums; }
  #useCaseSelect {
    background: var(--bg); color: var(--text); border: 1px solid var(--line);
    border-radius: 4px; font: inherit; font-size: 0.8rem; padding: 0.15rem 0.4rem;
  }
  #useCaseSelect:disabled { opacity: 0.6; }
  #useCaseError { color: var(--down); font-size: 0.78rem; }

  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--live); flex: none; }
  .dot.warn { background: var(--stale); }
  .dot.down { background: var(--down); }

  main { padding: 1.25rem; max-width: 1500px; margin: 0 auto; }
  section { margin-bottom: 1.75rem; }
  h2 {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
    color: var(--muted); text-transform: uppercase; margin: 0 0 0.6rem;
  }
  .note { color: var(--muted); font-size: 0.8rem; margin: 0 0 0.6rem; }

  #banner {
    display: none; margin-bottom: 1.25rem; padding: 0.7rem 1rem;
    border: 1px solid var(--down); border-left-width: 4px; border-radius: 4px;
    background: rgba(255, 77, 79, 0.09); color: #ffd9d9; font-size: 0.9rem;
  }
  #banner.shown { display: block; }

  /* -- join requests ------------------------------------------------------ */
  /* Above the help queue, and the only blue on the page: a phone waiting at
     the door is a trainee nobody is watching yet, which is urgent, but it is
     not the same kind of urgent as a fall and must not borrow that colour. */
  #joins { display: flex; flex-direction: column; gap: 0.4rem; }
  .join {
    display: grid; grid-template-columns: 1fr auto; align-items: center;
    gap: 0.9rem; padding: 0.7rem 0.95rem;
    background: rgba(74, 163, 255, 0.09);
    border: 1px solid var(--line); border-left: 4px solid var(--left);
    border-radius: 5px;
  }
  .jwho { font-size: 1.05rem; font-weight: 600; }
  .jwhere { color: var(--muted); font-size: 0.82rem; }
  .jbtns { display: flex; gap: 0.5rem; }
  .jbtns button {
    font: inherit; font-size: 0.85rem; font-weight: 600;
    padding: 0.4rem 0.9rem; border-radius: 4px; cursor: pointer;
    border: 1px solid var(--line); background: var(--panel-2); color: var(--text);
  }
  .jbtns button.yes { background: rgba(53, 208, 127, 0.16); border-color: var(--live); color: var(--live); }
  .jbtns button.no { background: rgba(255, 77, 79, 0.12); border-color: var(--down); color: var(--down); }
  .jbtns button:disabled { opacity: 0.45; cursor: default; }

  /* -- the help queue: the primary view ---------------------------------- */
  .queue { display: flex; flex-direction: column; gap: 0.4rem; }
  .qrow {
    display: grid; grid-template-columns: 4.2rem 1fr auto; align-items: center;
    gap: 0.9rem; padding: 0.7rem 0.95rem;
    background: var(--panel); border: 1px solid var(--line);
    border-left: 4px solid var(--bone); border-radius: 5px;
  }
  /* Three tiers, kept visually distinct: a 0.20 styled like a 0.80 teaches a
     trainer to discount the colour, and then the colour stops working when it
     matters. Red is reserved for at-or-above the configured alert threshold. */
  .qrow.attn { border-left-color: var(--down); background: rgba(255, 77, 79, 0.10); }
  .qrow.watch { border-left-color: var(--bone); }
  .qrow.silent { border-left-color: var(--stale); background: rgba(245, 166, 35, 0.08); }
  .qscore { font-size: 1.5rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .qrow.watch .qscore, .qrow.silent .qscore { color: var(--muted); }
  .qwho { font-size: 1.05rem; font-weight: 600; }
  .qrow.watch .qwho { font-weight: 550; }
  .qwhy { color: var(--muted); font-size: 0.87rem; }
  .qwhere { color: var(--muted); font-size: 0.8rem; text-align: right; }
  .qvol { color: var(--muted); font-size: 0.74rem; margin-top: 0.15rem;
          font-variant-numeric: tabular-nums; }
  .qtier {
    display: block; font-size: 0.68rem; letter-spacing: 0.09em;
    text-transform: uppercase; margin-bottom: 0.15rem;
  }
  .qrow.attn .qtier { color: var(--down); }
  .qrow.watch .qtier { color: var(--muted); }
  .qrow.silent .qtier { color: var(--stale); }
  .calm {
    padding: 0.9rem 1rem; border: 1px dashed var(--line); border-radius: 5px;
    color: var(--muted); font-size: 0.9rem;
  }

  /* -- station cards ------------------------------------------------------ */
  .grid {
    display: grid; gap: 0.9rem;
    grid-template-columns: repeat(auto-fill, minmax(248px, 1fr));
  }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-top: 3px solid var(--live); border-radius: 6px;
    padding: 0.75rem 0.8rem 0.65rem;
  }
  /* Ready, not live and not broken: a station whose trainee has not arrived
     is working perfectly, and drawing it green would claim someone is being
     watched while drawing it amber would send an instructor to a rack with
     nothing wrong. Its own, quieter colour. */
  .card.waiting { border-top-color: var(--bone); }
  .card.waiting .cchip { background: rgba(152, 162, 179, 0.16); color: var(--bone); }
  .card.stale { border-top-color: var(--stale); }
  .card.dropped, .card.gone, .card.nolink { border-top-color: var(--down); }
  /* Fade the pose, not the frame: the note explaining the silence lives in
     here too, and it is the one thing that must stay legible. */
  .card.stale canvas, .card.dropped canvas, .card.gone canvas,
  .card.nolink canvas { opacity: 0.45; }
  .card.gone { opacity: 0.72; }

  .chead { display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.15rem; }
  .cwho { font-size: 1rem; font-weight: 650; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cchip {
    margin-left: auto; flex: none; font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase;
    padding: 0.12rem 0.42rem; border-radius: 3px;
    background: rgba(53, 208, 127, 0.16); color: var(--live);
  }
  .card.stale .cchip { background: rgba(245, 166, 35, 0.16); color: var(--stale); }
  .card.dropped .cchip, .card.gone .cchip, .card.nolink .cchip {
    background: rgba(255, 77, 79, 0.16); color: var(--down);
  }
  .cwhere { color: var(--muted); font-size: 0.75rem; margin-bottom: 0.5rem;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .frame { position: relative; width: 100%; aspect-ratio: 1 / 1;
           background: var(--panel-2); border-radius: 4px; overflow: hidden; }
  .frame canvas { width: 100%; height: 100%; display: block; }
  /* Sits along the bottom rather than over the middle: the last known pose is
     the most useful thing on a card whose station has gone quiet, so the
     words explaining the silence must not cover it up. */
  .fnote {
    position: absolute; left: 0; right: 0; bottom: 0;
    padding: 0.85rem 0.5rem 0.4rem; text-align: center;
    font-size: 0.75rem; line-height: 1.3; color: var(--text);
    background: linear-gradient(to top, rgba(11, 13, 16, 0.96), rgba(11, 13, 16, 0));
  }
  .fnote:empty { display: none; }
  .card.stale .fnote { color: #ffe0b0; }
  .card.dropped .fnote, .card.gone .fnote { color: #ffd0d0; }

  .bar { height: 5px; background: var(--panel-2); border-radius: 3px; margin: 0.6rem 0 0.35rem; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--bone); width: 0; }
  .bar > i.hot { background: var(--down); }
  .cscore { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; }
  .cscore b { font-size: 0.95rem; font-variant-numeric: tabular-nums; }
  /* A score computed from frozen history is not a reading of the trainee
     now. It stays on screen -- hiding it would lose information -- but it
     stops looking like a live number. */
  .cscore b.frozen { color: var(--muted); }
  .cscore em { font-style: normal; color: var(--muted); font-size: 0.78rem; }
  .crolling { color: var(--muted); font-size: 0.72rem; margin-top: 0.1rem;
              font-variant-numeric: tabular-nums; }
  .crolling:empty { display: none; }
  .cwork { color: #cfd4da; font-size: 0.78rem; margin-top: 0.35rem;
           font-variant-numeric: tabular-nums; }
  .cwork:empty { display: none; }
  .creasons { font-size: 0.82rem; margin-top: 0.3rem; min-height: 1.2em; }
  .creasons.hot { color: #ffb4b4; }
  .creasons.quiet { color: var(--muted); }
  .cmeta { color: var(--muted); font-size: 0.76rem; margin-top: 0.35rem;
           font-variant-numeric: tabular-nums; }
  .cprofile { color: var(--muted); font-size: 0.76rem; margin-top: 0.25rem; }
  .cprofile:empty { display: none; }
  /* A profile that switches a check off is not a neutral fact about the
     scoring config -- it is a gap in what the floor is being watched for, and
     it reads as one. */
  .cprofile.blind {
    color: var(--stale); border-left: 2px solid var(--stale);
    padding-left: 0.45rem; margin-left: -0.05rem;
  }
  .cwarm { color: var(--stale); font-size: 0.74rem; margin-top: 0.2rem; }

  footer {
    border-top: 1px solid var(--line); padding: 0.9rem 1.25rem 2rem;
    color: var(--muted); font-size: 0.79rem; max-width: 1500px; margin: 0 auto;
  }
  footer b { color: #cfd4da; font-weight: 600; }
  .swatch { display: inline-block; width: 0.62rem; height: 0.62rem; border-radius: 2px;
            vertical-align: baseline; margin-right: 0.2rem; }
</style>
</head>
<body>
<header>
  <h1>ARGUS <span id="session">trainer console</span></h1>
  <select id="useCaseSelect" title="What this floor is running"></select>
  <span id="useCaseError"></span>
  <div id="counts"></div>
  <div id="link"><span class="dot" id="linkdot"></span><span id="linktext">connecting…</span></div>
</header>

<main>
  <div id="banner"></div>

  <section id="joinsection" style="display:none">
    <h2>Waiting to join</h2>
    <div id="joins"></div>
  </section>

  <section>
    <h2>Who needs attention</h2>
    <div class="queue" id="queue"></div>
  </section>

  <section>
    <h2>Stations</h2>
    <p class="note">
      Normalized phone-frame space, drawn square — not corrected for the
      camera's aspect ratio, which the protocol does not carry.
      <span class="swatch" style="background:var(--left)"></span>subject's left,
      <span class="swatch" style="background:var(--right)"></span>subject's right.
      Joints below the scorer's confidence threshold are not drawn.
    </p>
    <div class="grid" id="grid"></div>
  </section>
</main>

<footer>
  <b>The rank is deterministic, not validated.</b>
  The weights have never been fitted to a real incident and no accuracy has
  ever been measured on this task, so treat the order as a prompt to look at
  someone — not as a diagnosis of what is happening to them. See
  docs/VALIDATION.md.
</footer>

<script>
"use strict";

// The one cadence not read from config, because it cannot be: it is how long
// to wait before retrying when the server has never answered, so there is no
// config to have read. Every other interval on this page comes from
// payload.config, which comes from configs/argus.default.toml.
const RETRY_BEFORE_FIRST_CONTACT_MS = 500;

// COCO-17, in the protocol's order. Limbs are coloured by the SUBJECT's own
// left and right, which is the convention docs/PROTOCOL.md warns about: a
// phone that has them backwards is visible here at a glance instead of only
// as an inexplicable off_task score.
const LEFT = "left", RIGHT = "right", MID = "mid";
const EDGES = [
  [0, 1, LEFT], [0, 2, RIGHT], [1, 3, LEFT], [2, 4, RIGHT],
  [5, 6, MID],
  [5, 7, LEFT], [7, 9, LEFT],
  [6, 8, RIGHT], [8, 10, RIGHT],
  [5, 11, MID], [6, 12, MID], [11, 12, MID],
  [11, 13, LEFT], [13, 15, LEFT],
  [12, 14, RIGHT], [14, 16, RIGHT]
];

// Triage reason codes as something a trainer can read. `possible_fall` is a
// value, not a sentence. Anything not listed here falls back to a prettified
// form of the code itself, so an operator who adds a form-error code to the
// config gets readable output without editing this page.
const PROSE = {
  possible_fall: "Possible fall",
  prolonged_stillness: "Not moving",
  hands_face_occluded: "Face and hands out of view",
  off_task_orientation: "Turned away from the station",
  form_error: "Form flagged by the phone",
  persistent_form_fault: "Cannot hold form — wrong most of this set",
  insufficient_depth: "Not reaching depth",
  knee_valgus: "Knees caving inward",
  rounded_back: "Rounded back",
  heels_rising: "Heels lifting",
  excessive_forward_lean: "Leaning too far forward",
  incomplete_lockout: "Not locking out",
  asymmetric_form: "Uneven left and right",
  hips_sagging: "Hips sagging — lower back dipping",
  hips_piked: "Hips piked — hips too high",
  lean_back_error: "Leaning back — swinging the weight up",
  knee_over_toe: "Front knee past the toes"
};

// The five scoring features, and what switching one off means in words. Kept
// in the scorer's own order so the sentence reads the same way every time.
const FEATURES = ["fall", "stillness", "occlusion", "off_task", "form_error"];
const FEATURE_WATCH = {
  fall: "falls",
  stillness: "stillness",
  occlusion: "being hidden from view",
  off_task: "facing away",
  form_error: "form"
};

function prose(code) {
  if (PROSE[code]) return PROSE[code];
  const words = String(code).split("_").join(" ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function proseList(codes) {
  return (codes || []).map(prose).join(" · ");
}

function duration(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  if (s < 60) return s + "s";
  return Math.floor(s / 60) + "m " + String(s % 60).padStart(2, "0") + "s";
}

// How much work a trainee has done, in whatever unit their exercise has.
// A plank has no reps and its whole quality is how long it was held well, so
// "3 of 14 reps" and "18s of 2m10s" are the same sentence about different
// units -- which is why `fault_rate` can be one number for both.
function volume(session) {
  if (!session) return "";
  if (session.reps > 0) {
    return session.reps_flagged + " of " + session.reps + " reps flagged";
  }
  if (session.hold_s > 0) {
    return duration(session.hold_flagged_s) + " of " + duration(session.hold_s) + " flagged";
  }
  return "";
}

// Which weight vector a trainee is actually being scored on, resolved the
// same way argus.config.ScoringConfig.weights_for resolves it: an exercise
// with no profile falls back to the defaults rather than being special.
function profileFor(exercise) {
  const key = (exercise || "").toLowerCase();
  const profiles = (state.cfg && state.cfg.exercise_weights) || {};
  if (key && profiles[key]) return { name: key, weights: profiles[key] };
  return { name: "", weights: (state.cfg && state.cfg.default_weights) || {} };
}

// A feature weighted zero contributes nothing to the score AND emits no
// reason code -- which is right, and is also why it has to be said out loud.
// The plank profile zeroes `fall` and `stillness` because a correct plank is
// horizontal and motionless; the cost is that a trainee who collapses
// mid-plank raises neither, and a card reading "nothing flagged" would look
// identical to one where those checks had actually run.
function notWatchedFor(weights) {
  if (!Object.keys(weights).length) return [];
  return FEATURES.filter((f) => !(weights[f] > 0)).map((f) => FEATURE_WATCH[f]);
}

const state = {
  cfg: null,
  known: new Map(),   // trainee_id -> {station, vanishedAtWall}
  cards: new Map(),   // trainee_id -> element refs, reused between polls
  lastOkWall: 0,
  lastTs: null,
  lastTsWall: 0
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  // textContent, always: every string on this page that came from a phone
  // is written as text and never as markup. tests/test_console.py pins that
  // structurally, by asserting this page contains no markup-writing call at
  // all -- a rule that holds by construction beats one held by remembering.
  if (text !== undefined) node.textContent = text;
  return node;
}

// -- use case ----------------------------------------------------------------
//
// The dropdown that changes `[session] use_case` at runtime. Options come
// from `cfg.known_use_cases` -- exactly what `POST /session/use_case` will
// accept -- rather than a list hardcoded here that could drift from the
// server's actual scorer registry.

let useCasePending = false;

function syncUseCaseSelect(cfg) {
  const select = document.getElementById("useCaseSelect");
  const options = cfg.known_use_cases || [];
  const optionsKey = options.join(",");
  if (select.dataset.optionsKey !== optionsKey) {
    while (select.firstChild) select.removeChild(select.firstChild);
    for (const useCase of options) {
      const option = el("option", null, useCase);
      option.value = useCase;
      select.appendChild(option);
    }
    select.dataset.optionsKey = optionsKey;
    select.addEventListener("change", onUseCaseChange);
  }
  if (!useCasePending) select.value = cfg.use_case;
  select.disabled = useCasePending;
}

async function onUseCaseChange(event) {
  const select = event.target;
  const chosen = select.value;
  const previous = state.cfg.use_case;
  const errorNode = document.getElementById("useCaseError");
  useCasePending = true;
  select.disabled = true;
  errorNode.textContent = "";
  try {
    const res = await fetch("/session/use_case", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ use_case: chosen }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.ok) {
      select.value = previous;
      errorNode.textContent = body.error || ("could not change use case (" + res.status + ")");
    }
  } catch (err) {
    select.value = previous;
    errorNode.textContent = "could not reach the server";
  } finally {
    useCasePending = false;
    select.disabled = false;
  }
}

// -- drawing ---------------------------------------------------------------
//
// `RENDERERS` is where a future use case's own visualization plugs in --
// `station.use_case` selects it, so drawing a torch angle or a checklist step
// is a new entry here, not a branch inside `renderFitness`. Today "fitness"
// is the only entry; anything else falls back to it, which is safe because
// `renderFitness` already draws nothing for a station with no `keypoints_xy`.

function renderFitness(canvas, station, faded) {
  const box = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(box.width * dpr));
  const h = Math.max(1, Math.round(box.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

  const g = canvas.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!station || !station.keypoints_xy) return;

  const conf = station.keypoints_conf || [];
  const min = state.cfg ? state.cfg.keypoint_conf_threshold : 0;
  const pts = station.keypoints_xy;
  const X = (i) => pts[i][0] * w;
  const Y = (i) => pts[i][1] * h;
  const ok = (i) => (conf[i] === undefined ? false : conf[i] >= min);

  g.globalAlpha = faded ? 0.45 : 1;

  if (station.bbox_xyxy) {
    const b = station.bbox_xyxy;
    g.strokeStyle = "#3b444f";
    g.lineWidth = Math.max(1, dpr);
    g.setLineDash([4 * dpr, 4 * dpr]);
    g.strokeRect(b[0] * w, b[1] * h, (b[2] - b[0]) * w, (b[3] - b[1]) * h);
    g.setLineDash([]);
  }

  const hue = { left: "#4aa3ff", right: "#ffa24a", mid: "#98a2b3" };
  g.lineWidth = Math.max(2, 2.4 * dpr);
  g.lineCap = "round";
  for (const [a, b, side] of EDGES) {
    if (!ok(a) || !ok(b)) continue;       // the scorer would not have used it either
    g.strokeStyle = hue[side];
    g.beginPath();
    g.moveTo(X(a), Y(a));
    g.lineTo(X(b), Y(b));
    g.stroke();
  }

  g.fillStyle = "#e8eaed";
  for (let i = 0; i < pts.length; i++) {
    if (!ok(i)) continue;
    g.beginPath();
    g.arc(X(i), Y(i), Math.max(1.8, 2.1 * dpr), 0, Math.PI * 2);
    g.fill();
  }
  g.globalAlpha = 1;
}

// Welding has no classifier and nothing to draw yet (see
// `argus.triage.compute_triage_welding`) -- this clears the canvas rather
// than drawing a skeleton that would only ever be blank, so a welding
// station's card reads as "nothing to show" instead of "broken".
function renderPlaceholder(canvas) {
  const box = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(box.width * dpr));
  const h = Math.max(1, Math.round(box.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  canvas.getContext("2d").clearRect(0, 0, w, h);
}

const RENDERERS = { fitness: renderFitness, welding: renderPlaceholder };

function draw(canvas, station, faded) {
  const renderer = (station && RENDERERS[station.use_case]) || renderFitness;
  renderer(canvas, station, faded);
}

// -- one station card ------------------------------------------------------

function makeCard(traineeId) {
  const root = el("div", "card");
  const head = el("div", "chead");
  const who = el("div", "cwho", traineeId);   // replaced per render
  const chip = el("span", "cchip");
  head.appendChild(who);
  head.appendChild(chip);

  const where = el("div", "cwhere");
  const frame = el("div", "frame");
  const canvas = document.createElement("canvas");
  const fnote = el("div", "fnote");
  frame.appendChild(canvas);
  frame.appendChild(fnote);

  const bar = el("div", "bar");
  const fill = el("i");
  bar.appendChild(fill);

  const score = el("div", "cscore");
  const scoreVal = el("b");
  const scoreAge = el("em");
  score.appendChild(scoreVal);
  score.appendChild(scoreAge);

  const rolling = el("div", "crolling");
  const reasons = el("div", "creasons");
  const work = el("div", "cwork");
  const meta = el("div", "cmeta");
  const profile = el("div", "cprofile");
  const warm = el("div", "cwarm");

  root.append(head, where, frame, bar, score, rolling, reasons, work, meta, profile, warm);
  const refs = {
    root, who, chip, where, canvas, fnote, fill, scoreVal, scoreAge,
    rolling, reasons, work, meta, profile, warm,
  };
  state.cards.set(traineeId, refs);
  return refs;
}

function statusOf(entry, snapshotTs) {
  if (entry.vanishedAtWall) return "gone";
  const s = entry.station;
  if (!s.connected) return "dropped";
  const age = snapshotTs - s.last_seen_ts;
  if (age >= state.cfg.stale_after_s) return "stale";
  // Reporting, but with nobody in frame. A station set up before its trainee
  // arrives is ready, and drawing it like a silent phone taught an instructor
  // to walk over to a rack that was working perfectly.
  return s.subject_present ? "live" : "waiting";
}

const CHIP = { live: "live", waiting: "ready", stale: "silent",
               dropped: "dropped", gone: "left floor" };

function renderCard(traineeId, entry, record, snapshotTs) {
  const c = state.cards.get(traineeId) || makeCard(traineeId);
  const s = entry.station;
  const status = statusOf(entry, snapshotTs);
  const age = snapshotTs - s.last_seen_ts;

  c.root.className = "card " + status;
  c.chip.textContent = CHIP[status];
  // A trainee_id is a device identifier. An instructor has to find a person
  // in a room, so the name leads and the id stays available underneath.
  c.who.textContent = s.display_name || s.trainee_id;
  c.where.textContent = (s.display_name && s.trainee_id !== s.station_id)
    ? s.station_id + " · " + s.trainee_id
    : (s.display_name ? s.trainee_id : s.station_id);

  // The frame note is where silence gets said in words rather than implied
  // by a greyed-out skeleton, which is easy to read as "calm" at a glance.
  if (status === "gone") {
    c.fnote.textContent = "Track dropped — this trainee is no longer being scored.";
  } else if (status === "dropped") {
    const left = Math.max(0, state.cfg.track_ttl_s - age);
    c.fnote.textContent = "Phone disconnected — dropping in " + left.toFixed(0) + "s";
  } else if (status === "stale") {
    c.fnote.textContent = "No frames for " + age.toFixed(1) + "s";
  } else if (!s.subject_present) {
    c.fnote.textContent = "Ready — waiting for a trainee";
  } else if (!s.keypoints_xy) {
    c.fnote.textContent = "Connected — no frames yet";
  } else {
    c.fnote.textContent = "";
  }
  draw(c.canvas, s, status !== "live");

  // The headline number is the *rolling* score: the instant one is correct
  // and unreadable, moving every tick off a two-second window. The instant
  // one is still what decides `hot`, because a fall must colour the card on
  // the frame it happens rather than once a mean has caught up.
  const score = record ? record.score : null;
  const rolling = s.session ? s.session.rolling_score : null;
  const shown = rolling !== null ? rolling : score;
  const hot = score !== null && score >= state.cfg.alert_threshold && status === "live";
  c.fill.style.width = (shown === null ? 0 : Math.min(100, shown * 100)) + "%";
  c.fill.className = hot ? "hot" : "";
  c.scoreVal.textContent = shown === null ? "—" : shown.toFixed(2);
  const livelike = status === "live" || status === "waiting";
  c.scoreVal.className = livelike ? "" : "frozen";
  c.scoreAge.textContent = status === "gone"
    ? "no longer scored"
    : (livelike ? "last seen " : "frozen, last seen ") + age.toFixed(1) + "s ago";

  // Say plainly that the big number is a session average and what the
  // trainee is doing this instant, so nobody reads a settled 0.31 as a claim
  // about right now.
  if (s.session) {
    const bits = ["session avg"];
    if (score !== null) bits.push("now " + score.toFixed(2));
    if (s.session.peak_score > 0) bits.push("peak " + s.session.peak_score.toFixed(2));
    c.rolling.textContent = bits.join(" · ");
  } else {
    c.rolling.textContent = "";
  }

  // A silent station keeps being scored off its frozen history, so it can
  // still carry a reason code — but that code describes the last frames that
  // arrived, not the trainee now. Rendering it in the same red as a live
  // finding would state something about the present that nothing observed.
  const codes = record && record.reason_codes.length ? record.reason_codes : [];
  if (status === "live") {
    c.reasons.className = codes.length ? "creasons hot" : "creasons quiet";
    c.reasons.textContent = codes.length ? proseList(codes) : "Nothing flagged";
  } else if (status === "waiting") {
    c.reasons.className = "creasons quiet";
    c.reasons.textContent = "No trainee at this station yet";
  } else if (codes.length) {
    c.reasons.className = "creasons quiet";
    c.reasons.textContent = "Last reading before it went quiet: " + proseList(codes);
  } else {
    c.reasons.className = "creasons quiet";
    c.reasons.textContent = "Not reporting — nothing flagged, nothing checked";
  }

  // The phone's own informational fields. Attributed to the phone on purpose:
  // Argus does not count reps or judge form, it relays what the device said.
  const bits = [];
  bits.push(s.exercise ? s.exercise : "exercise not reported");
  if (s.rep_count !== null && s.rep_count !== undefined) bits.push(s.rep_count + " reps");
  if (s.form_ok === true) bits.push("phone says form OK");
  else if (s.form_ok === false) bits.push("phone says form off");
  else bits.push("no form verdict");
  if (s.form_reason_codes && s.form_reason_codes.length) {
    bits.push(proseList(s.form_reason_codes));
  }
  c.meta.textContent = bits.join(" · ");

  // Volume, and the fault rate it earns. `fault_rate` is null until enough
  // work has been seen, and that stays null on screen rather than becoming a
  // reassuring 0% — "no faults yet" and "not enough to say" are different
  // things to put in front of someone deciding who to walk over to.
  if (s.session) {
    const vol = volume(s.session);
    const rate = s.session.fault_rate;
    let line = vol;
    if (rate !== null && rate !== undefined) {
      line += " · " + Math.round(rate * 100) + "% fault rate";
    } else if (vol) {
      line += " · too little work to call a rate yet";
    }
    if (s.session.active_s > 0) {
      line = (line ? line + " · " : "") + duration(s.session.active_s) + " observed";
    }
    c.work.textContent = line;
  } else {
    c.work.textContent = "";
  }

  // Which weights are running, and what they are not looking at. Only shown
  // when a named profile applies or when something is switched off -- on the
  // ordinary default vector with all five features live there is nothing here
  // worth a trainer's attention.
  const active = profileFor(s.exercise);
  const blind = notWatchedFor(active.weights);
  if (blind.length) {
    c.profile.className = "cprofile blind";
    c.profile.textContent =
      (active.name ? "Scored as " + active.name : "Default scoring") +
      " · not watching for " + blind.join(", ");
  } else if (active.name) {
    c.profile.className = "cprofile";
    c.profile.textContent = "Scored as " + active.name + " · all checks active";
  } else {
    c.profile.textContent = "";
  }

  const need = state.cfg.history_len;
  if (status !== "gone" && status !== "waiting" && s.observations < need) {
    c.warm.textContent = "Warming up " + s.observations + "/" + need +
                         " frames — some checks cannot fire yet";
  } else {
    c.warm.textContent = "";
  }
  return c.root;
}

// -- the help queue --------------------------------------------------------

function renderQueue(payload, order) {
  const queue = document.getElementById("queue");
  const byId = new Map((payload.records || []).map((r) => [r.trainee_id, r]));
  const rows = [];

  // Two tiers, ordered on two different numbers, on purpose.
  //
  // Alerts are ranked by the *instant* score and come first: that is the
  // "someone is in trouble now" signal, and burying a fall behind a trainee
  // with a bad session average would defeat the whole system.
  //
  // Everything below is ranked by the *rolling* score, which is what makes
  // the list stable enough to read. Sorting the calm end by the instant score
  // meant it reshuffled every tick and an instructor could not keep their
  // place in it.
  const listed = [];
  for (const r of payload.records || []) {
    const over = r.score >= state.cfg.alert_threshold;
    const entry = state.known.get(r.trainee_id);
    const status = entry ? statusOf(entry, payload.ts) : "live";
    if (status !== "live") continue;   // listed under "not reporting" instead
    const session = entry && entry.station.session;
    const rollingScore = session ? session.rolling_score : r.score;
    // Below the threshold, earn a place by having either a reason code now or
    // a session average worth reading -- not by a single frame.
    if (!over && !r.reason_codes.length && rollingScore < 0.01) continue;

    // Rank on the worse of two readings: the session average, and how much of
    // the trainee's actual work was flagged.
    //
    // The average alone gets this wrong. Two trainees can both sit at 0.20
    // because the same non-form feature dominates both -- one motionless with
    // clean reps, one motionless with half their reps bad -- and the second
    // is who a coach should walk to. Form is only weighted 0.15 in the
    // default vector, so a fault rate of 50% moves the average by 0.045 and
    // loses to a stillness term worth 0.20.
    //
    // A known fault rate is therefore also read on the alert scale: half your
    // reps wrong ranks like half the alert threshold. It is a floor, never an
    // alert -- `over` above is still the instant score alone, so this can
    // reorder the calm end of the queue and can neither raise nor suppress an
    // alarm. `fault_rate` is null until enough work is seen, so this cannot
    // fire on one bad rep.
    const known = session && session.fault_rate !== null && session.fault_rate !== undefined;
    const rank = known
      ? Math.max(rollingScore, session.fault_rate * state.cfg.alert_threshold)
      : rollingScore;
    listed.push({ r, over, entry, session, rollingScore, rank });
  }
  // Ties on the rolling score are broken by how much of the trainee's work
  // was actually flagged. Two trainees can average the same because the same
  // non-form feature dominates both -- one motionless with clean reps, one
  // motionless with half their reps bad. The second needs a coach and the
  // first does not, and only the volume says so.
  const rate = (x) => (x.session && x.session.fault_rate) || 0;
  listed.sort((a, b) =>
    (b.over - a.over) ||
    (b.over ? b.r.score - a.r.score : b.rank - a.rank) ||
    (rate(b) - rate(a)) ||
    a.r.trainee_id.localeCompare(b.r.trainee_id));

  for (const item of listed) {
    const { r, over, entry, session, rollingScore, rank } = item;
    const row = el("div", "qrow " + (over ? "attn" : "watch"));
    // Show the number the row was ordered on, and say which reading it came
    // from. A row displaying 0.15 while sitting above one displaying 0.20
    // looks like a bug even when the ordering is right.
    row.appendChild(el("div", "qscore", (over ? r.score : rank).toFixed(2)));
    const mid = el("div");
    mid.appendChild(el("span", "qtier",
      over ? "above alert threshold · now"
           : (rank > rollingScore ? "ranked on fault rate" : "session average")));
    mid.appendChild(el("div", "qwho",
      (entry && entry.station.display_name) || r.trainee_id));
    mid.appendChild(el("div", "qwhy",
      proseList(r.reason_codes) || (session && volume(session)) || "No reason code fired"));
    row.appendChild(mid);
    const where = el("div", "qwhere");
    where.appendChild(el("div", "", entry ? entry.station.station_id : ""));
    if (session && volume(session) && r.reason_codes.length) {
      where.appendChild(el("div", "qvol", volume(session)));
    }
    row.appendChild(where);
    rows.push(row);
  }

  // Silence is ranked alongside danger, not below it: an unreachable station
  // and a calm one look identical in the score, and only this row tells them
  // apart.
  for (const traineeId of order) {
    const entry = state.known.get(traineeId);
    const status = statusOf(entry, payload.ts);
    if (status === "live" || status === "waiting") continue;
    const age = payload.ts - entry.station.last_seen_ts;
    const row = el("div", "qrow silent");
    row.appendChild(el("div", "qscore", "?"));
    const mid = el("div");
    mid.appendChild(el("span", "qtier", "not reporting"));
    mid.appendChild(el("div", "qwho",
      entry.station.display_name || traineeId));
    let why;
    if (status === "gone") why = "Track dropped — no longer being scored";
    else if (status === "dropped") why = "Phone disconnected " + age.toFixed(1) + "s ago";
    else why = "No frames for " + age.toFixed(1) + "s — not being scored, not calm";
    mid.appendChild(el("div", "qwhy", why));
    row.appendChild(mid);
    row.appendChild(el("div", "qwhere", entry.station.station_id));
    rows.push(row);
  }

  queue.textContent = "";
  if (!rows.length) {
    const n = order.length;
    const waiting = (payload.pending || []).length;
    let message;
    if (n) {
      message = "All " + n + " station" + (n === 1 ? " is" : "s are") +
                " reporting and nobody is above the alert threshold.";
    } else if (waiting) {
      // "Nobody is connected" would be true and useless here: somebody is
      // trying to, and the reason they are not is a button on this screen.
      message = waiting + " phone" + (waiting === 1 ? " is" : "s are") +
                " waiting for you to approve them above. Nobody is being scored yet.";
    } else {
      message = "No trainee is connected. Start a phone, or run `argus replay` " +
                "against a fixture to drive this console without one.";
    }
    queue.appendChild(el("div", "calm", message));
  } else {
    for (const row of rows) queue.appendChild(row);
  }
  return byId;
}

// -- join requests ---------------------------------------------------------

// Ids already acted on, so a decision that is still in flight does not get a
// second press when the next poll re-renders the same row.
const deciding = new Set();

async function decide(requestId, approve, row) {
  deciding.add(requestId);
  for (const b of row.querySelectorAll("button")) b.disabled = true;
  try {
    const res = await fetch("/join/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, approve: approve }),
    });
    if (!res.ok) {
      // 409 means the phone hung up or timed out between this row being drawn
      // and the button being pressed; 403 means this browser is not on the
      // machine running Argus. Both are worth saying out loud rather than
      // leaving a button that looks like it did nothing.
      const body = await res.json().catch(() => ({}));
      row.querySelector(".jwhere").textContent =
        body.error || ("decision refused (" + res.status + ")");
    }
  } catch (err) {
    row.querySelector(".jwhere").textContent = "could not reach the server";
  } finally {
    deciding.delete(requestId);
  }
}

function renderJoins(payload) {
  const section = document.getElementById("joinsection");
  const list = document.getElementById("joins");
  const pending = payload.pending || [];
  section.style.display = pending.length ? "" : "none";

  list.textContent = "";
  for (const req of pending) {
    const row = el("div", "join");
    const who = el("div");
    who.appendChild(el("div", "jwho", req.display_name || req.trainee_id));
    const left = Math.max(0, req.expires_ts - payload.ts);
    // Name the trainee id even when a display name was given: the id is what
    // an alert is dispatched against, so it is the thing the instructor is
    // actually admitting.
    who.appendChild(el("div", "jwhere",
      req.station_id + " · trainee " + req.trainee_id +
      " · expires in " + left.toFixed(0) + "s"));
    row.appendChild(who);

    const buttons = el("div", "jbtns");
    const yes = el("button", "yes", "Approve");
    const no = el("button", "no", "Decline");
    yes.onclick = () => decide(req.request_id, true, row);
    no.onclick = () => decide(req.request_id, false, row);
    if (deciding.has(req.request_id)) { yes.disabled = true; no.disabled = true; }
    buttons.append(yes, no);
    row.appendChild(buttons);
    list.appendChild(row);
  }
}

// -- the poll loop ---------------------------------------------------------

function renderLink(wall, ok) {
  const dot = document.getElementById("linkdot");
  const text = document.getElementById("linktext");
  const banner = document.getElementById("banner");
  const gap = state.lastOkWall ? (wall - state.lastOkWall) / 1000 : null;

  if (!state.lastOkWall) {
    dot.className = "dot warn";
    text.textContent = "connecting…";
    return;
  }
  if (!ok || gap > 3) {
    dot.className = "dot down";
    text.textContent = "server unreachable " + gap.toFixed(0) + "s";
    banner.textContent = "This console has not reached the Argus server for " +
      gap.toFixed(0) + "s. Everything below is the last snapshot that arrived and is " +
      "not live — the floor is unmonitored from here until this clears.";
    banner.className = "shown";
    // Renders stop when the fetch stops, so every card would otherwise keep
    // its last chip -- a station frozen mid-"live" reading as live is the
    // worst version of this failure, since the banner is the only thing
    // contradicting it and it is at the top of a page a trainer may have
    // scrolled past.
    for (const refs of state.cards.values()) {
      refs.root.className = "card nolink";
      refs.chip.textContent = "no link";
      refs.fnote.textContent = "Console lost the server — last frame received.";
      refs.scoreVal.className = "frozen";
    }
    return;
  }
  dot.className = "dot";
  text.textContent = "live";

  // The server can answer while its rank loop is dead. Fetches keep
  // succeeding, the snapshot timestamp stops advancing, and every age on the
  // page freezes at whatever it last was -- which reads as a calm floor.
  const stalled = (wall - state.lastTsWall) / 1000;
  if (state.lastTs !== null && stalled > 5) {
    banner.textContent = "The server is answering but its snapshot has not advanced for " +
      stalled.toFixed(0) + "s. The rank loop may have stopped; ages shown below are frozen.";
    banner.className = "shown";
  } else {
    banner.className = "";
  }
}

function render(payload, wall) {
  state.cfg = payload.config;
  if (payload.ts !== state.lastTs) { state.lastTs = payload.ts; state.lastTsWall = wall; }

  const seen = new Set();
  for (const s of payload.stations || []) {
    seen.add(s.trainee_id);
    state.known.set(s.trainee_id, { station: s, vanishedAtWall: null });
  }
  for (const [traineeId, entry] of state.known) {
    if (seen.has(traineeId)) continue;
    // A trainee whose session was evicted must not simply disappear from the
    // grid: vanishing without a word is the quiet degradation this console
    // exists to prevent. Hold the card briefly, marked, then let it go.
    if (!entry.vanishedAtWall) entry.vanishedAtWall = wall;
    if (wall - entry.vanishedAtWall > state.cfg.track_ttl_s * 1000) {
      state.known.delete(traineeId);
      const card = state.cards.get(traineeId);
      if (card) { card.root.remove(); state.cards.delete(traineeId); }
    }
  }

  renderJoins(payload);

  const order = Array.from(state.known.keys()).sort();
  const byId = renderQueue(payload, order);

  const grid = document.getElementById("grid");
  for (const traineeId of order) {
    const node = renderCard(traineeId, state.known.get(traineeId), byId.get(traineeId) || null, payload.ts);
    grid.appendChild(node);   // appendChild moves an existing node; no rebuild
  }

  document.getElementById("session").textContent =
    state.cfg.session_name || "trainer console";
  syncUseCaseSelect(state.cfg);

  const live = order.filter((t) => {
    const st = statusOf(state.known.get(t), payload.ts);
    return st === "live" || st === "waiting";
  }).length;
  const flagged = (payload.records || []).filter((r) => r.score >= state.cfg.alert_threshold).length;
  let counts = order.length + " station" + (order.length === 1 ? "" : "s") + " · " +
    live + " reporting · " + flagged + " above threshold";
  // "No phones have joined" means two different things depending on the mode:
  // nobody has started one, or somebody did and is waiting on this screen.
  if (state.cfg.approval === "manual") counts += " · approving joins manually";
  document.getElementById("counts").textContent = counts;
}

async function poll() {
  let payload = null;
  try {
    const res = await fetch("/console", { cache: "no-store" });
    if (res.ok) payload = await res.json();
  } catch (err) {
    payload = null;   // unreachable; renderLink says so rather than hiding it
  }
  const wall = Date.now();
  if (payload) { state.lastOkWall = wall; render(payload, wall); }
  renderLink(wall, Boolean(payload));

  const next = state.cfg ? state.cfg.poll_interval_ms : RETRY_BEFORE_FIRST_CONTACT_MS;
  setTimeout(poll, next);   // self-rescheduling: a slow reply delays the next
}                           // poll instead of stacking another on top of it

poll();
</script>
</body>
</html>
"""
