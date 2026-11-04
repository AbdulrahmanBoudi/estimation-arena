"use strict";
const $ = (id) => document.getElementById(id);
const SUITS = ["♣", "♦", "♥", "♠"], RANKS = "23456789TJQKA", TRUMPS = [...SUITS, "NT"];
const POS = ["var(--king)", "var(--subking)", "var(--subkooz)", "var(--kooz)"];
const POS_NAMES = ["King", "Sub-King", "Sub-Kooz", "Kooz"];
const STILL = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (x) => `${Math.round(x * 100)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function card(c, attrs = "") {
  const red = c >= 13 && c < 39 ? " red" : "";
  const r = RANKS[c % 13] === "T" ? "10" : RANKS[c % 13];
  return `<span class="card${red}" data-c="${c}" ${attrs}>${r}<b>${SUITS[Math.floor(c / 13)]}</b></span>`;
}

const S = { run: null, data: null, at: -1, snap: null, cache: new Map(), round: 0, step: 0, playing: false, auto: false };

// ---- the reel: one column per saved version, stacked by final position
function drawReel() {
  const vs = S.data.versions;
  $("reel").innerHTML = vs.map((v, i) => `<div class="v${i === S.at ? " on" : ""}" data-i="${i}" title="Iteration ${v.iteration.toLocaleString()}: ${v.quick.map((x, k) => `${POS_NAMES[k]} ${pct(x)}`).join(", ")}">${v.quick.map((x, k) => `<i style="height:${x * 100}%;background:${POS[k]}"></i>`).join("")}</div>`).join("");
  const r = $("reel");
  r.setAttribute("aria-valuemin", 0); r.setAttribute("aria-valuemax", Math.max(vs.length - 1, 0)); r.setAttribute("aria-valuenow", S.at);
}
function pickFromX(x) {
  const r = $("reel").getBoundingClientRect();
  const n = S.data.versions.length;
  go(Math.max(0, Math.min(n - 1, Math.floor(((x - r.left) / r.width) * n))));
}
{
  const reel = $("reel");
  let down = false;
  reel.addEventListener("pointerdown", (e) => { down = true; stopPlay(); reel.setPointerCapture(e.pointerId); pickFromX(e.clientX); });
  reel.addEventListener("pointermove", (e) => { if (down) pickFromX(e.clientX); });
  reel.addEventListener("pointerup", () => { down = false; });
  reel.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { stopPlay(); go(Math.max(0, S.at - 1)); e.preventDefault(); }
    if (e.key === "ArrowRight") { stopPlay(); go(Math.min(S.data.versions.length - 1, S.at + 1)); e.preventDefault(); }
  });
}

// ---- what it thinks: the fixed probes, updated in place so the colours morph between versions
function buildGrid() {
  let h = `<div></div>`;
  for (let n = 4; n <= 13; n++) h += `<div class="hd">${n}</div>`;
  TRUMPS.forEach((t, k) => {
    h += `<div class="hd${k === 1 || k === 2 ? " red" : ""}">${t}</div>`;
    for (let n = 4; n <= 13; n++) h += `<div class="cell" data-a="${2 + (n - 4) * 5 + k}" title="Bid ${n}${t}"><i></i><span></span></div>`;
  });
  $("grid").innerHTML = h;
}
function drawProbes(p) {
  if (!$("grid").children.length) buildGrid();
  $("probe-hand").innerHTML = [...p.hand].sort((a, b) => b - a).map((c) => card(c)).join("");
  const top = Math.max(0.0001, ...Object.entries(p.opening).filter(([a]) => +a !== 53).map(([, x]) => x));
  $("grid").querySelectorAll(".cell").forEach((cell) => {
    const x = p.opening[cell.dataset.a] ?? 0;
    cell.classList.toggle("off", !(cell.dataset.a in p.opening));
    cell.querySelector("i").style.opacity = Math.pow(x / top, 0.6) * Math.min(1, 0.15 + x * 4);
    cell.querySelector("span").textContent = x >= 0.1 ? pct(x) : "";
  });
  const pass = p.opening["53"] ?? 0;
  $("dash-fill").style.width = pct(p.dash_call); $("dash-val").textContent = pct(p.dash_call);
  $("pass-fill").style.width = pct(pass); $("pass-val").textContent = pct(pass);
  const lead = Object.entries(p.lead).map(([c, x]) => [+c, x]).sort((a, b) => b[0] - a[0]);
  const ids = lead.map(([c]) => c).join();
  if ($("lead").dataset.ids !== ids) {
    $("lead").dataset.ids = ids;
    $("lead").innerHTML = lead.map(([c]) => `<div class="lead-col" data-c="${c}"><small></small><div class="lead-bar"></div>${card(c)}</div>`).join("");
  }
  lead.forEach(([c, x]) => {
    const col = $("lead").querySelector(`[data-c="${c}"].lead-col`);
    col.querySelector(".lead-bar").style.height = `${Math.round(x * 56)}px`;
    col.querySelector("small").textContent = x >= 0.05 ? pct(x) : "";
  });
}

// ---- a version
async function fetchSnap(i) {
  const it = S.data.versions[i].iteration;
  if (!S.cache.has(it)) S.cache.set(it, fetch(`/api/lab/${S.run}/snap/${it}`).then((r) => r.json()));
  return S.cache.get(it);
}
function go(i) {
  if (!S.data.versions.length) return;
  S.at = i;
  const v = S.data.versions[i];
  const row = nearestRow(v.iteration);
  $("iter").textContent = v.iteration.toLocaleString();
  $("iter-sub").textContent = `Iteration, after ${row ? millions(row.decisions) : "?"} decisions of self-play`;
  drawReel();
  drawProbes(v.probes);
  const ok = $("verified");
  ok.hidden = false;
  ok.className = `verified${v.verified ? "" : " bad"}`;
  ok.textContent = v.verified ? "Both engines agree on this game" : "The two engines disagreed on this game";
  drawMultiples();
  fetchSnap(i).then((snap) => {
    if (S.at !== i) return;
    S.snap = snap;
    const rounds = snap.game.rounds;
    S.round = Math.min(S.round, rounds.length - 1);
    S.step = Math.min(S.step, rounds[S.round].ev.length);
    if (!S.playing) drawRoundChips();
    drawTable(false);
    drawRace();
  });
}
const millions = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} billion` : `${Math.round(n / 1e6)} million`);
function nearestRow(it) {
  const rows = S.data.rows;
  let best = null;
  for (const r of rows) if (!best || Math.abs(r.iteration - it) < Math.abs(best.iteration - it)) best = r;
  return best;
}

let playTimer = null;
function stopPlay() {
  S.playing = false; clearTimeout(playTimer); $("play").textContent = "Play timelapse";
}
function startPlay() {
  S.playing = true; $("play").textContent = "Pause";
  if (S.at >= S.data.versions.length - 1) go(0);
  const tick = () => {
    if (!S.playing) return;
    if (S.at >= S.data.versions.length - 1) return stopPlay();
    go(S.at + 1);
    playTimer = setTimeout(tick, STILL ? 1200 : 450);
  };
  playTimer = setTimeout(tick, 450);
}
$("play").onclick = () => (S.playing ? stopPlay() : startPlay());

// ---- the table
function stateAt(round, step) {
  const hands = round.hands.map((h) => new Set(h));
  const st = { hands, trick: [], won: [0, 0, 0, 0], est: [null, null, null, null], calls: ["", "", "", ""], trump: round.fixed ?? null, caller: null, lastWin: null, turn: round.first, phase: round.fixed == null ? 0 : 2 };
  let clear = false;
  for (let i = 0; i < step; i++) {
    const e = round.ev[i];
    if (clear) { st.trick = []; clear = false; st.lastWin = null; }
    if (e.a >= 68) { hands[e.s].delete(e.a - 68); st.trick.push([e.s, e.a - 68]); } else st.calls[e.s] = e.l;
    if (e.ph !== undefined) st.phase = e.ph;
    if (e.est) st.est = e.est;
    if (e.tr !== undefined) st.trump = e.tr;
    if (e.ca !== undefined) st.caller = e.ca;
    if (e.w !== undefined) { st.won[e.w]++; st.lastWin = e.w; clear = true; }
    st.turn = e.w !== undefined ? e.w : (round.ev[i + 1] ? round.ev[i + 1].s : null);
  }
  if (step < round.ev.length) st.turn = round.ev[step].s;
  return st;
}
function drawTable(animate = true) {
  if (!S.snap) return;
  const before = new Map();
  if (animate && !STILL) $("felt").querySelectorAll(".card").forEach((el) => before.set(el.dataset.c, el.getBoundingClientRect()));
  const g = S.snap.game, round = g.rounds[S.round], st = stateAt(round, S.step);
  const seat = (k) => {
    const model = g.net_seats.includes(k);
    const role = st.caller === k ? ", Caller" : "";
    const est = st.est[k] == null ? "" : `, ${st.won[k]} of ${st.est[k]}`;
    const call = st.phase < 3 && st.calls[k] ? `, said ${esc(st.calls[k])}` : "";
    const cards = [...st.hands[k]].sort((a, b) => b - a).map((c) => card(c)).join("");
    return `<div class="seat s${k}${st.turn === k && S.step < round.ev.length ? " turn" : ""}"><h4>${model ? "Model" : "Heuristic"}<span class="sub">${role}${est}${call}</span></h4><div class="cards">${cards}</div></div>`;
  };
  const pit = st.trick.map(([s, c]) => card(c, `data-p`).replace('class="card', `class="card p${s}${st.lastWin === s ? " win" : ""}`)).join("");
  const trump = st.trump == null ? "no trump yet" : `trump ${TRUMPS[st.trump]}`;
  const done = S.step >= round.ev.length;
  const end = round.skipped ? "Round skipped, stakes double" : `Round ${round.no} scores ${round.result.map((x, k) => `${g.net_seats.includes(k) ? "M" : "H"} ${x > 0 ? "+" : ""}${x}`).join(", ")}`;
  const banner = !st.trick.length ? `<div class="banner">${done ? end : `Round ${round.no}, ${trump}${round.carry > 1 ? `, stakes ×${round.carry}` : ""}`}</div>` : "";
  $("felt").innerHTML = seat(2) + seat(1) + `<div class="pit">${banner}${pit}</div>` + seat(3) + seat(0);
  if (before.size) {
    $("felt").querySelectorAll(".card").forEach((el) => {
      const a = before.get(el.dataset.c);
      if (!a) return;
      const b = el.getBoundingClientRect();
      const dx = a.left - b.left, dy = a.top - b.top;
      if (Math.abs(dx) + Math.abs(dy) < 2) return;
      const base = getComputedStyle(el).transform;
      el.animate([{ transform: `translate(${dx}px,${dy}px) ${base === "none" ? "" : base}` }, { transform: base }], { duration: 260, easing: "cubic-bezier(.2,.8,.2,1)" });
    });
  }
  $("move").textContent = `Move ${S.step} of ${round.ev.length}`;
  drawThink(round);
}
function drawThink(round) {
  const e = round.ev[S.step];
  if (!e) { $("think").innerHTML = `<p class="none">End of the round.</p>`; return; }
  if (!S.snap.game.net_seats.includes(e.s)) { $("think").innerHTML = `<p class="none">Heuristic is playing ${esc(e.l)}.</p>`; return; }
  if (!e.p) { $("think").innerHTML = `<p class="none">Only one legal move: ${esc(e.l)}.</p>`; return; }
  $("think").innerHTML = e.p.map(([a, x]) => `<div class="opt${a === e.a ? " chosen" : ""}"><span>${esc(label(a))}</span><div class="t"><i style="width:${x * 100}%"></i></div><b>${pct(x)}</b></div>`).join("");
}
function label(a) {
  if (a === 0) return "No Dash"; if (a === 1) return "Dash Call";
  if (a <= 51) return `${Math.floor((a - 2) / 5) + 4}${TRUMPS[(a - 2) % 5]}`;
  if (a === 52) return "With"; if (a === 53) return "Pass";
  if (a < 68) return `Estimate ${a - 54}`;
  const c = a - 68; return `${RANKS[c % 13] === "T" ? "10" : RANKS[c % 13]}${SUITS[Math.floor(c / 13)]}`;
}
function drawRoundChips() {
  const rounds = S.snap.game.rounds;
  $("rounds").innerHTML = rounds.map((r, i) => `<button type="button" class="${i === S.round ? "on" : ""}" data-r="${i}" aria-label="Round ${r.no}">${r.no}</button>`).join("");
}
$("rounds").onclick = (e) => {
  const b = e.target.closest("button"); if (!b) return;
  S.round = +b.dataset.r; S.step = 0; stopAuto(); drawRoundChips(); drawTable(false); drawRace();
};
const move = (d) => { const n = S.snap.game.rounds[S.round].ev.length; S.step = Math.max(0, Math.min(n, S.step + d)); drawTable(d > 0); };
$("next").onclick = () => { stopAuto(); move(1); };
$("prev").onclick = () => { stopAuto(); move(-1); };
let autoTimer = null;
function stopAuto() { S.auto = false; clearTimeout(autoTimer); $("auto").textContent = "Play round"; }
$("auto").onclick = () => {
  if (S.auto) return stopAuto();
  S.auto = true; $("auto").textContent = "Pause";
  if (S.step >= S.snap.game.rounds[S.round].ev.length) S.step = 0;
  const t = () => {
    if (!S.auto) return;
    if (S.step >= S.snap.game.rounds[S.round].ev.length) return stopAuto();
    move(1); autoTimer = setTimeout(t, 650);
  };
  t();
};
function drawRace() {
  const g = S.snap.game, w = 420, h = 150, pad = 4;
  const pts = g.rounds.map((r) => r.scores.map((x, k) => x + r.result[k]));
  const all = pts.flat(); const lo = Math.min(0, ...all), hi = Math.max(1, ...all);
  const X = (i) => pad + (i / Math.max(pts.length - 1, 1)) * (w - pad * 2), Y = (v) => h - pad - ((v - lo) / (hi - lo)) * (h - pad * 2);
  const seats = [0, 1, 2, 3].map((k) => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(p[k]).toFixed(1)}`).join("")}" fill="none" stroke="${g.net_seats.includes(k) ? "#f2c14e" : "#7fb0ff"}" stroke-width="2" stroke-linejoin="round" ${k > 1 ? 'stroke-dasharray="5 4"' : ""}/>`).join("");
  $("race").setAttribute("viewBox", `0 0 ${w} ${h}`);
  $("race").innerHTML = `<line x1="0" x2="${w}" y1="${Y(0)}" y2="${Y(0)}" stroke="#3a4a9a"/><line x1="${X(S.round)}" x2="${X(S.round)}" y1="0" y2="${h}" stroke="#fbf6e9" opacity=".45"/>${seats}`;
}

// ---- behaviour, one small chart per measure
const MEASURES = [
  ["caller_made", "Caller makes its bid", "Higher means bids it can actually make", (x) => pct(x)],
  ["all_lose", "Everyone lost the round", "Should fall as estimates improve", (x) => pct(x)],
  ["skipped", "Rounds skipped", "Nobody called, so stakes double", (x) => pct(x)],
  ["dash", "Dash Calls", "Share of Dash decisions", (x) => pct(x)],
  ["with_", "With", "Share of auction moves", (x) => pct(x)],
  ["pass_", "Pass", "Share of auction moves", (x) => pct(x)],
  ["super_", "Super calls", "Bids at fixed-trump estimation", (x) => pct(x)],
  ["rounds_per_game", "Rounds per game", "18 with no replays, 38 at the cap", (x) => x.toFixed(1)],
  ["entropy", "Randomness left", "Near 0 would mean it stopped exploring", (x) => x.toFixed(2)],
  ["final_mse", "Error predicting final position", "0.52 means no idea; lower is better", (x) => x.toFixed(3)],
  ["pos_vs_start", "Reward against its first version", "0 is even, the best possible is +0.6", (x) => (x > 0 ? "+" : "") + x.toFixed(2)],
  ["per_s", "Decisions per second", "Drops if something else needs the GPU", (x) => Math.round(x).toLocaleString()],
];
function drawMultiples() {
  const rows = S.data.rows, it = S.data.versions[S.at]?.iteration ?? rows.at(-1)?.iteration;
  const maxIt = rows.at(-1)?.iteration || 1;
  $("multiples").innerHTML = MEASURES.map(([key, name, sub, fmt]) => {
    const pts = rows.filter((r) => r[key] !== undefined && !(key === "rounds_per_game" && !r[key]));
    if (pts.length < 2) return "";
    const vals = pts.map((r) => r[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
    const X = (i) => (i / maxIt) * 300, Y = (v) => 60 - ((v - lo) / span) * 56;
    const d = pts.map((r, i) => `${i ? "L" : "M"}${X(r.iteration).toFixed(1)} ${Y(r[key]).toFixed(1)}`).join("");
    const near = pts.reduce((b, r) => (Math.abs(r.iteration - it) < Math.abs(b.iteration - it) ? r : b), pts[0]);
    return `<div class="mult"><h3><span>${name}</span><b>${fmt(near[key])}</b></h3><small>${sub}</small><svg viewBox="0 0 300 64" preserveAspectRatio="none" aria-label="${name} over training"><path d="${d}" fill="none" stroke="#f8e3a8" stroke-width="1.6" vector-effect="non-scaling-stroke"/><line x1="${X(it)}" x2="${X(it)}" y1="0" y2="64" stroke="#fbf6e9" opacity=".5" vector-effect="non-scaling-stroke"/></svg></div>`;
  }).join("");
}

// ---- arena checks
function drawChecks() {
  const c = S.data.checks;
  if (!c.length) { $("checks").innerHTML = `<p class="empty">The first check runs 30 minutes into training.</p>`; return; }
  $("checks").innerHTML = c.map((x) => `<div class="check"><span>${x.minutes} min, iteration ${x.iteration.toLocaleString()}</span><div class="bar" role="img" aria-label="${x.shares.map((s, k) => `${POS_NAMES[k]} ${pct(s)}`).join(", ")}">${x.shares.map((s, k) => `<i style="width:${s * 100}%;background:${POS[k]}"></i>`).join("")}</div><span>King ${pct(x.king)}</span><span>${x.won}-${x.lost}</span></div>`).join("");
}

// ---- loading
async function loadRun(run, keepAt) {
  S.run = run;
  const res = await fetch(`/api/lab/${run}/summary`);
  S.data = await res.json();
  const latest = S.data.versions.length - 1;
  drawChecks();
  if (latest < 0) { $("iter-sub").textContent = "Waiting for the first saved version"; drawMultiples(); return; }
  if (S.playing) { drawReel(); return; }
  go(keepAt && S.at >= 0 && S.at < latest ? S.at : latest);
}
(async () => {
  const runs = await (await fetch("/api/lab/runs")).json();
  $("run").innerHTML = runs.map((r) => `<option>${esc(r)}</option>`).join("");
  if (!runs.length) { $("iter-sub").textContent = "No training runs yet"; return; }
  $("run").value = runs.at(-1);
  $("run").onchange = () => { stopPlay(); S.cache.clear(); S.at = -1; S.snap = null; loadRun($("run").value); };
  await loadRun($("run").value);
  setInterval(() => loadRun(S.run, true), 30000);
})();
