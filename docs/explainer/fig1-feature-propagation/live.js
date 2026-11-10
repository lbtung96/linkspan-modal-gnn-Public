// Fig 1 live code. Runs after the kit (iso.mjs) and bench (bench.mjs) sources.
(() => {
const M = deckModel(D);
const C = camera(30, 32);
fit(C, [...benchCorners(0), [0, 96.7, ZD + 66], [357, 96.7, ZD + 66], [-26, -88, -16], [386, -88, -16]], W, H, 12);   // same camera as Fig 2

const svg = document.querySelector(".iso-svg");
const layer = (n) => svg.querySelector(`[data-layer="${n}"]`);
const plate = document.querySelector(".plate");
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const N = M.nodes.length, SENS = D.sensors, isSensor = new Set(SENS);
const GAIN = 1 / 0.3, DATUM = 11, PMAX = 24;

// ---------------------------------------------------------------- propagation, exactly as features.feature_propagation
const deg = new Array(N).fill(0);
for (const [, t] of D.edges_directed) deg[t] += 1;
const dinv = deg.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
function propagate(col, kMax) {          // col: values at the sensors (in SENS order)
  let x = new Float64Array(N);
  SENS.forEach((s, q) => (x[s] = col[q]));
  const out = [Float64Array.from(x)];
  for (let k = 1; k <= kMax; k++) {
    const y = new Float64Array(N);
    for (const [s, t] of D.edges_directed) y[s] += dinv[s] * dinv[t] * x[t];   // sparse (row = src, col = dst) @ x
    SENS.forEach((s, q) => (y[s] = col[q]));
    x = y; out.push(Float64Array.from(x));
  }
  return out;
}
// self-check against the Python export (sample 0)
const chk = propagate(SENS.map((_, q) => D.sensor_series[q][0]), 50);
let worst = 0;
for (const k of D.check.iters) D.check.sample0[k].forEach((v, i) => (worst = Math.max(worst, Math.abs(v - chk[k][i]))));
window.__propCheck = worst;              // largest |JS - Python| over the recorded iterations (Python rounds to 1e-3)

// k = 50 at every sample of the window, for the time trace of a free node
const at50 = Array.from({ length: D.ns }, (_, t) => propagate(SENS.map((_, q) => D.sensor_series[q][t]), 50)[50]);

// ---------------------------------------------------------------- state
const T_START = D.sensor_series[0].map((_, t) => D.sensor_series.reduce((a, r) => a + Math.abs(r[t]), 0)).reduce((b, v, t, arr) => (v > arr[b] ? t : b), 0);
const S = { t: T_START, k: 0, kShow: 0, sel: 27, f: "B1", playing: false, touring: !still.matches, tourT: 0, idle: 0, loop: 0 };
let iters = propagate(SENS.map((_, q) => D.sensor_series[q][S.t]), 50);

// ---------------------------------------------------------------- DAQ box and its screens
const DAQ = { x: -16, y: -80, w: 126, d: 22, h: 24 };
const TR = { x0: -10, x1: 38, z0: 4.5, z1: 20 }, SP = { x0: 50, x1: 100, z0: 4.5, z1: 20 };
const CNT = { x0: 44, x1: 58, z0: 21, z1: 23 };
const onFace = (x, z) => pr(C, [x, DAQ.y, z]);
const win = (R) => `<path class="iso-line" data-tone="lo" d="${poly2d([onFace(R.x0 - 1.2, R.z0 - 1.2), onFace(R.x1 + 1.2, R.z0 - 1.2), onFace(R.x1 + 1.2, R.z1 + 1.2), onFace(R.x0 - 1.2, R.z1 + 1.2)])}"/><path class="screen-bg" d="${poly2d([onFace(R.x0, R.z0), onFace(R.x1, R.z0), onFace(R.x1, R.z1), onFace(R.x0, R.z1)])}"/>`;
const PORTS = SENS.map((_, q) => DAQ.x + 14 + q * 13);     // cable glands on the DAQ's back face
function daq() {
  let s = halo(C, rrect(DAQ.x, DAQ.y, DAQ.w, DAQ.d, 2), 0);
  s += prism(C, rrect(DAQ.x, DAQ.y, DAQ.w, DAQ.d, 2), 0, DAQ.h, { tone: "mid" });
  s += win(TR) + win(SP);
  const fmax = D.df * 399;
  s += Object.values(D.mode_bins).map((b) => { const x = SP.x0 + ((b * D.df) / fmax) * (SP.x1 - SP.x0); return line(C, [[x, DAQ.y, SP.z1], [x, DAQ.y, SP.z1 - 1.6]], "mid"); }).join("");
  s += line(C, [[TR.x0, DAQ.y, (TR.z0 + TR.z1) / 2], [TR.x1, DAQ.y, (TR.z0 + TR.z1) / 2]], "faint");
  s += dots(C, [[DAQ.x + 3, DAQ.y + 3, DAQ.h], [DAQ.x + DAQ.w - 3, DAQ.y + 3, DAQ.h], [DAQ.x + 3, DAQ.y + DAQ.d - 3, DAQ.h], [DAQ.x + DAQ.w - 3, DAQ.y + DAQ.d - 3, DAQ.h]], 0.5, "mid");
  // a status LED per channel on the top
  s += dots(C, PORTS.map((x) => [x, DAQ.y + 6, DAQ.h]), 0.7, "hi");
  return s;
}

// ---------------------------------------------------------------- cables (sensor -> down -> lane -> DAQ gland)
const order = SENS.map((s, q) => q).sort((a, b) => M.nodes[SENS[a]][0] - M.nodes[SENS[b]][0] || M.nodes[SENS[a]][1] - M.nodes[SENS[b]][1]);
function cableRoute(q) {
  const i = SENS[q], [x, y] = M.nodes[i];
  const rank = order.indexOf(q);
  const side = y > 50 ? 1 : -1;                       // leave the girder on its outer side
  const xd = x + (y > 50 ? 2.4 : y > 1 ? 4.4 : 0) - (x > M.L - 3 ? 6 : 0);
  const lane = -50 - (SENS.length - 1 - rank) * 1.6;  // larger x -> lane nearer the DAQ
  const xp = PORTS[rank];
  const z = ZD + 1.6;
  return [[x, y + side * 3.2, z], [xd, y + side * 6, z], [xd, y + side * 6, 0.7], [xd, lane, 0.7], [xp, lane, 0.7], [xp, DAQ.y + DAQ.d, 0.7]];
}
function cables() {
  return paint(C, SENS.map((_, q) => ({ at: [M.nodes[SENS[q]][0], -40, 0], bias: -50, svg: tube(C, cableRoute(q), 0.9, { ends: false }) +
    cyl(C, PORTS[order.indexOf(q)], DAQ.y + DAQ.d + 0.6, 1.3, 0, 2.2, { tone: "mid", n: 16 }) })));
}

// ---------------------------------------------------------------- deck: members, phase needles, pins, sensors
function phaseNeedles() {
  const U = D.edges_undirected.length, ph = D.phase[S.f], mg = D.mag[S.f];
  return M.members.map((m, k) => {
    if (m.group === "diagonal" || m.group === "bottom") return "";
    const p = memberPlan(M, m);
    const cx = (p.a[0] + p.b[0]) / 2, cy = (p.a[1] + p.b[1]) / 2;
    const [top] = [m.group === "bottom" ? ZD - GD : ZD];
    const ux = (p.b[0] - p.a[0]), uy = (p.b[1] - p.a[1]), L = Math.hypot(ux, uy);
    const a = ph[k], len = 0.6 + 3.4 * mg[k];
    const dx = (ux / L) * Math.cos(a) - (uy / L) * Math.sin(a), dy = (ux / L) * Math.sin(a) + (uy / L) * Math.cos(a);
    const anti = Math.abs(a) > Math.PI / 2;
    return cyl(C, cx, cy, 1.2, top, 0.5, { tone: "mid", n: 16 }) +
      line(C, [[cx, cy, top + 0.55], [cx + dx * len, cy + dy * len, top + 0.55]], anti ? "lit" : "hi", ` data-free="end" style="stroke-width:${anti ? 1.1 : 0.8}"`) +
      dots(C, [[cx, cy, top + 0.55]], 0.4, "hi");
    void U;
  }).join("");
}
function pinsSvg() {
  const x = iters[S.kShow], tru = D.true_series;
  const items = [];
  M.nodes.forEach(([nx, ny], i) => {
    const z0 = nodeTop(M, i) + 1.4 + (isSensor.has(i) ? 3.2 : 0);
    const hp = Math.max(0.3, Math.min(PMAX, DATUM + GAIN * x[i]));
    const sel = i === S.sel;
    let s = cyl(C, nx, ny, 0.7, z0, hp, { tone: sel ? "lit" : isSensor.has(i) ? "hi" : "mid", lit: sel, n: 12 });
    if (isSensor.has(i)) s += cyl(C, nx, ny, 1.25, z0, 1.6, { tone: "hi", n: 16 });                      // clamp collar
    else if (Math.abs(DATUM + GAIN * tru[i][S.t] - PMAX / 2) < PMAX / 2) { const zt = z0 + DATUM + GAIN * tru[i][S.t]; s += `<path class="iso-line" data-tone="${sel ? "mid" : "lo"}" d="${ringPath(C, nx, ny, 1.25, zt, 16)}"/>`; }
    items.push({ at: [nx, ny, z0], bias: 14, svg: s });
  });
  return items;
}
function sensorItems() {
  return SENS.map((i) => {
    const [x, y] = M.nodes[i], z = nodeTop(M, i);
    const side = y > 50 ? 1 : -1;
    let s = prism(C, rrect(x - 2.6, y - 3.2, 5.2, 6.4, 0.8), z, 3.2, { tone: "hi" });
    s += dots(C, [[x - 1.7, y - 2.3, z + 3.2], [x + 1.7, y + 2.3, z + 3.2]], 0.4, "mid");
    s += cyl(C, x, y + side * 3.2, 0.9, z + 1.0, 1.2, { tone: "mid", n: 12 });   // connector stub (cable leaves here)
    return { at: [x, y, z + 2], bias: 13, svg: s };
  });
}
function deckLayer() {
  const dz = new Array(N).fill(0);
  const deck = deckSvg(C, M, dz, { rails: (m) => "" });
  const pucks = puckItems(C, M, dz, (i) => ({ tone: i === S.sel ? "lit" : "mid", lit: i === S.sel })).filter((p) => !isSensor.has(p.i));
  pucks.forEach((p) => (p.bias = 12));
  return deck + phaseNeedles() + paint(C, pucks.concat(sensorItems(), pinsSvg()));
}

// ---------------------------------------------------------------- screens
function screens() {
  let s = "";
  const sel = S.sel, n = D.ns;
  const all = D.true_series[sel], prop = at50.map((x) => x[sel]);
  const peak = Math.max(1e-6, ...all.map(Math.abs), ...prop.map(Math.abs));
  const mid = (TR.z0 + TR.z1) / 2, amp = (TR.z1 - TR.z0) / 2 - 0.8;
  const tr = (arr) => "M" + arr.map((v, k) => pt(onFace(TR.x0 + (k / (n - 1)) * (TR.x1 - TR.x0), mid + (v / peak) * amp))).join("L");
  if (!isSensor.has(sel)) s += `<path class="trace2" d="${tr(all)}"/>`;
  s += `<path class="trace" d="${tr(isSensor.has(sel) ? all : prop)}"/>`;
  const cx = TR.x0 + (S.t / (n - 1)) * (TR.x1 - TR.x0);
  s += line(C, [[cx, DAQ.y, TR.z0], [cx, DAQ.y, TR.z1]], "lit");
  const psd = D.psd[sel];
  s += `<path class="trace" d="M${psd.map((v, k) => pt(onFace(SP.x0 + (k / 399) * (SP.x1 - SP.x0), SP.z0 + 0.6 + v * (SP.z1 - SP.z0 - 1.8)))).join("L")}"/>`;
  // the iteration counter, a little display on the top face of the DAQ
  s += faceText(C, [DAQ.x + DAQ.w - 30, DAQ.y + 12, DAQ.h], [1, 0, 0], [0, -1, 0], `k ${String(S.kShow).padStart(2, "0")}/50`, { size: 5, tone: "hi" });
  return s;
}

// ---------------------------------------------------------------- render
const MODEL_STATIC = baseSvg(C, { L: M.L });
function render() {
  layer("deck").innerHTML = deckLayer();
  layer("top").innerHTML = daqSvg + screens();
  setText();
}
let daqSvg = "";
const f2 = (v) => v.toFixed(2);
function setText() {
  const x = iters[S.kShow][S.sel], tt = (D.t0 + S.t / D.fs).toFixed(3);
  const who = isSensor.has(S.sel) ? `sensor node ${S.sel} ${f2(x)} mm/s² (clamped)` : `node ${S.sel} ${f2(x)} mm/s² (true ${f2(D.true_series[S.sel][S.t])})`;
  const t = `t ${tt} s · k ${S.kShow}/50 · ${who} · phase at ${S.f}`;
  if (readout.textContent !== t) readout.textContent = t;
  document.querySelectorAll("[data-f]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.f === S.f)));
  const r = document.querySelector("[data-time]"); if (+r.value !== S.t) r.value = S.t;
  plate.setAttribute("aria-valuetext", t);
}
function setTime(t) { S.t = Math.max(0, Math.min(D.ns - 1, t)); iters = propagate(SENS.map((_, q) => D.sensor_series[q][S.t]), 50); render(); }

// ---------------------------------------------------------------- loop
let frame = 0, last = 0, visible = false, acc = 0;
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 20) : 1 / 60;
  last = now;
  if (S.touring) {
    S.tourT += dt;
    if (S.k === 0 && S.tourT > 1.2) S.playing = true;
    if (S.k === 50 && S.tourT > 3.0) {
      S.loop++; S.tourT = 0; S.k = 0; S.kShow = 0;
      if (S.loop % 2 === 0) S.f = S.f === "B1" ? "T1" : "B1";
      S.sel = [27, 9, 40, 56, 18, 33][S.loop % 6];
      setTime((S.t + 17) % D.ns);
    }
  } else if ((S.idle += dt) > 10 && !still.matches) { S.touring = true; S.tourT = 0; }
  if (S.playing) {
    acc += dt;
    const step = S.k < 10 ? 0.22 : 0.06;              // slow for the first steps, where most of the change happens
    while (acc > step && S.k < 50) { acc -= step; S.k++; }
    if (S.k >= 50) { S.playing = false; acc = 0; if (S.touring) S.tourT = 0; }
  }
  if (S.kShow !== S.k) { S.kShow = S.k; render(); }
  frame = visible ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };

// ---------------------------------------------------------------- input
const hold = () => { S.touring = false; S.idle = 0; };
function nodeAt(evt) {
  const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  let best = -1, bd = 1e9;
  M.nodes.forEach(([x, y], i) => { const s = pr(C, [x, y, nodeTop(M, i) + 1]); const d = Math.hypot(s[0] - q.x, s[1] - q.y); if (d < bd) { bd = d; best = i; } });
  return bd < 14 ? best : -1;
}
svg.addEventListener("pointermove", (e) => { const i = nodeAt(e); if (i >= 0 && i !== S.sel) { hold(); S.sel = i; render(); } });
svg.addEventListener("click", (e) => { const i = nodeAt(e); if (i >= 0) { hold(); S.sel = i; render(); } });
const play = () => { hold(); if (S.k >= 50) { S.k = 0; S.kShow = -1; } S.playing = true; run(); };
plate.addEventListener("keydown", (e) => {
  let used = true;
  if (e.key === " ") play();
  else if (e.key === "ArrowRight") { hold(); setTime(S.t + 1); }
  else if (e.key === "ArrowLeft") { hold(); setTime(S.t - 1); }
  else if (e.key === "ArrowUp") { hold(); S.k = Math.min(50, S.k + 1); }
  else if (e.key === "ArrowDown") { hold(); S.k = Math.max(0, S.k - 1); }
  else if (e.key === "b" || e.key === "B") { hold(); S.f = "B1"; render(); }
  else if (e.key === "t" || e.key === "T") { hold(); S.f = "T1"; render(); }
  else used = false;
  if (used) e.preventDefault();
});
document.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.act === "play") play();
  if (b.dataset.act === "step") { hold(); S.playing = false; S.k = Math.min(50, S.k + 1); }
  if (b.dataset.act === "reset") { hold(); S.playing = false; S.k = 0; }
}));
document.querySelectorAll("[data-f]").forEach((b) => b.addEventListener("click", () => { hold(); S.f = b.dataset.f; render(); }));
document.querySelector("[data-time]").addEventListener("input", (e) => { hold(); setTime(+e.target.value); });
new IntersectionObserver((en) => { visible = en[en.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(plate);
if (still.matches) { S.k = 50; S.kShow = 50; }

window.__fig = { S, hold, setTime, setK: (k) => { hold(); S.playing = false; S.k = k; S.kShow = k; render(); }, setSel: (i) => { hold(); S.sel = i; render(); }, setF: (f) => { hold(); S.f = f; render(); } };

layer("base").innerHTML = MODEL_STATIC;
layer("under").innerHTML = paint(C, pedestalItems(C, M)) + cables();
daqSvg = daq();
render();
})();
