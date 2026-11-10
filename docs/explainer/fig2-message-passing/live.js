// Fig 2 live code. Runs in the page after the kit (iso.mjs) and bench (bench.mjs) sources.
(() => {
const M = deckModel(D);
const C = camera(30, 32);
const ZR = ZD + 52, ZN = ZD + 28;              // raised receiver block, neighbour blocks
fit(C, [...benchCorners(0), [0, 96.7, ZR + 14], [357, 96.7, ZR + 14], [-26, -88, -16], [386, -88, -16]], W, H, 12);

const svg = document.querySelector(".iso-svg");
const layer = (n) => svg.querySelector(`[data-layer="${n}"]`);
const plate = document.querySelector(".plate");
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const MODES = ["B1", "T1", "B2", "T2"];
const N = M.nodes.length, E = D.edges_directed.length;
const inEdges = M.nodes.map(() => []);
D.edges_directed.forEach(([s, t], c) => inEdges[t].push(c));
const memberOf = new Map();
M.members.forEach((m, k) => { memberOf.set(m.a + "," + m.b, k); memberOf.set(m.b + "," + m.a, k); });

// hop distance from a node (undirected graph; directed edges come in pairs)
function hops(i) {
  const d = new Array(N).fill(Infinity); d[i] = 0; const q = [i];
  while (q.length) { const u = q.shift(); for (const v of M.nbr[u]) if (d[v] === Infinity) { d[v] = d[u] + 1; q.push(v); } }
  return d;
}

// ---------------------------------------------------------------- state
const S = { stage: 0, mode: 0, head: 0, recv: 32, t: 0, stageT: 0, pinsFrom: null, pinsTo: null, pinsT: 1,
  touring: !still.matches, tourT: 0, idle: 0, dial: [0, 0, 0, 0], defl: 0 };
const TOUR = { recvs: [32, 36, 44, 13, 69, 51], dur: [3.8, 4.2, 4.2, 4.2, 7.0] };

// ---------------------------------------------------------------- consoles
const ENC = { x: -16, y: -80, w: 126, d: 22, h: 24 };
const SV = { x0: -10, x1: 42, z0: 4.5, z1: 20 }, SE = { x0: 48, x1: 86, z0: 4.5, z1: 20 };
const KNOB = [98, -69], KZ = ENC.h;
const FH = { x: 250, y: -82, w: 124, d: 24, h: 12 };
const DIALS = [266, 297, 328, 359].map((x) => [x, -70]), DR = 11, DZ = FH.h + 3;
const STAGE_ANG = [150, 105, 60, 15, -30].map((a) => (a * Math.PI) / 180);
const fAng = (f) => ((225 - (Math.min(12, Math.max(0, f)) / 12) * 270) * Math.PI) / 180;
const onFace = (x, z) => pr(C, [x, ENC.y, z]);

function screenFrame(R) {
  const corners = [onFace(R.x0, R.z0), onFace(R.x1, R.z0), onFace(R.x1, R.z1), onFace(R.x0, R.z1)];
  const bezel = [onFace(R.x0 - 1.2, R.z0 - 1.2), onFace(R.x1 + 1.2, R.z0 - 1.2), onFace(R.x1 + 1.2, R.z1 + 1.2), onFace(R.x0 - 1.2, R.z1 + 1.2)];
  // mode markers: ticks hanging from the window's top edge at the FE frequencies (bin x df)
  return `<path class="iso-line" data-tone="lo" d="${poly2d(bezel)}"/><path class="screen-bg" d="${poly2d(corners)}"/>`;
}
function consoles() {
  let s = "";
  s += halo(C, rrect(ENC.x, ENC.y, ENC.w, ENC.d, 2), 0);
  s += prism(C, rrect(ENC.x, ENC.y, ENC.w, ENC.d, 2), 0, ENC.h, { tone: "mid" });
  s += screenFrame(SV) + screenFrame(SE);
  // mode ticks along the S(v) window's bottom edge (FE frequencies of B1..T2)
  const fmax = D.df * 399;
  s += Object.values(D.mode_bins).map((b) => { const x = SV.x0 + ((b * D.df) / fmax) * (SV.x1 - SV.x0); return line(C, [[x, ENC.y, SV.z0], [x, ENC.y, SV.z0 + 1.6]], "mid"); }).join("");
  s += line(C, [[SE.x0, ENC.y, (SE.z0 + SE.z1) / 2], [SE.x1, ENC.y, (SE.z0 + SE.z1) / 2]], "faint");
  // screws and a vent row on the console top
  s += dots(C, [[ENC.x + 3, ENC.y + 3, ENC.h], [ENC.x + ENC.w - 3, ENC.y + 3, ENC.h], [ENC.x + 3, ENC.y + ENC.d - 3, ENC.h], [ENC.x + ENC.w - 3, ENC.y + ENC.d - 3, ENC.h]], 0.5, "mid");
  for (let i = 0; i < 9; i++) { const x = ENC.x + 10 + i * 6; s += `<path class="iso-line" data-tone="lo" d="${poly2d(rrect(x, ENC.y + 8, 3.2, 8, 1.2, 2).map((p) => pr(C, [p[0], p[1], ENC.h])))}"/>`; }
  // stage knob: detents and a knurled body
  s += dots(C, STAGE_ANG.map((a) => [KNOB[0] + 8.2 * Math.cos(a), KNOB[1] + 8.2 * Math.sin(a), KZ]), 0.55, "hi");
  s += cyl(C, KNOB[0], KNOB[1], 6.6, KZ, 1.2, { tone: "lo" });
  s += cyl(C, KNOB[0], KNOB[1], 5.4, KZ + 1.2, 4.2, { tone: "mid", inner: knurl(KNOB[0], KNOB[1], 5.4, KZ + 1.2, KZ + 5.4, 30) });
  // frequency head
  s += halo(C, rrect(FH.x, FH.y, FH.w, FH.d, 2), 0);
  s += prism(C, rrect(FH.x, FH.y, FH.w, FH.d, 2), 0, FH.h, { tone: "mid" });
  s += dots(C, [[FH.x + 3, FH.y + 3, FH.h], [FH.x + FH.w - 3, FH.y + 3, FH.h], [FH.x + 3, FH.y + FH.d - 3, FH.h], [FH.x + FH.w - 3, FH.y + FH.d - 3, FH.h]], 0.5, "mid");
  for (const [x, y] of DIALS) {
    let inner = `<path class="iso-line" data-tone="lo" d="${ringPath(C, x, y, DR - 2.2, DZ, 48)}"/>`;
    let minor = "", major = "";
    for (let k = 0; k <= 24; k++) {
      const a = fAng(k / 2), r0 = DR - 2.2, r1 = r0 - (k % 4 === 0 ? 2.6 : 1.3);
      const d = seg(C, [x + r0 * Math.cos(a), y + r0 * Math.sin(a), DZ], [x + r1 * Math.cos(a), y + r1 * Math.sin(a), DZ]);
      if (k % 4 === 0) major += d; else minor += d;
    }
    inner += `<path class="iso-line" data-tone="lo" d="${minor}"/><path class="iso-line" data-tone="mid" d="${major}"/>`;
    s += cyl(C, x, y, DR, FH.h, 3, { tone: "hi", crease: "lo", n: 48, inner: inner + knurl(x, y, DR, FH.h, FH.h + 1.6, 64) });
  }
  return s;
}
function knurl(cx, cy, r, z0, z1, n) {
  let d = "";
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU, nx = Math.cos(a), ny = Math.sin(a);
    const facing = -nx * C.sa + ny * C.ca;
    if (facing > -0.12) continue;
    d += seg(C, [cx + r * nx, cy + r * ny, z0], [cx + r * nx, cy + r * ny, z1]);
  }
  return `<path class="iso-line" data-tone="lo" d="${d}"/>`;
}

// ---------------------------------------------------------------- under the deck: pedestals, mean manifold
const HZ = 5;
function underSvg(dz) {
  const items = pedestalItems(C, M);
  for (const y of M.gy) {
    items.push({ at: [190, y, HZ], bias: -40, svg: tube(C, [[4.5, y, HZ], [366, y, HZ]], 2.2, { ends: false }) });
    for (const x of [60, 150, 260, 340]) items.push({ at: [x, y, 1], bias: -45, svg: prism(C, rrect(x - 1.6, y - 2, 3.2, 4, 0.6), 0, HZ - 1.1, { tone: "lo" }) });
  }
  items.push({ at: [366, 20, HZ], bias: -39, svg: tube(C, [[366, M.gy[2], HZ], [366, FH.y + FH.d, HZ]], 2.2, { ends: false }) });
  items.push({ at: [366, M.gy[2], HZ + 0.1], bias: -38, svg: `<circle class="iso-cap" cx="${pr(C, [366, M.gy[2], HZ])[0].toFixed(2)}" cy="${pr(C, [366, M.gy[2], HZ])[1].toFixed(2)}" r="1.5"/>` });
  // a tap from every node into the manifold
  M.nodes.forEach(([x, y], i) => {
    let pts;
    if (M.onGirder[i] && x > 1) pts = [[x, y, ZD - GD + dz[i]], [x, y, HZ + 1.1]];
    else if (i === 67 || i === 68) { const g = i === 67 ? M.gy[0] : M.gy[2], s = Math.sign(g - y); pts = [[x, y + s * 4.5, HZ], [x, g - s * 1.1, HZ]]; }
    else if (i === 69) pts = [[x, y, ZD - 4 + dz[i]], [x, y, HZ], [x, M.gy[0] - 1.1, HZ]];
    else return;
    items.push({ at: pts[pts.length - 1], bias: -20, svg: line(C, pts, "lo", ` data-tap="${i}"`) + dots(C, [pts[pts.length - 1]], 0.5, "mid") });
  });
  return paint(C, items);
}

// ---------------------------------------------------------------- the deck, stations, valves, posts
function deckLayer(dz) {
  const st = S.stage, i0 = S.recv, layerIdx = st >= 1 && st <= 3 ? st - 1 : -1;
  const reach = hops(i0);
  const R = st === 4 ? -1 : st === 0 ? 0 : st;
  const nb = new Set(M.nbr[i0]);
  const railTone = (s, t) => {
    if (st === 4) return "faint";
    if (t === i0) return st === 0 ? "hi" : "lit";
    if (layerIdx >= 0 && reach[t] < R && reach[s] <= R) return "lo";
    return "faint";
  };
  const rails = (m, zAt) => {
    if (m.group === "diagonal" || m.group === "bottom") return "";
    const p = memberPlan(M, m);
    const dx = p.b[0] - p.a[0], dy = p.b[1] - p.a[1], L = Math.hypot(dx, dy);
    const off = m.group === "girder" || m.group === "front" ? 0.9 : 0.55;
    const rx = (dy / L) * off, ry = (-dx / L) * off;               // right of a->b
    const L3 = (q) => q.map(([x, y]) => [x, y, zAt(x, y) + 0.05]);
    return line(C, L3([[p.a[0] + rx, p.a[1] + ry], [p.b[0] + rx, p.b[1] + ry]]), railTone(m.a, m.b)) +
      line(C, L3([[p.a[0] - rx, p.a[1] - ry], [p.b[0] - rx, p.b[1] - ry]]), railTone(m.b, m.a));
  };
  const deck = deckSvg(C, M, dz, { rails });
  const pucks = puckItems(C, M, dz, (i) => {
    if (st === 4) return { tone: "mid" };
    if (i === i0) return { tone: "lit", lit: true, dot: "lit" };
    if (reach[i] <= R) return { tone: "hi", dot: "hi" };
    return { tone: "lo" };
  });
  for (const p of pucks) p.bias = 12;
  const extra = [];
  if (st <= 3) {
    // the receiver's valves: one per in-channel, lever angle = alpha (open along the channel)
    for (const c of inEdges[i0]) {
      const j = D.edges_directed[c][0];
      const A = M.nodes[i0], B = M.nodes[j];
      const L = Math.hypot(B[0] - A[0], B[1] - A[1]);
      const ux = (A[0] - B[0]) / L, uy = (A[1] - B[1]) / L;             // channel direction, toward i
      const grp = M.members[memberOf.get(i0 + "," + j)].group;
      const off = grp === "cross" || grp === "diagonal" || grp === "edge" ? 0.55 : 0.9;
      const rx = uy * off, ry = -ux * off;                               // right of j->i
      const dv = Math.min(8, 0.36 * L);
      const vx = A[0] - ux * dv + rx, vy = A[1] - uy * dv + ry;
      const zt = nodeTop(M, i0) + dz[i0] - (grp === "diagonal" ? 0 : 0);
      const zv = grp === "diagonal" || grp === "bottom" ? nodeTop(M, i0) + dz[i0] : zt;
      const a = layerIdx >= 0 ? D.alpha[layerIdx][S.head][c] : 1 / inEdges[i0].length;
      const th = (1 - a) * (Math.PI / 2);
      const lx = ux * Math.cos(th) - uy * Math.sin(th), ly = ux * Math.sin(th) + uy * Math.cos(th);
      let v = cyl(C, vx, vy, 1.25, zv, 1.8, { tone: "hi", n: 18 });
      v += line(C, [[vx, vy, zv + 1.8], [vx + lx * 4.2, vy + ly * 4.2, zv + 1.8]], st === 0 ? "mid" : "lit", ' style="stroke-width:1.1"');
      v += dots(C, [[vx, vy, zv + 1.8]], 0.5, "hi");
      extra.push({ at: [vx, vy, zv + 2], bias: 13, svg: v });
    }
  }
  if (st <= 3) {
    // posts: receiver up to its collector, neighbours up to their blocks (layers only)
    const posts = [[i0, ZR - 9]];
    if (st >= 1) for (const j of nb) posts.push([j, zN(j)]);
    for (const [j, zTop] of posts) {
      const [x, y] = M.nodes[j], z0 = nodeTop(M, j) + 1.4 + dz[j];
      extra.push({ at: [x, y, z0], bias: 14, svg: cyl(C, x, y, 0.75, z0, zTop - z0, { tone: j === i0 ? "lit" : "mid", lit: j === i0, n: 14 }) });
    }
  }
  return deck + paint(C, pucks.concat(extra));
}

// ---------------------------------------------------------------- pin blocks
// neighbours of the receiver stand at two levels, alternating along x, so close nodes never collide
function zN(j) {
  const nb = [...new Set(M.nbr[S.recv])].sort((p, q) => M.nodes[p][0] - M.nodes[q][0] || M.nodes[p][1] - M.nodes[q][1]);
  return ZN + (nb.indexOf(j) % 2) * 9;
}
function pinBlock(cx, cy, z, vals, own) {
  let s = "";
  const plan = rrect(cx - 5.4, cy - 10.2, 10.8, 20.4, 1.1);
  s += prism(C, plan, z, 2.5, { tone: own ? "lit" : "mid", lit: own });
  s += `<path class="iso-line" data-tone="faint" d="${poly2d(rrect(cx - 4.5, cy - 9.3, 9, 18.6, 0.7).map((p) => pr(C, [p[0], p[1], z + 2.5])))}"/>`;
  const pins = [];
  for (let h = 0; h < 4; h++) for (let k = 0; k < 8; k++) {
    const px = cx - 3.3 + h * 2.2, py = cy - 7.7 + k * 2.2, v = vals ? vals[h * 8 + k] : 0;
    const hp = Math.max(0.35, Math.min(11, 4.5 + 1.5 * v));
    const tone = own ? (h === S.head ? "lit" : "hi") : (h === S.head ? "hi" : "mid");
    pins.push({ at: [px, py, z], svg: cyl(C, px, py, 0.66, z + 2.5, hp, { tone, lit: own && h === S.head, n: 12 }) });
  }
  return s + paint(C, pins);
}
function collector(cx, cy) {
  return cyl(C, cx, cy, 3.6, ZR - 9, 3.4, { tone: "hi", n: 28, inner: `<path class="iso-line" data-tone="faint" d="${ringPath(C, cx, cy, 2.2, ZR - 5.6, 24)}"/>` }) +
    cyl(C, cx, cy, 1.4, ZR - 5.6, 5.6, { tone: "hi", n: 16 });
}
function blocksSvg() {
  const st = S.stage, i0 = S.recv;
  if (st === 4) return "";
  const items = [];
  const hRecv = S.pinsFrom && S.pinsTo ? S.pinsTo.map((v, k) => S.pinsFrom[k] + (v - S.pinsFrom[k]) * ease(S.pinsT)) : D.h[Math.max(0, st)][i0];
  if (st >= 1) for (const j of new Set(M.nbr[i0])) {
    const [x, y] = M.nodes[j];
    items.push({ at: [x, y, zN(j)], svg: pinBlock(x, y, zN(j), D.h[st - 1][j], false) });
  }
  const [x, y] = M.nodes[i0];
  items.push({ at: [x, y, ZR + 40], svg: collector(x, y) + pinBlock(x, y, ZR, hRecv, true) });
  return paint(C, items);
}
const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

// the in-edge whose coherency the S(e) screen shows: the one with the most structure (largest spread of Re)
function shownEdge(i) {
  const spread = (c) => { const r = D.coh.re[c], m = r.reduce((a, b) => a + b, 0) / r.length; return r.reduce((a, b) => a + (b - m) ** 2, 0); };
  return inEdges[i].reduce((best, c) => (spread(c) > spread(best) ? c : best), inEdges[i][0]);
}
// ---------------------------------------------------------------- front: traces, needles, knob pointer
function frontSvg() {
  let s = "";
  const psd = D.psd[S.recv];
  const n = psd.length;
  const svPts = psd.map((v, k) => onFace(SV.x0 + (k / (n - 1)) * (SV.x1 - SV.x0), SV.z0 + 0.6 + v * (SV.z1 - SV.z0 - 1.6)));
  s += `<path class="trace" d="M${svPts.map(pt).join("L")}"/>`;
  const c = shownEdge(S.recv);
  const re = D.coh.re[c], im = D.coh.im[c], m = re.length, mid = (SE.z0 + SE.z1) / 2, amp = (SE.z1 - SE.z0) / 2 - 0.8;
  s += `<path class="trace2" d="M${im.map((v, k) => pt(onFace(SE.x0 + (k / (m - 1)) * (SE.x1 - SE.x0), mid + v * amp))).join("L")}"/>`;
  s += `<path class="trace" d="M${re.map((v, k) => pt(onFace(SE.x0 + (k / (m - 1)) * (SE.x1 - SE.x0), mid + v * amp))).join("L")}"/>`;
  // needles
  DIALS.forEach(([x, y], k) => {
    const a = fAng(S.dial[k]), r = DR - 3.4;
    const tone = S.stage === 4 && k === S.mode ? "lit" : "hi";
    s += line(C, [[x - 2.6 * Math.cos(a), y - 2.6 * Math.sin(a), DZ + 0.1], [x + r * Math.cos(a), y + r * Math.sin(a), DZ + 0.1]], tone, ' data-free="start" style="stroke-width:1.1"');
    s += `<path class="iso-line" data-tone="mid" d="${ringPath(C, x, y, 1.3, DZ + 0.1, 14)}"/>`;
  });
  // knob pointer
  const a = S.knob ?? STAGE_ANG[S.stage];
  s += line(C, [[KNOB[0] - 1.4 * Math.cos(a), KNOB[1] - 1.4 * Math.sin(a), KZ + 5.4], [KNOB[0] + 4.6 * Math.cos(a), KNOB[1] + 4.6 * Math.sin(a), KZ + 5.4]], "lit", ' style="stroke-width:1.1"');
  return s;
}

// ---------------------------------------------------------------- pulses
let pulseEls = [];
function pulsePaths() {
  const st = S.stage, i0 = S.recv;
  const paths = [];
  if (st >= 1 && st <= 3) {
    const [xi, yi] = M.nodes[i0];
    for (const c of inEdges[i0]) {
      const j = D.edges_directed[c][0];
      const [xj, yj] = M.nodes[j];
      const L = Math.hypot(xi - xj, yi - yj), ux = (xi - xj) / L, uy = (yi - yj) / L;
      const grp = M.members[memberOf.get(i0 + "," + j)].group;
      const off = grp === "cross" || grp === "diagonal" || grp === "edge" ? 0.55 : 0.9, rx = uy * off, ry = -ux * off;
      const zj = nodeTop(M, j), zi = nodeTop(M, i0);
      const pts = [[xj, yj, zN(j)], [xj, yj, zj + 1.6], [xj + rx + ux * 2.5, yj + ry + uy * 2.5, zj + 0.4], [xi + rx - ux * 2.5, yi + ry - uy * 2.5, zi + 0.4], [xi, yi, zi + 1.6], [xi, yi, ZR - 7]];
      paths.push({ pts, r: 0.9 + 2.6 * D.alpha[st - 1][S.head][c] });
    }
  } else if (st === 4) {
    for (const y of M.gy) for (let k = 0; k < 3; k++) paths.push({ pts: [[8, y, HZ], [366, y, HZ], [366, FH.y + FH.d + 3, HZ]], r: 1.1, phase: k / 3, loop: true });
  }
  return paths.map((p) => {
    const segL = []; let tot = 0;
    for (let k = 1; k < p.pts.length; k++) { const l = Math.hypot(...p.pts[k].map((v, q) => v - p.pts[k - 1][q])); segL.push(l); tot += l; }
    return { ...p, segL, tot };
  });
}
function at(path, u) {
  let d = u * path.tot;
  for (let k = 0; k < path.segL.length; k++) {
    if (d <= path.segL[k] || k === path.segL.length - 1) { const t = Math.min(1, d / path.segL[k]); const a = path.pts[k], b = path.pts[k + 1]; return pr(C, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]); }
    d -= path.segL[k];
  }
}
let paths = [];
function rebuildPulses() {
  paths = pulsePaths();
  layer("pulses").innerHTML = paths.map(() => `<circle class="pulse" r="0" cx="0" cy="0"/>`).join("");
  pulseEls = [...layer("pulses").children];
}
function drawPulses() {
  const st = S.stage;
  paths.forEach((p, k) => {
    const el = pulseEls[k];
    let u, vis;
    if (still.matches) { el.setAttribute("r", "0"); return; }
    if (st === 4) { u = (S.t * 0.22 + p.phase) % 1; vis = u < 0.97; }
    else { u = (S.stageT % 2.6) / 1.9; vis = u <= 1; u = Math.min(u, 1); }
    const q = at(p, u);
    el.setAttribute("cx", q[0].toFixed(2)); el.setAttribute("cy", q[1].toFixed(2));
    const fade = st === 4 ? 1 : Math.min(1, u * 8, (1 - u) * 8 + 0.15);
    el.setAttribute("r", vis ? (p.r * Math.max(0.2, fade)).toFixed(2) : "0");
  });
}

// ---------------------------------------------------------------- readout
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
function readoutText() {
  const st = S.stage, i0 = S.recv, k = inEdges[i0].length;
  if (st === 0) { const c = shownEdge(i0); return `encode · node ${i0} · S(v) 400 → H(v) 32 · S(e) ${D.edges_directed[c][0]}→${i0} 800 → H(e) 32`; }
  if (st === 4) { const m = S.mode; return `decode · ${MODES[m]} ${f2(D.pred.freq_hz[m])} Hz (FE ${f2(D.fe.freq_hz[m])}) · MAC ${f3(D.pred.mac[m])}`; }
  const a = inEdges[i0].map((c) => D.alpha[st - 1][S.head][c]).sort((p, q) => q - p);
  return `l = ${st} · node ${i0} · ${k} in-edges · α ${a.map(f2).join(" ")} (head ${S.head + 1})`;
}

// ---------------------------------------------------------------- render
let dz = new Array(N).fill(0);
function renderStatic() {
  layer("base").innerHTML = baseSvg(C, { L: M.L });
  layer("front").dataset.consoles = "1";
}
let consolesSvg = "";
function renderDeck() {
  layer("under").innerHTML = underSvg(dz);
  layer("deck").innerHTML = deckLayer(dz) + consolesSvg;
}
function renderTop() { layer("top").innerHTML = frontSvg(); }
function renderBlocks() { layer("blocks").innerHTML = blocksSvg(); }
function setText() {
  const t = readoutText();
  if (readout.textContent !== t) readout.textContent = t;
  document.querySelectorAll("[data-stage]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.stage === S.stage)));
  document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.mode === S.mode)));
  document.querySelectorAll("[data-head]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.head === S.head)));
}

function setStage(st) {
  st = Math.max(0, Math.min(4, st));
  const prev = S.stage;
  S.stage = st; S.stageT = 0;
  if (st >= 1 && st <= 3) { S.pinsFrom = D.h[st - 1][S.recv]; S.pinsTo = D.h[st][S.recv]; S.pinsT = still.matches ? 1 : -0.75; }
  else if (st === 0) { S.pinsFrom = new Array(32).fill(0); S.pinsTo = D.h[0][S.recv]; S.pinsT = still.matches ? 1 : -0.2; }
  else { S.pinsFrom = S.pinsTo = null; }
  if (st !== 4) { dz = new Array(N).fill(0); }
  void prev;
  rebuildPulses(); renderDeck(); renderBlocks(); renderTop(); setText();
}
function setRecv(i) {
  if (i === S.recv) return;
  S.recv = i;
  setStage(S.stage);
}

// ---------------------------------------------------------------- loop
let frame = 0, last = 0, visible = false;
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 20) : 1 / 60;
  last = now;
  S.t += dt; S.stageT += dt;
  if (S.touring) {
    S.tourT += dt;
    if (S.tourT > TOUR.dur[S.stage]) {
      S.tourT = 0;
      if (S.stage === 4) { const loop = (S.loop = (S.loop || 0) + 1); S.recv = TOUR.recvs[loop % TOUR.recvs.length]; S.mode = loop % 4; setStage(0); }
      else setStage(S.stage + 1);
    }
  } else if ((S.idle += dt) > 9 && !still.matches) { S.touring = true; S.tourT = 0; }
  // pins
  let blocksDirty = false;
  if (S.pinsTo && S.pinsT < 1) { S.pinsT = Math.min(1, S.pinsT + dt / 0.9); blocksDirty = S.pinsT > 0; }
  if (blocksDirty) renderBlocks();
  // dials and knob
  const target = S.stage === 4 ? D.pred.freq_hz : [0, 0, 0, 0];
  let topDirty = false;
  S.dial = S.dial.map((v, k) => { const n = still.matches ? target[k] : follow(v, target[k], dt, 0.45); if (Math.abs(n - v) > 1e-4) topDirty = true; return Math.abs(n - target[k]) < 1e-3 ? target[k] : n; });
  const ka = STAGE_ANG[S.stage]; const kn = S.knob === undefined ? ka : still.matches ? ka : follow(S.knob, ka, dt, 0.18);
  if (S.knob === undefined || Math.abs(kn - S.knob) > 1e-4) topDirty = true;
  S.knob = Math.abs(kn - ka) < 1e-4 ? ka : kn;
  if (topDirty) renderTop();
  // mode shape
  if (S.stage === 4) {
    S.defl = still.matches ? 1 : Math.min(1, S.defl + dt / 0.8);
    const amp = 13 * S.defl * (still.matches ? 1 : Math.sin(S.t * TAU * 0.45));
    const sh = D.pred.shapes[S.mode];
    dz = sh.map((v) => v * amp);
    renderDeck();
  } else S.defl = 0;
  drawPulses();
  frame = visible ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };

// ---------------------------------------------------------------- input
function nodeAt(evt) {
  const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  let best = -1, bd = 1e9;
  M.nodes.forEach(([x, y], i) => { const s = pr(C, [x, y, nodeTop(M, i) + 1]); const d = Math.hypot(s[0] - q.x, s[1] - q.y); if (d < bd) { bd = d; best = i; } });
  return bd < 14 ? best : -1;
}
const hold = () => { S.touring = false; S.idle = 0; };
svg.addEventListener("pointermove", (e) => { const i = nodeAt(e); if (i >= 0) { hold(); if (S.stage === 4) setStage(1); setRecv(i); } });
svg.addEventListener("click", (e) => { const i = nodeAt(e); if (i >= 0) { hold(); setRecv(i); } });
plate.addEventListener("keydown", (e) => {
  let used = true;
  if (e.key === "ArrowRight") setStage(S.stage + 1);
  else if (e.key === "ArrowLeft") setStage(S.stage - 1);
  else if (e.key === "Home") setStage(0);
  else if (e.key === "End") setStage(4);
  else if (["1", "2", "3", "4"].includes(e.key)) { S.mode = +e.key - 1; setStage(4); }
  else if (e.key === "h" || e.key === "H") { S.head = (S.head + 1) % 4; renderDeck(); renderBlocks(); rebuildPulses(); setText(); }
  else if (e.key === "]" || e.key === "[") setRecv((S.recv + (e.key === "]" ? 1 : N - 1)) % N);
  else used = false;
  if (used) { e.preventDefault(); hold(); }
});
document.querySelectorAll("[data-stage]").forEach((b) => b.addEventListener("click", () => { hold(); setStage(+b.dataset.stage); }));
document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { hold(); S.mode = +b.dataset.mode; setStage(4); }));
document.querySelectorAll("[data-head]").forEach((b) => b.addEventListener("click", () => { hold(); S.head = +b.dataset.head; renderDeck(); renderBlocks(); rebuildPulses(); setText(); }));
new IntersectionObserver((en) => { visible = en[en.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(plate);
still.addEventListener?.("change", () => { if (still.matches) S.touring = false; });

// test hook for the capture script
window.__fig = { S, setStage, setRecv, setMode: (m) => { S.mode = m; setStage(4); }, setHead: (h) => { S.head = h; setStage(S.stage); }, hold };

renderStatic();
consolesSvg = consoles();
setStage(0);
S.knob = STAGE_ANG[0];
renderTop();
})();
