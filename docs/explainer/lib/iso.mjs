// A small isometric SVG kit for the explainer figures (no dependencies).
// Written for this repo, following the conventions of the "anatomy" skill
// (github.com/wheresryan22/anatomy, MIT): solids filled in paper colour and
// painted back to front, four side tones by facing plus a lighter top, thin
// lines, one lit accent, a framed card with title, hint and live readout.
// The same file is imported by the Node build scripts and inlined into the
// pages (build.mjs strips the `export` keywords), so the live parts are drawn
// by exactly the code that draws the static ones.

export const TAU = Math.PI * 2;

export function camera(az = 28, el = 30) {
  const a = (az * Math.PI) / 180, e = (el * Math.PI) / 180;
  return { ca: Math.cos(a), sa: Math.sin(a), ce: Math.cos(e), se: Math.sin(e), k: 1, ox: 0, oy: 0 };
}

// screen point of a world point [x, y, z] (z up, y away from the viewer)
export function pr(C, p) {
  const f = -p[0] * C.sa + p[1] * C.ca;
  return [C.ox + C.k * (p[0] * C.ca + p[1] * C.sa), C.oy + C.k * (-p[2] * C.ce - f * C.se)];
}

// nearness to the viewer: paint in ascending order
export function near(C, p) {
  return -(-p[0] * C.sa + p[1] * C.ca) * C.ce + p[2] * C.se;
}

export function fit(C, pts, W, H, pad = 24) {
  C.k = 1; C.ox = 0; C.oy = 0;
  const s = pts.map((p) => pr(C, p));
  const x0 = Math.min(...s.map((q) => q[0])), x1 = Math.max(...s.map((q) => q[0]));
  const y0 = Math.min(...s.map((q) => q[1])), y1 = Math.max(...s.map((q) => q[1]));
  C.k = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
  C.ox = (W - C.k * (x1 + x0)) / 2;
  C.oy = (H - C.k * (y1 + y0)) / 2;
  return C;
}

const f2 = (v) => (Math.round(v * 100) / 100).toString();
export const pt = (q) => f2(q[0]) + " " + f2(q[1]);
export const poly2d = (qs, close = true) => (qs.length ? "M" + qs.map(pt).join("L") + (close ? "Z" : "") : "");

export function hull(qs) {
  const p = [...qs].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

// plan shapes, counter-clockwise in (x, y)
export function rrect(x, y, w, d, r = 0, steps = 4) {
  r = Math.max(0, Math.min(r, w / 2, d / 2));
  if (r === 0) return [[x, y], [x + w, y], [x + w, y + d], [x, y + d]];
  const out = [];
  const corner = (cx, cy, a0) => { for (let i = 0; i <= steps; i++) { const a = a0 + (i / steps) * (Math.PI / 2); out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } };
  corner(x + w - r, y + r, -Math.PI / 2); corner(x + w - r, y + d - r, 0); corner(x + r, y + d - r, Math.PI / 2); corner(x + r, y + r, Math.PI);
  return out;
}
export function circ(cx, cy, r, n = 32) {
  return Array.from({ length: n }, (_, i) => [cx + r * Math.cos((i / n) * TAU), cy + r * Math.sin((i / n) * TAU)]);
}
// a bar of width w from plan point a to b (both ends square)
export function bar(a, b, w) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
  const nx = (-dy / L) * (w / 2), ny = (dx / L) * (w / 2);
  const q = [[a[0] - nx, a[1] - ny], [b[0] - nx, b[1] - ny], [b[0] + nx, b[1] + ny], [a[0] + nx, a[1] + ny]];
  return area(q) < 0 ? q.reverse() : q;
}
const area = (q) => q.reduce((s, p, i) => { const n = q[(i + 1) % q.length]; return s + p[0] * n[1] - n[0] * p[1]; }, 0) / 2;
export function inset(x, y, w, d, by) { return [x + by, y + by, w - 2 * by, d - 2 * by]; }

const LIGHT = [-0.62, -0.78];          // horizontal light direction (from front left)
export function shadeOf(nx, ny) {
  const v = (nx * LIGHT[0] + ny * LIGHT[1] + 1) / 2;
  return Math.max(0, Math.min(3, Math.floor(v * 3.999)));
}

// convex prism on a plan polygon, from z0 up by h. Optional zTop(x, y) tilts the top.
export function prism(C, plan, z0, h, o = {}) {
  const q = area(plan) < 0 ? [...plan].reverse() : plan;
  const zt = o.zTop || (() => z0 + h);
  const zb = o.zBot || (() => z0);
  const bot = q.map((p) => pr(C, [p[0], p[1], zb(p[0], p[1])]));
  const top = q.map((p) => pr(C, [p[0], p[1], zt(p[0], p[1])]));
  let faces = "", crease = [];
  for (let i = 0; i < q.length; i++) {
    const j = (i + 1) % q.length;
    const dx = q[j][0] - q[i][0], dy = q[j][1] - q[i][1], L = Math.hypot(dx, dy) || 1;
    const nx = dy / L, ny = -dx / L;
    if (-nx * C.sa + ny * C.ca >= -1e-6) continue;          // faces away
    faces += `<path class="iso-shade" data-shade="${shadeOf(nx, ny)}" d="${poly2d([bot[i], bot[j], top[j], top[i]])}"/>`;
    crease.push([top[i], top[j]]);
  }
  const outline = hull(bot.concat(top));
  const tone = o.tone || "mid";
  const cr = crease.map(([a, b]) => "M" + pt(a) + "L" + pt(b)).join("");
  const attrs = (o.lit ? ' data-lit=""' : "") + (o.attrs || "");
  return `<g class="iso-solid"${attrs}><path class="iso-fill" d="${poly2d(outline)}"/>${faces}<path class="iso-top" d="${poly2d(top)}"/>` +
    (o.inner || "") +
    `<path class="iso-line" data-tone="${o.crease || "faint"}" d="${cr}"/><path class="iso-line iso-edge" data-tone="${tone}" d="${poly2d(outline)}"/></g>`;
}

// a straight beam through stations [{x, y, zt, zb}] (top and bottom may vary: a deflected member)
export function beam(C, st, w, o = {}) {
  const a = st[0], b = st[st.length - 1], L = Math.hypot(b.x - a.x, b.y - a.y) || 1;
  const ux = (b.x - a.x) / L, uy = (b.y - a.y) / L, nx = -uy, ny = ux, hw = w / 2;
  const vis = (px, py) => -px * C.sa + py * C.ca < -1e-6;
  const sp = vis(nx, ny) ? 1 : -1;                                  // the visible long side
  const P = (s, side, top) => pr(C, [s.x + side * nx * hw, s.y + side * ny * hw, top ? s.zt : s.zb]);
  const nT = st.map((s) => P(s, sp, 1)), nB = st.map((s) => P(s, sp, 0)), fT = st.map((s) => P(s, -sp, 1));
  const eVis = vis(ux, uy), bVis = vis(-ux, -uy);
  let faces = `<path class="iso-shade" data-shade="${shadeOf(sp * nx, sp * ny)}" d="${poly2d(nT.concat([...nB].reverse()))}"/>`;
  if (eVis) faces += `<path class="iso-shade" data-shade="${shadeOf(ux, uy)}" d="${poly2d([P(b, 1, 1), P(b, -1, 1), P(b, -1, 0), P(b, 1, 0)])}"/>`;
  if (bVis) faces += `<path class="iso-shade" data-shade="${shadeOf(-ux, -uy)}" d="${poly2d([P(a, 1, 1), P(a, -1, 1), P(a, -1, 0), P(a, 1, 0)])}"/>`;
  const top = poly2d(nT.concat([...fT].reverse()));
  // silhouette: near bottom, the end, far top back, the start
  const out = [...nB];
  if (eVis) out.push(P(b, -sp, 0), P(b, -sp, 1)); else out.push(P(b, sp, 1));
  out.push(...[...fT].reverse().slice(eVis ? 0 : 0));
  if (bVis) out.push(P(a, -sp, 0)); else out.push(P(a, sp, 1));
  let cr = "M" + nT.map(pt).join("L");
  if (eVis) cr += "M" + pt(P(b, sp, 0)) + "L" + pt(P(b, sp, 1)) + "L" + pt(P(b, -sp, 1));
  if (bVis) cr += "M" + pt(P(a, sp, 0)) + "L" + pt(P(a, sp, 1)) + "L" + pt(P(a, -sp, 1));
  const tone = o.tone || "mid";
  return `<g class="iso-solid"${o.lit ? ' data-lit=""' : ""}>${faces}<path class="iso-top" d="${top}"/>${o.inner || ""}` +
    `<path class="iso-line" data-tone="${o.crease || "faint"}" d="${cr}"/><path class="iso-line iso-edge" data-tone="${tone}" d="${poly2d(out)}"/></g>`;
}

// vertical cylinder, smooth side (one gradient), flat top
export function cyl(C, cx, cy, r, z0, h, o = {}) {
  const n = o.n || 40;
  const ring = circ(cx, cy, r, n);
  const bot = ring.map((p) => pr(C, [p[0], p[1], z0]));
  const top = ring.map((p) => pr(C, [p[0], p[1], z0 + h]));
  const outline = hull(bot.concat(top));
  const tone = o.tone || "mid";
  const attrs = (o.lit ? ' data-lit=""' : "") + (o.attrs || "");
  // front arc of the top rim, ordered along the rim
  const idx = ring.map((_, i) => i).filter((i) => { const a = (i / n) * TAU; return -Math.cos(a) * C.sa + Math.sin(a) * C.ca < 0; });
  const start = Math.max(0, idx.findIndex((i) => !idx.includes((i - 1 + n) % n)));
  const ordered = idx.length ? idx.slice(start).concat(idx.slice(0, start)) : [];
  const arc = ordered.length ? "M" + ordered.map((i) => pt(top[i])).join("L") : "";
  return `<g class="iso-solid iso-cyl"${attrs}><path class="iso-fill" d="${poly2d(outline)}"/><path class="iso-side" d="${poly2d(outline)}"/>` +
    `<path class="iso-top" d="${poly2d(top)}"/>${o.inner || ""}<path class="iso-line" data-tone="${o.crease || "faint"}" d="${arc}"/><path class="iso-line iso-edge" data-tone="${tone}" d="${poly2d(outline)}"/></g>`;
}

export function line(C, pts, tone = "lo", extra = "") {
  return `<path class="iso-line" data-tone="${tone}"${extra} d="M${pts.map((p) => pt(pr(C, p))).join("L")}"/>`;
}
export function seg(C, a, b) { return "M" + pt(pr(C, a)) + "L" + pt(pr(C, b)); }
export function ringPath(C, cx, cy, r, z, n = 24) {
  return poly2d(circ(cx, cy, r, n).map((p) => pr(C, [p[0], p[1], z])));
}
export function dots(C, pts, r = 0.5, tone = "mid") {
  return `<g class="iso-dots" data-tone="${tone}">${pts.map((p) => { const q = pr(C, p); return `<circle cx="${f2(q[0])}" cy="${f2(q[1])}" r="${r}"/>`; }).join("")}</g>`;
}
// a tube along a 3-D polyline: body stroke, shine, end discs
export function tube(C, pts, w = 2.4, o = {}) {
  const d = "M" + pts.map((p) => pt(pr(C, p))).join("L");
  const a = pr(C, pts[0]), b = pr(C, pts[pts.length - 1]);
  const ends = o.ends === false ? "" : `<circle class="iso-cap" cx="${f2(a[0])}" cy="${f2(a[1])}" r="${f2(w * 0.62)}"/><circle class="iso-cap" cx="${f2(b[0])}" cy="${f2(b[1])}" r="${f2(w * 0.62)}"/>`;
  return `<g class="iso-tube"${o.attrs || ""}><path class="tb-edge" d="${d}" style="stroke-width:${f2(w + 1.2)}"/><path class="tb-body" d="${d}" style="stroke-width:${f2(w)}"/>` +
    `<path class="tb-shine" d="${d}" style="stroke-width:${f2(Math.max(0.5, w * 0.28))};transform:translate(${f2(-w * 0.18)}px,${f2(-w * 0.22)}px)"/>${ends}</g>`;
}
// text engraved on a horizontal face: matrix mapping text x to world dir u, text y to world dir v
export function faceText(C, origin, u, v, text, o = {}) {
  const O = pr(C, origin), U = pr(C, [origin[0] + u[0], origin[1] + u[1], origin[2] + u[2]]), V = pr(C, [origin[0] + v[0], origin[1] + v[1], origin[2] + v[2]]);
  const m = [U[0] - O[0], U[1] - O[1], V[0] - O[0], V[1] - O[1], O[0], O[1]].map(f2).join(" ");
  return `<text class="iso-face-text" data-tone="${o.tone || "lo"}" transform="matrix(${m})" font-size="${o.size || 3}" text-anchor="${o.anchor || "start"}">${text}</text>`;
}

// painter: items {at: [x,y,z], svg} sorted far to near, plus an optional bias
export function paint(C, items) {
  return items.map((it) => ({ ...it, key: near(C, it.at) + (it.bias || 0) })).sort((a, b) => a.key - b.key).map((it) => it.svg).join("");
}

export function halo(C, plan, z) {
  return `<path class="iso-halo" d="${poly2d(hull(plan.map((p) => pr(C, [p[0], p[1], z]))))}"/>`;
}

// ruler ticks along x on a strip top at y = edge, every `step`, major every `major`
export function ticksX(C, x0, x1, step, major, y, z, len) {
  let minor = "", maj = "";
  for (let i = 0, x = x0; x <= x1 + 1e-6; i++, x = x0 + i * step) {
    const s = seg(C, [x, y, z], [x, y + (i % major === 0 ? len[1] : len[0]), z]);
    if (i % major === 0) maj += s; else minor += s;
  }
  return `<path class="iso-line" data-tone="lo" d="${minor}"/><path class="iso-line" data-tone="mid" d="${maj}"/>`;
}

// a smooth exponential follow
export const follow = (v, target, dt, tau) => v + (target - v) * (1 - Math.exp(-dt / tau));
