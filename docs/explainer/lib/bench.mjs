// The shared object of both figures: a bench model of the Ng = 3 deck, built
// from the real plan coordinates in data/deck.json (1 world unit = 50 mm).
import { beam, paint, prism, cyl, rrect, circ, bar, line, seg, dots, tube, ringPath, ticksX, faceText, halo, pr, poly2d, hull } from "./iso.mjs";

export const MM = 50;              // mm per world unit
export const ZD = 30;              // girder top
export const GD = 8;               // girder depth
export const PLATE = { x: -26, y: -88, w: 412, d: 236, r: 9 };
export const PLATE_H = 10, FOOT_H = 6;

const SPEC = {                     // top, depth, width per member group
  girder: [ZD, GD, 4], front: [ZD, GD, 4], cross: [ZD, 5, 2.4], balcony: [ZD, 4, 3],
  bottom: [ZD - GD, 5, 4], diagonal: [ZD - GD, 1.4, 1.6], edge: [ZD, 4, 2.4], balcony_brace: [ZD, 3, 2],
};

export function deckModel(D) {
  const nodes = D.xy_mm.map(([x, y]) => [x / MM, y / MM]);
  const gy = D.girders_y_mm.map((y) => y / MM);
  const onGirder = nodes.map(([, y]) => gy.some((g) => Math.abs(g - y) < 1e-3));
  const L = Math.max(...nodes.map((n) => n[0]));
  const members = D.edges_undirected.map(([a, b], k) => ({ a, b, k, group: D.edge_group[k] }));
  const nbr = nodes.map(() => []);
  for (const [s, t] of D.edges_directed) nbr[t].push(s);
  return { nodes, gy, onGirder, L, members, nbr, supports: D.supports };
}

// the plan bar of one member, trimmed where it lands on a girder's side
export function memberPlan(M, m) {
  const [w] = [SPEC[m.group][2]];
  let A = M.nodes[m.a], B = M.nodes[m.b];
  const dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy);
  const ux = dx / len, uy = dy / len;
  let ta = 0, tb = len;
  if (m.group !== "girder") {
    const outer = (i) => Math.abs(M.nodes[i][1] - M.gy[0]) < 1e-3 || Math.abs(M.nodes[i][1] - M.gy[M.gy.length - 1]) < 1e-3;
    const trimAt = (i) => (m.group === "front" ? (outer(i) ? -2 : 0) : 2);
    if (M.onGirder[m.a]) ta += Math.abs(uy) > 0.2 ? trimAt(m.a) / Math.abs(uy) : 0;
    if (M.onGirder[m.b]) tb -= Math.abs(uy) > 0.2 ? trimAt(m.b) / Math.abs(uy) : 0;
  } else if (Math.abs(B[0] - M.L) < 1e-6) tb -= 2;
  else if (Math.abs(A[0] - M.L) < 1e-6) ta += 2;
  const a = [A[0] + ux * ta, A[1] + uy * ta], b = [A[0] + ux * tb, A[1] + uy * tb];
  return { a, b, w, ta: ta / len, tb: tb / len, len };
}

// the deck, painted in rows from the far girder to the near one; dz[node] lifts each node.
// o.rails(m, zAt) returns the twin-rail lines of member m (drawn on its top face).
export function deckSvg(C, M, dz, o = {}) {
  const rails = o.rails || (() => "");
  const zAt = (m, x, y) => {
    const A = M.nodes[m.a], B = M.nodes[m.b], L2 = (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2;
    const t = Math.max(0, Math.min(1, ((x - A[0]) * (B[0] - A[0]) + (y - A[1]) * (B[1] - A[1])) / L2));
    return dz[m.a] + (dz[m.b] - dz[m.a]) * t;
  };
  const one = (m) => {
    const [top, depth, w] = SPEC[m.group], p = memberPlan(M, m);
    const st = [p.a, p.b].map(([x, y]) => ({ x, y, zt: top + zAt(m, x, y), zb: top - depth + zAt(m, x, y) }));
    return { at: [(p.a[0] + p.b[0]) / 2, (p.a[1] + p.b[1]) / 2, top], svg: beam(C, st, w, { inner: rails(m, (x, y) => top + zAt(m, x, y)) }) };
  };
  const gIdx = (i) => M.gy.findIndex((g) => Math.abs(g - M.nodes[i][1]) < 1e-3);
  const under = [], rows = M.gy.map(() => ({ girder: [], between: [] })), after = [];
  for (const m of M.members) {
    if (m.group === "diagonal" || m.group === "bottom") under.push(one(m));
    else if (m.group === "girder") rows[gIdx(m.a)].girder.push(m);
    else if (m.group === "front") after.push(one(m));
    else if (M.onGirder[m.a] && M.onGirder[m.b]) rows[Math.max(gIdx(m.a), gIdx(m.b))].between.push(one(m));
    else { const g = M.onGirder[m.a] ? gIdx(m.a) : gIdx(m.b); (g === 0 ? after : rows[g].between).push(one(m)); }
  }
  let out = paint(C, under);
  for (let g = M.gy.length - 1; g >= 0; g--) {
    const segs = rows[g].girder.sort((p, q) => M.nodes[p.a][0] + M.nodes[p.b][0] - M.nodes[q.a][0] - M.nodes[q.b][0]);
    const ids = [...new Set(segs.flatMap((m) => [m.a, m.b]))].sort((p, q) => M.nodes[p][0] - M.nodes[q][0]);
    const st = ids.map((i) => ({ x: Math.min(M.nodes[i][0], M.L - 2), y: M.nodes[i][1], zt: ZD + dz[i], zb: ZD - GD + dz[i] }));
    out += beam(C, st, 4, { inner: segs.map((m) => rails(m, (x, y) => ZD + zAt(m, x, y))).join("") });
    out += paint(C, rows[g].between);
  }
  return out + paint(C, after);
}

export function nodeTop(M, i) {
  const [x, y] = M.nodes[i];
  if (M.onGirder[i]) return ZD;
  return Math.abs(x - 231.02) < 1 ? ZD - GD : ZD;     // bottom-beam ends sit lower
}

// the station puck at every node
export function puckItems(C, M, dz, toneOf) {
  return M.nodes.map(([x, y], i) => {
    const t = toneOf ? toneOf(i) : { tone: "mid" };
    const z = nodeTop(M, i) + dz[i];
    return { at: [x, y, z + 1], bias: 0.5, svg: cyl(C, x, y, 1.9, z, 1.4, { tone: t.tone, lit: t.lit, n: 24 }) + (t.dot ? dots(C, [[x, y, z + 1.4]], 0.55, t.dot) : "") , i };
  });
}

// hinge bearings (x = 0) and bottom-beam bearings, base to member
export function pedestalItems(C, M) {
  const items = [];
  for (const i of M.supports) {
    const [x, y] = M.nodes[i];
    const zTopOf = nodeTop(M, i) - (M.onGirder[i] ? GD : 5);
    const zPin = zTopOf - 4;
    const pad = rrect(x - 6, y - 6, 12, 12, 1.5);
    let s = prism(C, pad, 0, 2, { tone: "lo" });
    s += prism(C, rrect(x - 4.5, y - 4.5, 9, 9, 1), 2, zPin - 6, { tone: "mid" });
    s += dots(C, [[x - 4.4, y - 4.4, 2], [x + 4.4, y + 4.4, 2], [x + 4.4, y - 4.4, 2]], 0.45, "mid");
    // cheek plates and the hinge pin across them
    s += prism(C, rrect(x - 3.5, y + 2.6, 7, 1.4, 0.4), zPin - 4, zTopOf - zPin + 4, { tone: "mid" });
    s += tube(C, [[x, y - 3.4, zPin], [x, y + 3.4, zPin]], 2.6);
    s += prism(C, rrect(x - 3.5, y - 4, 7, 1.4, 0.4), zPin - 4, zTopOf - zPin + 4, { tone: "mid" });
    items.push({ at: [x, y, 6], svg: s, bias: -80 });
  }
  return items;
}

export function baseSvg(C, o = {}) {
  const P = PLATE;
  const plan = rrect(P.x, P.y, P.w, P.d, P.r, 4);
  let s = halo(C, plan, -PLATE_H - FOOT_H);
  const feet = [[P.x + 22, P.y + 22], [P.x + P.w - 22, P.y + 22], [P.x + 22, P.y + P.d - 22], [P.x + P.w - 22, P.y + P.d - 22]];
  feet.sort((a, b) => (-a[0] * C.sa + a[1] * C.ca) > (-b[0] * C.sa + b[1] * C.ca) ? -1 : 1);
  for (const [x, y] of feet) {
    s += cyl(C, x, y, 9, -PLATE_H - FOOT_H, 1.8, { tone: "lo" });
    s += cyl(C, x, y, 7, -PLATE_H - FOOT_H + 1.8, FOOT_H - 1.8, { tone: "mid" });
  }
  s += prism(C, plan, -PLATE_H, PLATE_H, { tone: "mid" });
  // the bevel lip and groove on the top
  s += `<path class="iso-line" data-tone="lo" d="${poly2d(rrect(P.x + 6, P.y + 6, P.w - 12, P.d - 12, P.r - 4, 4).map((p) => pr(C, [p[0], p[1], 0])))}"/>`;
  const screws = [[P.x + 12, P.y + 12], [P.x + P.w - 12, P.y + 12], [P.x + 12, P.y + P.d - 12], [P.x + P.w - 12, P.y + P.d - 12]];
  s += screws.map(([x, y]) => `<path class="iso-line" data-tone="lo" d="${ringPath(C, x, y, 2.4, 0, 16)}"/>`).join("");
  s += dots(C, screws.map(([x, y]) => [x, y, 0]), 0.55, "mid");
  // ruler strip along the back, 0 to L in 0.25 m steps, a major tick every metre
  const L = o.L || 357;
  const ry = 124;
  s += prism(C, rrect(-6, ry, L + 12, 8, 1), 0, 0.7, { tone: "lo", crease: "faint" });
  s += ticksX(C, 0, L, 5, 4, ry + 0.9, 0.7, [2.2, 4]);
  s += dots(C, [[-3, ry + 4, 0.7], [L + 3, ry + 4, 0.7]], 0.45, "mid");
  // label plate, back left: the one engraved label
  const lp = { x: P.x + 14, y: 104, w: 34, d: 16 };
  s += prism(C, rrect(lp.x, lp.y, lp.w, lp.d, 1.2), 0, 0.9, { tone: "mid" });
  s += `<path class="iso-line" data-tone="faint" d="${poly2d(rrect(lp.x + 1.6, lp.y + 1.6, lp.w - 3.2, lp.d - 3.2, 0.6).map((p) => pr(C, [p[0], p[1], 0.9])))}"/>`;
  s += faceText(C, [lp.x + 4, lp.y + 10.5, 0.9], [1, 0, 0], [0, -1, 0], o.label || "Ng 3 · L 17.8 m", { size: 3.4, tone: "mid" });
  s += line(C, [[lp.x + 4, lp.y + 6.2, 0.9], [lp.x + 24, lp.y + 6.2, 0.9]], "lo", ' data-free=""');
  s += line(C, [[lp.x + 4, lp.y + 3.8, 0.9], [lp.x + 17, lp.y + 3.8, 0.9]], "lo", ' data-free=""');
  s += dots(C, [[lp.x + 1.9, lp.y + lp.d / 2, 0.9], [lp.x + lp.w - 1.9, lp.y + lp.d / 2, 0.9]], 0.45, "mid");
  return s;
}

// the deck plan's corners, for fitting the camera
export function benchCorners(extraZ = 0) {
  const P = PLATE;
  return [[P.x, P.y, -PLATE_H - FOOT_H], [P.x + P.w, P.y, -PLATE_H - FOOT_H], [P.x, P.y + P.d, -PLATE_H - FOOT_H], [P.x + P.w, P.y + P.d, -PLATE_H - FOOT_H],
    [P.x, P.y + P.d, ZD + extraZ], [P.x + P.w, P.y + P.d, ZD + extraZ], [P.x, P.y, ZD], [P.x + P.w, P.y, ZD]];
}

export { prism, cyl, rrect, circ, bar, line, seg, dots, tube, ringPath, faceText, pr, poly2d, hull };
