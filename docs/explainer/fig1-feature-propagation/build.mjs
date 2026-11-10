// Fig 1, "Feature propagation": node build.mjs [--light]
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { page, kitSource, benchSource } from "../lib/page.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";
const D = JSON.parse(readFileSync(join(HERE, "../data/deck.json"), "utf8"));
const m = D.measured;
const data = {
  xy_mm: D.xy_mm, girders_y_mm: D.girders_y_mm, supports: D.supports,
  edges_undirected: D.edges_undirected, edge_group: D.edge_group, edges_directed: D.edges_directed,
  df: D.freq_df_hz, mode_bins: D.mode_bins, fe: { freq_hz: D.fe.freq_hz },
  sensors: m.sensors, fs: m.fs_hz, t0: m.t0_s, ns: m.n_samples, sensor_series: m.sensor_series, true_series: m.true_series,
  check: { iters: m.iters, sample0: Object.fromEntries(Object.entries(m.propagated).map(([k, v]) => [k, v.map((r) => r[0])])) },
  psd: m.psd, phase: m.phase, mag: m.coh_mag,
};

const W = 760, H = 430;
const svg = `<svg class="iso-svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="f1t"><title id="f1t">Bench model of the deck with seven accelerometers and a pin at every node</title>
<defs><linearGradient id="cylG" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="var(--s1)"/><stop offset=".38" stop-color="var(--s3)"/><stop offset=".7" stop-color="var(--s2)"/><stop offset="1" stop-color="var(--s0)"/></linearGradient></defs>
<g data-layer="base"></g><g data-layer="under"></g><g data-layer="deck"></g><g data-layer="top"></g>
</svg>`;
const btn = (k, v, label, pressed) => `<button data-${k}="${v}" aria-pressed="${pressed ? "true" : "false"}">${label}</button>`;
const controls =
  `<span class="grp">iterations ${btn("act", "play", "play 0→50", false)}${btn("act", "step", "step", false)}${btn("act", "reset", "k = 0", false)}</span>` +
  `<span class="grp">phase at ${btn("f", "B1", "B1", true)}${btn("f", "T1", "T1", false)}</span>` +
  `<span class="grp">time <input type="range" min="0" max="${m.n_samples - 1}" value="0" data-time aria-label="time sample"></span>`;

const keys = [
  { mark: "hi", label: "<b>accelerometers</b> at 7 girder nodes, cabled to the DAQ box — the only measured nodes" },
  { mark: "hi", label: "<b>pins</b> — each node's band-passed vertical acceleration at the chosen sample (1 unit per 0.3 mm/s²; the support nodes, which carry the ground input, are clipped)" },
  { mark: "mid", label: "<b>clamp collars</b> — sensor pins stay locked to the recording at every iteration" },
  { mark: "lo", label: "<b>faint ring</b> on a free pin — what that node really did (the full simulation), for comparison" },
  { mark: "lit", label: "<b>selected node</b> — hover or click; its PSD is on the analyser screen" },
  { mark: "mid", label: "<b>counter</b> on the DAQ — iteration k of 50" },
  { mark: "mid", label: "<b>analyser</b> — the node's S(v): 400 bins, 0–38.96 Hz, ticks at FE B1, T1, B2, T2" },
  { mark: "hi", label: "<b>phase needles</b> on members — coherency phase at B1 or T1; along the member = 0, reversed = π; length = |γ|" },
];
const caption = `Only the sensor nodes of a measured deck are recorded. Feature propagation fills the rest: X starts at zero with the sensors' values clamped in; each iteration replaces every node by Â X with Â = D<sup>−1/2</sup> A D<sup>−1/2</sup> (one weight per directed edge, D the in-degree), then re-clamps the sensors; 50 iterations. It is linear, so every time sample diffuses on its own: the pins show one sample (scrub the time), and the page iterates it exactly as <code>features.feature_propagation</code> does (checked against <code>data/extract.py</code> at k = 0, 1, 2, 5, 10, 20, 50). The band-passed time series propagated to every node give the coherency of every member, γ<sub>ij</sub>(f) = P<sub>ij</sub>/√(P<sub>ii</sub>P<sub>jj</sub>); the node PSDs are computed at the sensors, propagated the same way and row-normalised. At B1 every member reads in phase. At T1 the deck twists: in this deck the centre girder moves slightly with girder 3, so over the front half the cross beams between girder 1 and the centre girder read about π while those to girder 3 stay near 0; near the hinges |γ| is small and the phase is noise. Data: the Ng = 3 deck (L = 17.8 m), 60 s at 200 Hz, 1 % damping, multi-support white-noise input; samples 10.00–10.995 s.`;

const live = readFileSync(join(HERE, "live.js"), "utf8");
const script = kitSource() + "\n" + benchSource() + `\nconst D = ${JSON.stringify(data)};\nconst W = ${W}, H = ${H};\n` + live;
const html = page({
  title: "Feature propagation", theme, fig: "Fig 1", cardTitle: "Feature propagation",
  hint: "Hover a node · space play · ← → time · B/T phase",
  readout: "k = 0 / 50",
  label: "Feature propagation on the deck: seven accelerometers, a pin at every node. Space plays the 50 iterations, arrow keys move in time, B and T pick the phase frequency.",
  description: "How measured accelerations at a few sensors are propagated to every node of the deck graph before the network sees them.",
  svg, controls, keys, caption, script,
});
const out = join(HERE, theme === "light" ? "fig1-feature-propagation-light.html" : "fig1-feature-propagation.html");
writeFileSync(out, html);
console.log("wrote", out, (html.length / 1024).toFixed(0) + " kB");
