// Fig 2, "Message passing": node build.mjs [--light]
// Writes fig2-message-passing.html (or -light.html) next to this script: a
// standalone page that draws the bench model of the Ng = 3 deck and runs the
// pretrained network's real numbers (data/deck.json) through it.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { page, kitSource, benchSource } from "../lib/page.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";
const D = JSON.parse(readFileSync(join(HERE, "../data/deck.json"), "utf8"));

const data = {
  xy_mm: D.xy_mm, girders_y_mm: D.girders_y_mm, supports: D.supports,
  edges_undirected: D.edges_undirected, edge_group: D.edge_group, edges_directed: D.edges_directed,
  fe: { freq_hz: D.fe.freq_hz }, pred: D.pred, h: D.h, alpha: D.alpha, psd: D.psd_all,
  coh: D.coh_every4, df: D.freq_df_hz, mode_bins: D.mode_bins, n_params: D.n_params, e_enc: D.e_enc,
};

const W = 760, H = 430;
const svg = `<svg class="iso-svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="f2t"><title id="f2t">Bench model of the deck with an instrument at every node</title>
<defs><linearGradient id="cylG" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="var(--s1)"/><stop offset=".38" stop-color="var(--s3)"/><stop offset=".7" stop-color="var(--s2)"/><stop offset="1" stop-color="var(--s0)"/></linearGradient></defs>
<g data-layer="base"></g><g data-layer="under"></g><g data-layer="deck"></g><g data-layer="top"></g><g data-layer="blocks"></g><g data-layer="pulses"></g><g data-layer="front"></g>
</svg>`;

const btn = (k, v, label, pressed) => `<button data-${k}="${v}" aria-pressed="${pressed ? "true" : "false"}">${label}</button>`;
const controls =
  `<span class="grp">stage ${["encode", "l = 1", "l = 2", "l = 3", "decode"].map((s, i) => btn("stage", i, s, i === 0)).join("")}</span>` +
  `<span class="grp">mode ${["B1", "T1", "B2", "T2"].map((s, i) => btn("mode", i, s, i === 0)).join("")}</span>` +
  `<span class="grp">head ${[1, 2, 3, 4].map((s, i) => btn("head", i, s, i === 0)).join("")}</span>`;

const keys = [
  { mark: "lit", label: "<b>receiver i</b> — the one lit station; hover or click any node to move it" },
  { mark: "hi", label: "<b>H(v)</b>, N × D<sub>v</sub> — the 4 × 8 pin block on a station: its 32 hidden values (4 heads × 8)" },
  { mark: "hi", label: "<b>twin rails</b> on every member — the two directed edges j→i and i→j (E = 222 for 111 members)" },
  { mark: "lit", label: "<b>valve</b> on each in-channel of i — lever angle = α<sub>ij</sub>; per head they sum to 1 (softmax)" },
  { mark: "mid", label: "<b>ψ</b> — a pulse: α<sub>ij</sub>(z<sub>j</sub> + f<sub>ij</sub>), the neighbour's state plus the edge's coherency" },
  { mark: "mid", label: "<b>⊕ aggregation</b> — the collector under i's block: the sum of the arriving messages" },
  { mark: "mid", label: "<b>φ</b> — the pins re-level: + W<sub>res</sub>h<sub>i</sub> + b, Linear, ELU, + h<sub>i</sub>, LayerNorm" },
  { mark: "mid", label: "<b>S(v), S(e)</b> — screens on the left console: the node's PSD (M = 400 bins) and an in-edge's coherency (d<sub>e</sub> = 800)" },
  { mark: "mid", label: "<b>knob</b> — the stage: encode, l = 1, 2, 3 (L = 3), decode" },
  { mark: "mid", label: "<b>F̂</b>, 1 × k — four dials on the right console: predicted B1, T1, B2, T2 in Hz (0–12 Hz scale)" },
  { mark: "lo", label: "<b>mean</b> — manifold under the girders: every node's h, averaged, feeds the frequency head" },
  { mark: "hi", label: "<b>Φ̂</b>, N × k — the deck itself deflects into the predicted mode shape" },
];

const caption = `The graph is the deck: its 70 nodes are the girder nodes of the FE model (rigid-link slaves collapsed onto their master), its 111 members are cross beams, girders, diagonals, the bottom and front beams and the balcony arm, each one a pair of directed edges. <b>Encode</b>: each node's Welch PSD S(v) (400 bins, 0–38.96 Hz) passes an MLP 400→32→32 into H(v); each directed edge's complex coherency S(e) (Re and Im of 400 bins) passes an MLP 800→32→32 into H(e), once — the same H(e) is set into the rails and valves of all three layers. <b>Message passing</b> (EdgeGATConv, 4 heads × 8): z = W h, f<sub>ij</sub> = W<sub>e</sub> e<sub>ij</sub>, score s<sub>ij</sub> = LeakyReLU(a<sub>l</sub>·z<sub>j</sub> + a<sub>r</sub>·z<sub>i</sub> + a<sub>e</sub>·f<sub>ij</sub>), α<sub>ij</sub> = softmax of s over the in-edges of i. The message ψ = α<sub>ij</sub>(z<sub>j</sub> + f<sub>ij</sub>) depends on h<sub>i</sub>, h<sub>j</sub> and e<sub>ij</sub>; the aggregation ⊕ is their sum; φ adds the residual W<sub>res</sub>h<sub>i</sub> + bias, then Linear 32→32, ELU, + h<sub>i</sub> and LayerNorm. Each layer lets i hear one hop further (lit pucks). <b>Decode</b>: the node decoder (MLP 32→32→4) gives the signed mode shape Φ̂ at every node; the mean of h over all nodes feeds the frequency head (MLP 32→32→4) and the training mean and std turn it into Hz. Every number shown is the pretrained network (55,912 parameters, N = 2,000 training decks) run on this deck by <code>data/extract.py</code>: attention α, pin values and predictions are the network's own. Test set of 500 decks: mean MAC 0.933, frequency MAE 0.219 Hz.`;

const live = readFileSync(join(HERE, "live.js"), "utf8");
const script = kitSource() + "\n" + benchSource() + `\nconst D = ${JSON.stringify(data)};\nconst W = ${W}, H = ${H};\n` + live;

const html = page({
  title: "Message passing",
  theme,
  fig: "Fig 2",
  cardTitle: "Message passing",
  hint: "Hover a node · ← → stage · 1–4 mode · H head",
  readout: "encode · node 32",
  label: "Message passing on the deck graph: a bench model of the Ng 3 linkspan deck. Arrow keys step encode, layers 1 to 3 and decode; keys 1 to 4 pick the mode; H cycles the attention head.",
  description: "How the EdgeGAT network passes messages on the deck graph of a linkspan, with the pretrained network's real attention, hidden states and predictions.",
  svg, controls, keys, caption, script,
});
const out = join(HERE, theme === "light" ? "fig2-message-passing-light.html" : "fig2-message-passing.html");
writeFileSync(out, html);
console.log("wrote", out, (html.length / 1024).toFixed(0) + " kB");
