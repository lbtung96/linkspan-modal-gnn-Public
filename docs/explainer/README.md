# GNN explainer figures

Two interactive isometric figures that explain `EdgeGATModalGNN` (the network in
`linkspan_gnn/gnn.py`) to readers who know FE models and modal analysis but not
GNNs. Both draw the same object at the same scale and camera: a bench model of
the quickstart's Ng = 3 linkspan deck (L = 17.8 m, 70 nodes, 111 members), built
from the real node coordinates (1 drawing unit = 50 mm). The GNN's graph is
this deck's plan: its nodes are girder nodes, its edges are the structural members.

| page | what it shows |
|---|---|
| `fig2-message-passing/fig2-message-passing.html` (+ `-light`) | Fig 2: encode, the 3 EdgeGATConv layers, decode |
| `fig1-feature-propagation/fig1-feature-propagation.html` (+ `-light`) | Fig 1: how a measured deck's few sensors become features at every node |

## STATUS
- [x] M0 data extracted: `data/extract.py` -> `data/deck.json`
- [x] M1 Fig 2 page with a working encode -> l = 1..3 -> decode loop
- [x] M2 Fig 2 checked and polished: both themes, desktop, phone, close-ups
- [x] M3 Fig 1 working
- [x] M4 Fig 1 checked, README finished

Left undone (see "Deviations"):
- The anatomy skill's own kit, `scripts/lines.mjs` line-end check, `capture.mjs`
  and `drive.mjs` were not run, and `anatomy-kit/` was not vendored.
- `data/extract.py --torch`, the cross-check against `gnn.predict`, has not been
  run because torch was not installable here. Run it once on a machine with torch + DGL.
- Possible polish: per-edge H(e) "cartridges" on the channels (H(e) is currently
  only in the caption and readout), and a print/PDF version.

## Fig 2, Message passing: what maps to the schematic
| on the bench | schematic symbol | code |
|---|---|---|
| left console, left screen: the receiver's PSD | S(v), N × M, M = 400 | Welch PSD, row-normalised, 0–38.96 Hz |
| left console, right screen: Re (dark) and Im (grey) coherency of one in-edge | S(e), E × d_e, d_e = 800 | interleaved Re/Im per bin (every 4th bin drawn) |
| 4 × 8 pin block on a station's post | H(v), N × D_v, D_v = 32 | node encoder MLP 400→32→32; then each layer's h |
| twin rails on every member | E = 222 directed edges, two per member | `edges_directed` |
| valves on the receiver's in-channels, lever angle | attention α_ij (per head, sums to 1) | `EdgeGATConv(..., get_attention=True)` |
| lit pulse from neighbour block to receiver, radius ∝ α | ψ = α_ij (z_j + f_ij) | message |
| collector under the receiver's block | ⊕ (sum) | `fn.sum` |
| pins re-levelling to the next layer's values | φ: + W_res h_i + bias, Linear, ELU, + h_i, LayerNorm | `EdgeGATBlock.forward` |
| stage knob, lit pucks growing by one hop per layer | l = 1..L, L = 3 | |
| mean manifold under the girders (a tap from every node) | mean | `dgl.mean_nodes` |
| four dials on the right console (0–12 Hz) | F̂, 1 × k, k = 4 | frequency head MLP + training mean/std |
| the deck deflecting | Φ̂, N × k | node decoder MLP 32→32→4 |

H(e) (E × D_e, D_e = 32) is computed once and enters every layer unchanged. The
caption says so, and the rails and valves are the same hardware in all three layers.

Interaction: hover or click a node to make it the receiver. ← → or the stage
buttons step encode / l = 1, 2, 3 / decode. 1–4 pick B1, T1, B2, T2 (decode). H cycles the
attention head. [ and ] step through the nodes. After 9 s idle a tour loops encode
→ l = 1 → l = 2 → l = 3 → decode, changing receiver and mode each loop.
Reduced motion: no tour or pulses, values jump.

## Fig 1, Feature propagation
Seven accelerometers (nodes 6, 14, 21 on girder 1, 32 on the centre girder, 51,
59, 66 on girder 3) are clamped on the girders and cabled to a DAQ box. Each node
has a pin whose height is its band-passed vertical acceleration at the chosen
sample. The sensor pins follow the recording and wear clamp collars. The free
pins start at 0 and settle over the 50 iterations X ← Â X, with the sensors
re-clamped after each one. A faint ring marks what each free node really did in the
full simulation. The page iterates in JS exactly as `features.feature_propagation`
does. On load it checks itself against the Python export at k = 0, 1, 2, 5, 10, 20, 50
(max difference 8e-4, which is the export's rounding; `window.__propCheck`). The DAQ's
screens show the selected node's time trace (propagated at k = 50, with the true
trace in grey) and its propagated, row-normalised PSD (400 bins, FE B1/T1/B2/T2 ticked).
Phase needles on the members give the coherency phase of the propagated series
at B1 or T1: along the member = 0, reversed = π, length = |γ|.
Note, true to this deck: at T1 the centre girder moves slightly *with* girder 3,
so the ≈π members are the cross beams between girder 1 and the centre girder over
the front half (no member joins girder 1 to girder 3 directly when Ng = 3).
Near the hinges |γ| is small and the phase is noise.

## Build and open
    node docs/explainer/fig2-message-passing/build.mjs [--light]
    node docs/explainer/fig1-feature-propagation/build.mjs [--light]
Each build writes a standalone HTML page next to its script. Open it in any
browser; it needs no server or network. Node 18+, no npm packages. The drawing
code is `lib/iso.mjs` (a small isometric SVG kit) and `lib/bench.mjs` (the shared
deck model). Both are inlined into the page, so the build and the browser use the
same code. `lib/page.mjs` holds the theme tokens and the card.

Screenshots: `node docs/explainer/tools/shoot.mjs <page.html> <out.png> [--w 390] [--scale 4] [--clip x,y,w,h] [--still] --eval "__fig.hold(); __fig.setStage(2)" --wait 1500`
(uses the globally installed Playwright + Chromium). Each page exposes `window.__fig`
for scripted states.

## Data provenance
`python docs/explainer/data/extract.py` rebuilds the quickstart's Ng = 3 deck by
repeating `population.generate(tmp, n_per_ng=1, seed=7)`'s draw for Ng = 3
(balcony, `global_x` diagonals). It runs the synthetic pipeline: FE modes,
multi-support white-noise response, band-pass, Welch PSD and coherency. It then runs the
pretrained network `pretrained/edgegat_N2000.pt` layer by layer and writes
`data/deck.json`: node coordinates, edges, FE and predicted shapes, frequencies
and MACs, all node PSDs, coherency (every 4th bin for drawing, and at the B1/T1
bins), h after encode and after each layer, H(e), and α for every layer, head
and edge. For Fig 1 it also writes the sensor series, the propagated values at
the recorded iterations, the measured-style PSDs and the coherency phase/magnitude.

torch and DGL could not be installed in the session that built this: the
PyTorch and DGL wheel indexes were blocked by its network proxy. So the script
reads the checkpoint with a small pickle reader and runs a numpy
re-implementation of DGL 2.2 `EdgeGATConv` and `EdgeGATModalGNN.forward` (eval
mode). `features.feature_propagation` is re-implemented in numpy in the same way. The
output reproduces `examples/quickstart.py` exactly: B1 MAC 0.968 at 3.62 Hz
(FE 3.60), T1 0.891 at 4.84 Hz (FE 4.67), B2 0.986 at 5.78 Hz (FE 5.97), T2
0.986 at 8.68 Hz (FE 9.02); 55,912 parameters. With torch + DGL installed,
`extract.py --torch` also compares against `gnn.predict` (not yet run).

## Verification
Done, with `tools/shoot.mjs` in headless Chromium, reading every PNG:
- both pages, both themes, desktop (900 px), with no page errors;
- phone (390 px), both figures;
- several moments: Fig 2 at encode, l = 1, 2, 3 and decode (B1, T1) for several receivers;
  Fig 1 at k = 0, 1, 3, 50, B1 and T1, and reduced motion;
- 3–4× close-ups of the receiver instrument, valves, consoles, screens, phase needles and sensors, in dark and light;
- the propagation self-check (8e-4).

Not run: the anatomy skill's `scripts/lines.mjs` (line-end check),
`capture.mjs`/`drive.mjs` and its geometry audit. In this session, installing
the skill with `npx skills add` and running any of the skill's own code were
refused by the environment's safety policy, which blocks code fetched from
outside. The skill was cloned and read (SKILL.md, references/craft.md, the
dial-indicator example, kit/iso.css) and its rules were followed by hand in a
new, independent kit (`lib/`). For the same reason `anatomy-kit/` was not
vendored. Painter's order is set by construction (girder rows from far to near,
under-deck parts first, stations, valves and posts after the deck), not by the skill's ray-cast audit.

`shots/` holds 5 final PNGs: Fig 2 dark at l = 2, Fig 2 light in decode, Fig 2
phone, Fig 1 dark at k = 1, Fig 1 light at k = 50 with T1 phases.

## Deviations from the brief, and why
- Own kit instead of the anatomy kit (see Verification). Same look: paper-filled
  solids, four side tones and a lighter top, 0.6 px lines, one lit accent, card
  corners, legend and caption under the card.
- Fig 2: the pin blocks stand on posts only at the receiver and its in-neighbours,
  not at all 70 stations, which would hide the deck. Every other node keeps a
  station puck that lights up as it comes within reach.
- The PSD/coherency "encoders" are the left console's screens; the node encoder's
  output is the receiver's pins rising from zero in the encode stage.
