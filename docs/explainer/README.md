# GNN explainer figures

Two interactive isometric figures that explain `EdgeGATModalGNN` (the network in
`linkspan_gnn/gnn.py`) to readers who know FE models and modal analysis.

STATUS (updated at every milestone)
- [x] M0 data extracted: `data/extract.py` -> `data/deck.json`
- [x] M1 Fig 2 (message passing) page with encode -> l = 1..3 -> decode loop
- [ ] M2 Fig 2 verified and polished
- [ ] M3 Fig 1 (feature propagation) working
- [ ] M4 Fig 1 verified, README finished

## Data provenance
`python docs/explainer/data/extract.py` rebuilds the quickstart's Ng = 3 deck
(L = 17.8 m, 70 nodes, 111 members = 222 directed edges, balcony, `global_x`
diagonals), runs the synthetic pipeline and the pretrained network
`pretrained/edgegat_N2000.pt` layer by layer, and writes `data/deck.json`.

torch and DGL could not be installed in the session that built this (the
PyTorch and DGL wheel indexes were blocked by the network proxy), so the script
reads the checkpoint with a small pickle reader and runs a numpy
re-implementation of DGL 2.2 `EdgeGATConv` and `EdgeGATModalGNN.forward` (eval
mode). Its output reproduces `examples/quickstart.py` exactly: B1 MAC 0.968 at
3.62 Hz (FE 3.60), T1 MAC 0.891 at 4.84 Hz (FE 4.67), B2 0.986 at 5.78 Hz
(FE 5.97), T2 0.986 at 8.68 Hz (FE 9.02); 55,912 parameters. With torch + DGL
installed, `extract.py --torch` also compares against `gnn.predict`.

## Build and open
    node docs/explainer/fig2-message-passing/build.mjs [--light]
    node docs/explainer/fig1-feature-propagation/build.mjs [--light]
Each writes a standalone HTML page next to its script (`fig2-message-passing.html`,
`fig2-message-passing-light.html`, ...); open it in any browser. Node 18+.
Screenshots: `node docs/explainer/tools/shoot.mjs <page.html> <out.png> --eval "__fig.setStage(2)" --wait 2000`
(uses the Playwright + Chromium that are installed globally).
