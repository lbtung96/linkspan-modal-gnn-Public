# linkspan-modal-gnn

Code for the paper **Automated operational modal analysis of Norwegian linkspans
using graph neural networks**, by Ba Tung Le (Smart Infrastructure Group,
Department of Ocean Operations and Civil Engineering, NTNU, Ålesund) —
manuscript in preparation.

A graph neural network that reads accelerations of a ferry
linkspan deck and predicts its first four modes — **signed** mode shapes at
every node and natural frequencies of B1, T1, B2, T2 (first/second bending and
torsion). It is trained entirely on a synthetic population of finite-element
decks, then applied to measured bridges.

```
 sample deck ──► FE model ──► 15 modes ──► label B1 T1 B2 T2 ──► ambient response ──► graph + features ──► GNN
 (geometry,      (3-D beam     (shift-     (template MAC +       (white noise at       (node PSD,          (EdgeGATConv)
  material,       grid, rigid   invert      frequency, torsion    every support DOF,    edge complex
  springs, mass)  links)        Lanczos)    re-ranking)           modal superposition)  coherency)
```

## Installation

Python 3.10+; everything runs on CPU (for a model this size a GPU was measured to
be no faster than two CPU threads).

```bash
git clone https://github.com/lbtung96/linkspan-modal-gnn-Public.git
cd linkspan-modal-gnn-Public
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/repo.html
pip install -r requirements.txt
python tests/test_pipeline.py                            # 7 tests, about a minute
```

Run the scripts from the repository root (they put it on the path themselves),
or `pip install -e .` to import `linkspan_gnn` from anywhere.

## Quick start

```bash
python examples/quickstart.py
```

draws four new decks (one per girder count, a seed the network was not trained
on), runs the whole pipeline on each and compares the pretrained network with
the FE solution:

```
Ng=3  L=17.8 m  70 nodes  balcony=yes  diagonals=global_x
   B1: MAC 0.968   f =  3.62 Hz (FE  3.60 Hz)
   T1: MAC 0.891   f =  4.84 Hz (FE  4.67 Hz)
   ...
```

The pretrained network (`pretrained/edgegat_N2000.pt`, 56 k parameters) is the
paper's main network; see [pretrained/README.md](pretrained/README.md).

## Reproducing the paper's synthetic results

```bash
python scripts/generate_population.py --out data/population                 # 5500 decks, ~1.5 h on one core
#   or four processes at once, one per girder count:  ... --out data/population --ng 2   (3, 4, 5)
python scripts/make_splits.py --population data/population --out data/splits
python scripts/train.py --population data/population --splits data/splits \
       --n 2000 --run-dir runs/edgegat_N2000                                 # 5000 epochs, ~4.5 h at 2 threads
python scripts/evaluate.py --checkpoint runs/edgegat_N2000/checkpoints/best_model.pt \
       --population data/population --list data/splits/test.txt --out results/test
```

* **Population**: 1375 decks per girder count (Ng = 2…5), seed 20260507. Per
  Ng, the first 1250 decks form the training pool and the next 125 the fixed
  test set (500 decks). A deck is rejected if any of its four labelled modes
  matches its template with MAC < 0.85 (`failures.log` lists every rejection).
* **Population-size sweep**: `make_splits.py` writes nested, girder-balanced
  training sets for N = 100 … 5000 (`--n` in `train.py`); every N is scored on
  the same 500 test decks.
* **Training**: Adam 1e-3, weight decay 1e-4, batch 16, loss = MSE(mode shapes)
  + 0.25 MSE(standardised frequencies), seed 123, best-validation checkpoint.
  An interrupted run resumes when the same command is run again.

`scripts/evaluate.py` with the shipped checkpoint on the regenerated test set
gives the numbers in the model card.

## Applying it to a measured deck

Describe the deck in one `.npz` — FE parameters of its geometry, sensor plan
positions, recorded vertical accelerations, and optionally reference mode
shapes and frequencies (e.g. from SSI) — as in [docs/FORMATS.md](docs/FORMATS.md). Then

```bash
python scripts/predict_measured.py --checkpoint pretrained/edgegat_N2000.pt --structures deck.npz --out results/zero_shot
python scripts/finetune_loocv.py   --checkpoint pretrained/edgegat_N2000.pt --structures a.npz b.npz c.npz ... --out results/finetune
```

Each sensor becomes a node of the deck's graph; only those nodes are measured,
and the rest are filled in by feature propagation before the network runs.
Fine-tuning is leave-one-structure-out with a separate validation structure
(structure *i* + 1) choosing the epoch, so the held-out structure is touched
once. `examples/make_synthetic_measured.py` writes three example input files
from simulated decks, to try both commands.

The field recordings, SSI mode identification and FE geometries of the nine
bridges in the paper are not part of this repository (see the paper's data
availability statement).

## Repository layout

```
linkspan_gnn/
  fe_model.py        parametric 3-D Timoshenko beam model of the deck, rigid links, modal solver
  mode_matching.py   which FE modes are B1, T1, B2, T2 (templates in linkspan_gnn/templates/)
  excitation.py      multi-support white-noise excitation, Newmark modal superposition
  graph.py           FE model -> graph on the deck plane
  features.py        Welch PSD, complex coherency, band-pass, feature propagation
  population.py      sampling ranges and the per-deck pipeline
  gnn.py             EdgeGATConv network, datasets, checkpoints, MAC
  training.py        training loop (resumable)
  measured.py        measured decks: sensor graph, features, zero-shot, fine-tuning
scripts/             generate_population, make_splits, train, evaluate, predict_measured, finetune_loocv
examples/            quickstart, make_synthetic_measured
pretrained/          the paper's main network (N = 2000)
docs/FORMATS.md      sample and measured-deck file formats
tests/               smoke and regression tests
```

## Details worth knowing

* **Excitation.** Each translational DOF of each support node gets its own
  independent white-noise ground acceleration (60 s at 200 Hz, 1 % damping, all
  15 modes, then a 1.5–20 Hz zero-phase band-pass). Because the inputs differ
  from support to support, torsion is excited even on a mirror-symmetric deck.
  The same noise seed is used for every deck.
* **Signs.** Mode shapes are sign-aligned to fixed templates, so the targets are
  signed; MAC, used for scoring, is sign-invariant. The phase of the edge
  coherency is what lets the network tell in-phase from anti-phase motion.
* **Features.** Node PSDs are divided by their maximum; coherency is interleaved
  Re/Im per bin; both cover 0–38.96 Hz in 400 bins of 0.0977 Hz. Measured
  signals at other sampling rates are interpolated onto that grid, not
  truncated.
* **Units.** N, mm, tonne, s throughout the FE model.

## Reproducibility

This code was checked against the code that produced the paper:

| | check | result |
|---|---|---|
| population | the same decks generated by both, same eigen-solver start vector | every stored array bit-identical |
| population | all 5500 decks re-drawn, solved and labelled | same parameters and FE mode labels for every deck, same 1224 rejected draws; frequencies within 1 float32 ulp |
| population | regenerated samples vs the paper's stored ones | mode shapes within 1e-9, features within 1 float32 ulp |
| training | 6 epochs on the same 80 decks, 2 threads; also stopped and resumed | every loss and every weight identical |
| inference | shipped network on the paper's 500 test decks | all 9000 per-deck values identical |
| end to end | population regenerated with these scripts, splits cut, shipped network scored on the new test set | all 18 split files identical; model-card numbers reproduced, per deck MAC within 6e-7 and frequency within 1e-6 Hz |
| measured decks | graph, features, zero-shot and fine-tuning on the nine field-measured bridges | identical to the paper's results |

The one difference: the paper's population was generated with ARPACK's own
random start vector, which changes from run to run and moves the mode shapes by
up to a few 1e-9; this code seeds it, so a regenerated population is identical
on every run (and whether girder counts are generated together or apart), and
matches the paper's to that precision. On CPU, training is
bit-reproducible at a fixed thread count (`--threads`, default 2), not across
thread counts.

## Citation

See [CITATION.cff](CITATION.cff). Licensed under the [MIT License](LICENSE).
