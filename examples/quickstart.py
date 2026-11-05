"""Quick start: the pretrained network on decks it has never seen.

    python examples/quickstart.py

Draws one new deck per girder count (a different seed from the training
population), runs the whole data pipeline on it - FE model, mode labels,
ambient response, graph and features - and compares the network's mode shapes
and frequencies with the FE solution.  Takes about ten seconds.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402

from linkspan_gnn import population  # noqa: E402
from linkspan_gnn.gnn import MODE_NAMES, load_checkpoint, load_sample, mode_macs, predict  # noqa: E402

model, scaler, ck = load_checkpoint(ROOT / "pretrained" / "edgegat_N2000.pt")
print(f"pretrained network: {ck['training']['n_population']} training decks, best epoch {ck['epoch']}\n")

with tempfile.TemporaryDirectory() as tmp:
    population.generate(tmp, n_per_ng=1, seed=7, verbose=False)
    for path in sorted(Path(tmp).glob("sample_*.npz")):
        g, f_true = load_sample(path)
        shapes, f_pred = predict(model, scaler, g)
        macs = mode_macs(shapes, g.ndata["y"].numpy())
        p = json.loads(str(np.load(path)["params_json"]))
        print(f"Ng={p['Ng']}  L={p['L_mm'] / 1000:.1f} m  {g.num_nodes()} nodes  "
              f"balcony={'yes' if p['balcony']['enable'] else 'no '}  diagonals={p['diag_pattern']}")
        for k, m in enumerate(MODE_NAMES):
            print(f"   {m}: MAC {macs[k]:.3f}   f = {f_pred[k]:5.2f} Hz (FE {float(f_true[k]):5.2f} Hz)")
