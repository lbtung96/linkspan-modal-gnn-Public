"""Zero-shot prediction on measured decks.

    python scripts/predict_measured.py --checkpoint pretrained/edgegat_N2000.pt --structures my_deck.npz [...] --out results/zero_shot

Each input file describes one deck (geometry, sensor positions, recorded vertical
accelerations; optionally reference mode shapes and frequencies) - see
docs/FORMATS.md.  Writes, per deck, <name>_prediction.npz (mode shapes at every
graph node and at the sensors, frequencies) and, when references are given,
zero_shot.csv with the MAC per mode at the sensors.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--structures", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from linkspan_gnn.gnn import MODE_NAMES, load_checkpoint
    from linkspan_gnn.measured import load_measured, prepare, zero_shot

    model, scaler, _ = load_checkpoint(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.structures:
        p = prepare(load_measured(path))
        r = zero_shot(model, scaler, p)
        np.savez(args.out / f"{p.name}_prediction.npz", mode_names=np.asarray(MODE_NAMES),
                 mode_shapes_all_nodes=r["mode_shapes_all_nodes"], mode_shapes_at_sensors=r["mode_shapes_at_sensors"],
                 freq_hz=r["freq_hz"], node_coords_xy_mm=p.graph["coords_xy_mm"], sensor_idx=p.graph["sensor_idx"],
                 n_fe_nodes=np.asarray(p.graph["n_fe_nodes"]))
        line = f"{p.name:<16} f = " + " ".join(f"{m} {f:.2f}" for m, f in zip(MODE_NAMES, r["freq_hz"])) + " Hz"
        if "mac" in r:
            line += " | MAC " + " ".join(f"{m} {v:.3f}" for m, v in zip(MODE_NAMES, r["mac"]))
            for k, m in enumerate(MODE_NAMES):
                rows.append({"structure": p.name, "mode": m, "mac": r["mac"][k], "pred_freq_hz": float(r["freq_hz"][k]),
                             "ref_freq_hz": float(p.ref_freq_hz[k]) if p.ref_freq_hz is not None else ""})
        print(line)
    if rows:
        with (args.out / "zero_shot.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("mean MAC: " + " ".join(f"{m} {np.mean([r['mac'] for r in rows if r['mode'] == m]):.3f}"
                                     for m in MODE_NAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
