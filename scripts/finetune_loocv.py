"""Leave-one-structure-out fine-tuning on measured decks with reference modes.

    python scripts/finetune_loocv.py --checkpoint pretrained/edgegat_N2000.pt \
        --structures deck_a.npz deck_b.npz deck_c.npz ... --out results/finetune

For deck i, deck i + 1 (cyclically, in the order given) is the validation deck
- it picks the epoch - and all others train; deck i is predicted once, at the
end.  Every weight is fine-tuned (encoders and attention layers at 1e-5, the
heads at 1e-4), on the reference mode shapes at the sensors and the reference
frequencies, for at most 500 epochs with patience 50.  Needs at least three decks.
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

    from linkspan_gnn.gnn import MODE_NAMES
    from linkspan_gnn.measured import finetune_loocv, load_measured, prepare

    structures = {}
    for path in args.structures:
        s = load_measured(path)
        if s.ref_mode_shapes is None or s.ref_freq_hz is None:
            raise SystemExit(f"{path}: fine-tuning needs ref_mode_shapes and ref_freq_hz")
        structures[s.name] = prepare(s)
    results = finetune_loocv(args.checkpoint, structures)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        print(f"{r['held_out']:<16} val={r['val']:<16} best epoch {r['best_epoch']:<4} MAC "
              + " ".join(f"{m} {v:.3f}" for m, v in zip(MODE_NAMES, r["mac"])))
        np.savez(args.out / f"{r['held_out']}_finetuned_prediction.npz", mode_names=np.asarray(MODE_NAMES),
                 mode_shapes_all_nodes=r["mode_shapes_all_nodes"], freq_hz=np.asarray(r["freq_hz"]))
        for k, m in enumerate(MODE_NAMES):
            rows.append({"held_out": r["held_out"], "val": r["val"], "n_train": r["n_train"], "mode": m,
                         "mac": r["mac"][k], "pred_freq_hz": r["freq_hz"][k], "best_epoch": r["best_epoch"],
                         "stopped_epoch": r["stopped_epoch"]})
    with (args.out / "finetune_loocv.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("mean MAC: " + " ".join(f"{m} {np.mean([r['mac'] for r in rows if r['mode'] == m]):.3f}" for m in MODE_NAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
