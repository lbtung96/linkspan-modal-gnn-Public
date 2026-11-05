"""Score a trained network on population samples (the paper's 500-deck test set by default).

    python scripts/evaluate.py --checkpoint pretrained/edgegat_N2000.pt \
        --population data/population --list data/splits/test.txt --out results/test

Writes per_sample.csv (MAC per mode against the FE mode shape at every node,
predicted and true frequency) and summary.csv (means, overall and per girder
count, with a bootstrap 95 % interval of the mean over test decks).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAME_RE = re.compile(r"sample_(\d{4})_Ng(\d)")


def bootstrap_ci(v: np.ndarray, n_boot: int = 2000, seed: int = 20260925) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True, help="manifest of sample file names")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    import torch
    from linkspan_gnn.gnn import MODE_NAMES, load_checkpoint, load_sample, mac, predict

    torch.set_num_threads(args.threads)
    model, scaler, ck = load_checkpoint(args.checkpoint)
    files = [s.strip() for s in args.list.read_text(encoding="utf-8").splitlines() if s.strip()]
    rows = []
    for name in files:
        g, f_true = load_sample(args.population / name)
        shapes, f_pred = predict(model, scaler, g)
        target = g.ndata["y"].numpy()
        f_true = f_true.numpy().reshape(-1)
        r = {"sample": name, "Ng": int(NAME_RE.search(name).group(2))}
        for k, m in enumerate(MODE_NAMES):
            r[f"mac_{m}"] = mac(shapes[:, k], target[:, k])
        for k, m in enumerate(MODE_NAMES):
            r[f"freq_pred_{m}_hz"] = float(f_pred[k])
            r[f"freq_true_{m}_hz"] = float(f_true[k])
            r[f"freq_abs_err_{m}_hz"] = float(abs(f_pred[k] - f_true[k]))
        r["mac_mean"] = float(np.mean([r[f"mac_{m}"] for m in MODE_NAMES]))
        r["freq_mae_hz"] = float(np.mean(np.abs(f_pred[None, :] - f_true[None, :])))
        rows.append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_sample.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = [f"mac_{m}" for m in MODE_NAMES] + ["mac_mean"] + [f"freq_abs_err_{m}_hz" for m in MODE_NAMES] + ["freq_mae_hz"]
    summary = []
    for label, sub in [("all", rows)] + [(f"Ng{g}", [r for r in rows if r["Ng"] == g]) for g in (2, 3, 4, 5)]:
        if not sub:
            continue
        s = {"subset": label, "n": len(sub)}
        for c in cols:
            v = np.array([r[c] for r in sub], float)
            s[c] = float(v.mean())
            if c in ("mac_mean", "mac_T1", "freq_mae_hz"):
                s[f"{c}_ci95_lo"], s[f"{c}_ci95_hi"] = bootstrap_ci(v)
        summary.append(s)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(f"{s['subset']:<4} n={s['n']:<4} MAC " + " ".join(f"{m} {s[f'mac_{m}']:.3f}" for m in MODE_NAMES)
              + f" | mean {s['mac_mean']:.3f} | freq MAE {s['freq_mae_hz']:.3f} Hz")
    print(f"checkpoint epoch {ck.get('epoch')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
