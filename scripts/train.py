"""Train the network on one of the nested training sets.

    python scripts/train.py --population data/population --splits data/splits --n 2000 --run-dir runs/edgegat_N2000

The paper's main network is N = 2000 (500 decks per girder count), 5000 epochs,
seed 123, 2 CPU threads; the checkpoint kept is the lowest validation loss.
That takes about 8 h on one laptop core pair; several runs can share a machine.
Re-running the same command after an interruption resumes from the last save.
CPU results are bit-reproducible at a fixed thread count only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--splits", type=Path, help="folder with N{n}_train.txt / N{n}_val.txt")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--train-list", type=Path, help="explicit manifest (overrides --splits/--n)")
    ap.add_argument("--val-list", type=Path)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many epochs (resume later)")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from linkspan_gnn.training import TrainConfig, train

    torch.set_num_threads(args.threads)
    tr = args.train_list or (args.splits / f"N{args.n}_train.txt")
    va = args.val_list or (args.splits / f"N{args.n}_val.txt")
    names = lambda m: [s.strip() for s in Path(m).read_text(encoding="utf-8").splitlines() if s.strip()]  # noqa: E731
    train_paths = [args.population / s for s in names(tr)]
    val_paths = [args.population / s for s in names(va)]
    missing = [p for p in train_paths + val_paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} samples missing, e.g. {missing[0]}")
    summary = train(train_paths, val_paths, args.run_dir, TrainConfig(epochs=args.epochs, seed=args.seed),
                    stop_after=args.stop_after)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
