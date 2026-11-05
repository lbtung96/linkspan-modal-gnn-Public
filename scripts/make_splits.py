"""Cut the population into the training pool, the fixed test set and nested training sets.

    python scripts/make_splits.py --population data/population --out data/splits

Within each girder count the generator writes decks in draw order, so:
  pool   the first 1250 decks of every Ng (5000)
  test   the next 125 decks of every Ng (500) - never trained or validated on
  N      the first N/4 pool decks of every Ng, so N = 100 is inside N = 200 is
         inside ... N = 5000; inside each Ng block 15 % go to validation, drawn
         with a seed built from (123, N, Ng)
Writes pool.txt, test.txt, N{N}_train.txt and N{N}_val.txt (file names, one per line).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"sample_(\d{4})_Ng(\d)\.npz$")
NG_VALUES = (2, 3, 4, 5)
TRAIN_FRAC = 0.85
SPLIT_SEED = 123


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool-per-ng", type=int, default=1250)
    ap.add_argument("--test-per-ng", type=int, default=125)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 600, 800, 1000, 2000, 5000])
    args = ap.parse_args()

    by_ng: dict[int, list[str]] = {g: [] for g in NG_VALUES}
    for f in sorted(p.name for p in args.population.glob("sample_*.npz")):
        by_ng[int(NAME_RE.search(f).group(2))].append(f)      # sorted name = draw order within Ng
    need = args.pool_per_ng + args.test_per_ng
    for g, lst in by_ng.items():
        if len(lst) < need:
            raise SystemExit(f"Ng={g}: {len(lst)} decks, need {need}")
    pool = {g: by_ng[g][:args.pool_per_ng] for g in NG_VALUES}
    test = {g: by_ng[g][args.pool_per_ng:need] for g in NG_VALUES}

    args.out.mkdir(parents=True, exist_ok=True)
    write = lambda name, files: (args.out / name).write_text("\n".join(files) + "\n", encoding="utf-8")  # noqa: E731
    write("pool.txt", [f for g in NG_VALUES for f in pool[g]])
    write("test.txt", [f for g in NG_VALUES for f in test[g]])
    test_all = {f for g in NG_VALUES for f in test[g]}
    prev: set[str] = set()
    for n in args.sizes:
        if n % len(NG_VALUES) or n // len(NG_VALUES) > args.pool_per_ng:
            raise SystemExit(f"N={n} must be a multiple of {len(NG_VALUES)} and fit in the pool")
        k = n // len(NG_VALUES)
        tr, va = [], []
        for g in NG_VALUES:
            block = pool[g][:k]
            n_val = int(np.floor((1.0 - TRAIN_FRAC) * k + 0.5))
            perm = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, n, g])).permutation(k)
            va += [block[i] for i in sorted(perm[:n_val])]
            tr += [block[i] for i in sorted(perm[n_val:])]
        write(f"N{n}_train.txt", tr)
        write(f"N{n}_val.txt", va)
        cur = set(tr) | set(va)
        assert prev <= cur and not (cur & test_all) and not (set(tr) & set(va)) and len(cur) == n
        prev = cur
        print(f"N={n}: {len(tr)} train, {len(va)} validation")
    print(f"test: {len(test_all)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
