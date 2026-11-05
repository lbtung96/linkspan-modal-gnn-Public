"""Generate the synthetic population.

    python scripts/generate_population.py --out data/population                 # the paper's 5500 decks
    python scripts/generate_population.py --out data/population --ng 3          # one girder count only
    python scripts/generate_population.py --out data/tiny --n-per-ng 5          # a quick test

The paper's population is 1375 decks per girder count (Ng = 2..5), seed 20260507:
the first 1250 of each are the training pool, the next 125 the test set
(``make_splits.py``).  About 1 s per deck on one CPU core (1.5 h for 5500); the
four girder counts can run as four processes (``--ng``), since each draws from
its own random stream - the files are identical to a single run's.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from linkspan_gnn import population as pop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-ng", type=int, default=1375)
    ap.add_argument("--ng", type=int, nargs="+", default=list(pop.NG_CHOICES), choices=list(pop.NG_CHOICES))
    ap.add_argument("--seed", type=int, default=pop.SEED)
    ap.add_argument("--labels-only", action="store_true",
                    help="FE + mode labels only, no response simulation (writes summary.csv)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {"n_per_ng": args.n_per_ng, "ng": args.ng, "seed": args.seed, "labels_only": args.labels_only,
           "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
           "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    res = pop.generate(args.out, args.n_per_ng, ngs=tuple(args.ng), seed=args.seed, labels_only=args.labels_only)
    cfg.update(res)
    (args.out / f"generation_config{pop.run_tag(tuple(args.ng))}.json").write_text(json.dumps(cfg, indent=2),
                                                                                   encoding="utf-8")
    print(json.dumps(res["counts"]), f"{res['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
