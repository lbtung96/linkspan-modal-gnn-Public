"""Write example input files for a measured deck - from simulated decks.

    python examples/make_synthetic_measured.py                     # -> examples/data/synthetic_deck_*.npz
    python scripts/predict_measured.py --checkpoint pretrained/edgegat_N2000.pt \
        --structures examples/data/synthetic_deck_*.npz --out results/example_zero_shot
    python scripts/finetune_loocv.py --checkpoint pretrained/edgegat_N2000.pt \
        --structures examples/data/synthetic_deck_*.npz --out results/example_finetune

Field data cannot be shipped, so these files stand in for it: each is a new
synthetic deck "instrumented" like a field test - 8 vertical accelerometers on
each outer girder - with 60 s of unfiltered acceleration at those 16 points
only, and the FE mode shapes there as the reference modes.  The format is that
of docs/FORMATS.md; replace geometry, positions and recordings by a real deck's.

Note what this does and does not test.  The network sees 16 measured nodes and
everything else is filled in by feature propagation, exactly as for a real deck.
But the "measurement" is the same simulator the network was trained on, with no
sensor noise and an exactly known geometry, so it says nothing about the gap
between simulation and a real bridge.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from linkspan_gnn import fe_model as fe  # noqa: E402
from linkspan_gnn import population as pop  # noqa: E402
from linkspan_gnn.excitation import FS_HZ, vertical_acceleration  # noqa: E402
from linkspan_gnn.features import normalize_cols_by_max_abs  # noqa: E402
from linkspan_gnn.measured import MeasuredStructure, save_measured  # noqa: E402

SENSOR_X_FRACS = [0.1, 0.22, 0.34, 0.46, 0.58, 0.7, 0.82, 0.94]


def main(n_decks: int = 3, seed: int = 11, out_dir: Path = ROOT / "examples" / "data") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = pop.load_all_templates()
    rng = np.random.default_rng(seed)
    written = []
    while len(written) < n_decks:
        ng = int(rng.choice([3, 4, 5]))
        sampled = pop.sample_params(ng, rng)
        if not pop.is_valid(sampled):
            continue
        params = pop.solver_params(sampled)          # the complete FE parameter dict of this deck
        try:
            solved = pop.solve(params)
            match = pop.label_modes(solved, templates[pop.template_name(ng, bool(sampled["balcony"]["enable"]))])
        except ValueError:
            continue
        stations = fe.girder_station_nodes(solved.model, SENSOR_X_FRACS)            # girder-major
        k = len(SENSOR_X_FRACS)
        sensors = np.r_[stations[:k], stations[-k:]]                                # outer girders only
        xy = solved.model["nodes"][sensors - 1, 1:3]
        acc = vertical_acceleration(solved.model, solved.phi_full, solved.freq_hz, solved.M_full, bandpass_hz=None)
        shapes = normalize_cols_by_max_abs(fe.mode_rows(solved.phi_full[:, match.selected_idx], sensors, 3))
        name = f"synthetic_deck_{len(written) + 1}"
        path = out_dir / f"{name}.npz"
        save_measured(path, MeasuredStructure(name=name, fe_params=params, sensor_xy_mm=xy, acc=acc[sensors - 1],
                                              fs_hz=FS_HZ, ref_mode_shapes=shapes,
                                              ref_freq_hz=solved.freq_hz[match.selected_idx]))
        print(f"{path.name}: Ng={ng}, L={sampled['L_mm'] / 1000:.1f} m, {len(sensors)} sensors, "
              f"f = {np.round(solved.freq_hz[match.selected_idx], 2).tolist()} Hz")
        written.append(path)
    return written


if __name__ == "__main__":
    main()
