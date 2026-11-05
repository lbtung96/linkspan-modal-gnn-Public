"""Smoke and regression tests.  Run with ``pytest tests`` or ``python tests/test_pipeline.py`` (about a minute)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from linkspan_gnn import fe_model as fe  # noqa: E402
from linkspan_gnn import population as pop  # noqa: E402
from linkspan_gnn.gnn import load_checkpoint, load_sample, mode_macs, predict  # noqa: E402

CKPT = ROOT / "pretrained" / "edgegat_N2000.pt"

# The first deck of each girder count in the paper's population (seed 20260507):
# FE mode index and natural frequency [Hz] of B1, T1, B2, T2, and the span [mm].
PAPER_FIRST_DECKS = {
    2: ([2, 3, 5, 6], [3.693138, 5.680254, 6.231398, 9.966521], 15802.783049096328),
    3: ([2, 3, 4, 5], [3.584917, 4.899928, 5.599998, 8.453856], 18184.16396801555),
    4: ([2, 3, 4, 6], [2.261409, 2.946002, 5.306377, 8.200696], 21946.658222109552),
    5: ([2, 3, 4, 5], [2.976761, 4.13839, 5.054257, 7.750553], 19992.616427197114),
}


def _one_deck(tmp: Path, ng: int) -> Path:
    pop.generate(tmp, n_per_ng=1, ngs=(ng,), verbose=False)
    return next(tmp.glob(f"sample_*_Ng{ng}.npz"))


def test_fe_nominal_deck():
    model = fe.build_model(fe.default_params())
    freq, phi, M = fe.solve_modes(model, 10)
    assert np.all(np.diff(freq) >= 0) and freq[2] > 0.5
    gen_mass = np.array([phi[:, k] @ (M @ phi[:, k]) for k in range(phi.shape[1])])
    assert np.allclose(gen_mass, 1.0, atol=1e-8)                       # mass-normalised


def test_population_sample_schema_and_conventions():
    with tempfile.TemporaryDirectory() as tmp:
        with np.load(_one_deck(Path(tmp), 3)) as z:
            d = {k: z[k] for k in z.files}
        n, e = d["graph_node_ids"].size, d["edge_index"].shape[1]
        assert d["x_psd"].shape == (n, 400) and d["edge_attr_cpsd_full"].shape == (e, 800)
        assert d["x_psd"].max() <= 1.0 + 1e-6 and np.abs(d["edge_attr_cpsd_full"]).max() <= 1.0 + 1e-5
        assert np.abs(d["y_mode_shape_signed"]).max() <= 1.0 + 1e-12      # normalised over all FE nodes
        assert np.all(d["matched_mac"] >= pop.MAC_GATE)
        assert list(d["target_mode_names"]) == ["B1", "T1", "B2", "T2"]


def test_first_decks_match_the_paper_population():
    """Sampler, FE model, mode labels and MAC gate reproduce the paper's decks."""
    import json
    templates = pop.load_all_templates()
    streams = dict(zip(pop.NG_CHOICES, np.random.SeedSequence(pop.SEED).spawn(len(pop.NG_CHOICES))))
    for ng, (modes, freqs, span) in PAPER_FIRST_DECKS.items():
        rng = np.random.default_rng(streams[ng])
        while True:                                                      # first accepted draw
            s = pop.sample_params(ng, rng)
            if not pop.is_valid(s):
                continue
            try:
                solved = pop.solve(pop.solver_params(s))
                match = pop.label_modes(solved, templates[pop.template_name(ng, bool(s["balcony"]["enable"]))])
                break
            except ValueError:
                continue
        assert json.dumps(s["L_mm"]) == json.dumps(span)
        assert match.selected_idx.tolist() == modes
        assert np.allclose(solved.freq_hz[match.selected_idx], freqs, rtol=1e-6, atol=0)


def test_generation_is_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        with np.load(_one_deck(Path(a), 2)) as da, np.load(_one_deck(Path(b), 2)) as db:
            assert da.files == db.files and all(np.array_equal(da[k], db[k]) for k in da.files)


def test_pretrained_network_on_a_new_deck():
    model, scaler, ck = load_checkpoint(CKPT)
    with tempfile.TemporaryDirectory() as tmp:
        pop.generate(tmp, n_per_ng=1, ngs=(4,), seed=7, verbose=False)
        g, f_true = load_sample(next(Path(tmp).glob("sample_*.npz")))
    shapes, f_pred = predict(model, scaler, g)
    macs = mode_macs(shapes, g.ndata["y"].numpy())
    assert shapes.shape == (g.num_nodes(), 4) and np.all(np.isfinite(macs))
    assert min(macs) > 0.5 and np.max(np.abs(f_pred - f_true.numpy())) < 2.0


def test_training_resume_is_exact():
    import torch
    from linkspan_gnn.training import TrainConfig, train
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pop.generate(tmp / "pop", n_per_ng=2, ngs=(2, 3), verbose=False)
        files = sorted((tmp / "pop").glob("sample_*.npz"))
        cfg = TrainConfig(epochs=3)
        train(files[:3], files[3:], tmp / "straight", cfg)
        train(files[:3], files[3:], tmp / "resumed", cfg, stop_after=1)
        train(files[:3], files[3:], tmp / "resumed", cfg)
        a = torch.load(tmp / "straight" / "checkpoints" / "last_model.pt", weights_only=False)["model_state_dict"]
        b = torch.load(tmp / "resumed" / "checkpoints" / "last_model.pt", weights_only=False)["model_state_dict"]
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_measured_deck_end_to_end():
    sys.path.insert(0, str(ROOT / "examples"))
    import make_synthetic_measured
    from linkspan_gnn.measured import finetune_fold, load_measured, prepare, zero_shot
    model, scaler, _ = load_checkpoint(CKPT)
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_synthetic_measured.main(n_decks=3, out_dir=Path(tmp))
        prepared = {p.stem: prepare(load_measured(p)) for p in paths}
    for p in prepared.values():
        assert p.x.shape[0] == p.graph["coords_xy_mm"].shape[0] and p.edge_x.shape[1] == 800
        r = zero_shot(model, scaler, p)
        assert len(r["mac"]) == 4 and np.all(np.isfinite(r["mac"]))
    names = list(prepared)
    r = finetune_fold(CKPT, prepared, held_out=names[0], val=names[1], train=[names[2]])
    assert r["best_epoch"] >= 1 and len(r["mac"]) == 4


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
