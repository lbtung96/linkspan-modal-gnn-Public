"""Training on a list of population samples.

Loss = MSE(mode shapes at every node) + 0.25 * MSE(standardised frequencies),
Adam (lr 1e-3, weight decay 1e-4), batches of 16 graphs, no learning-rate
schedule.  The checkpoint kept is the one with the lowest validation loss
(``best_model.pt``); ``last_model.pt`` is the final epoch.

Runs can be interrupted: every ``resume_every`` epochs the model, the optimiser,
all random states and the history are written to ``resume.pt``, and a
restarted run continues from there.  On CPU, at a fixed thread count, the
continuation is bit-identical to an uninterrupted run.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from .gnn import EdgeGATModalGNN, FrequencyScaler, ModelConfig, SampleDataset, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 5000
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    freq_loss_weight: float = 0.25
    seed: int = 123
    resume_every: int = 25
    model: ModelConfig = field(default_factory=ModelConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss(out, shape_target, freq_hz, scaler: FrequencyScaler, mse: nn.Module, w: float):
    loss_shape = mse(out.mode_shapes, shape_target)
    loss_freq = mse(out.freq_z, scaler.to_z(freq_hz))
    pred_hz = scaler.to_hz(out.freq_z)
    return loss_shape + float(w) * loss_freq, loss_shape, loss_freq, pred_hz


def _run_epoch(model, loader, scaler, mse, w, opt=None) -> tuple[float, float, float, float]:
    tot = shp = frq = mae = 0.0
    for g, f_hz in loader:
        out = model(g, g.ndata["x"], g.edata["edge_x"])
        loss, ls, lf, pred_hz = _loss(out, g.ndata["y"], f_hz, scaler, mse, w)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot += float(loss.detach().item())
        shp += float(ls.detach().item())
        frq += float(lf.detach().item())
        mae += float(torch.mean(torch.abs(pred_hz - f_hz)).detach().item())
    n = max(1, len(loader))
    return tot / n, shp / n, frq / n, mae / n


def train(train_paths: list[Path], val_paths: list[Path], run_dir: str | Path, cfg: TrainConfig = TrainConfig(),
          *, stop_after: int | None = None, log_every: int = 25) -> dict:
    """Train one network; returns the run summary (also written to run_dir/run_summary.json)."""
    if not train_paths or not val_paths:
        raise ValueError("training needs a non-empty training and validation set (the checkpoint is chosen on validation)")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if (run_dir / "run_summary.json").exists():
        return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "resume.pt"

    set_seed(cfg.seed)
    train_set, val_set = SampleDataset(train_paths), SampleDataset(val_paths)
    model_cfg = ModelConfig(**{**asdict(cfg.model), "node_feat_dim": int(train_set.graphs[0].ndata["x"].shape[1]),
                               "edge_feat_dim": int(train_set.graphs[0].edata["edge_x"].shape[1])})
    scaler = FrequencyScaler.fit(train_set.freq_hz)
    train_loader = GraphDataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = GraphDataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    model = EdgeGATModalGNN(model_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    (run_dir / "train_files.txt").write_text("\n".join(p.name for p in train_paths) + "\n", encoding="utf-8")
    (run_dir / "val_files.txt").write_text("\n".join(p.name for p in val_paths) + "\n", encoding="utf-8")

    history: list[dict] = []
    best_val, best_epoch, start, elapsed0 = float("inf"), -1, 0, 0.0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        history, best_val, best_epoch = list(st["history"]), float(st["best_val"]), int(st["best_epoch"])
        start, elapsed0 = int(st["epoch_completed"]) + 1, float(st["elapsed_s"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        if int(st["threads"]) != torch.get_num_threads():
            print(f"[warn] resumed with {torch.get_num_threads()} threads, saved with {st['threads']}")
        print(f"[resume] continuing after epoch {start - 1}", flush=True)

    def save_resume(epoch_completed: int, elapsed: float) -> None:
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"epoch_completed": epoch_completed, "model": model.state_dict(), "opt": opt.state_dict(),
                    "history": history, "best_val": best_val, "best_epoch": best_epoch,
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                    "py_rng": random.getstate(), "threads": torch.get_num_threads(), "elapsed_s": elapsed}, tmp)
        os.replace(tmp, resume_path)

    meta = {"n_train": len(train_set), "n_val": len(val_set), "seed": cfg.seed, "epochs": cfg.epochs}
    print(f"[train] {run_dir.name}: {len(train_set)} train / {len(val_set)} val, {cfg.epochs} epochs, "
          f"{torch.get_num_threads()} threads", flush=True)
    t0, ran = time.time(), 0
    for epoch in range(start, cfg.epochs):
        model.train()
        tr = _run_epoch(model, train_loader, scaler, mse, cfg.freq_loss_weight, opt)
        model.eval()
        with torch.no_grad():
            va = _run_epoch(model, val_loader, scaler, mse, cfg.freq_loss_weight)
        history.append({"epoch": epoch, "train_loss_total": tr[0], "train_loss_shape": tr[1],
                        "train_loss_freq": tr[2], "train_freq_mae_hz": tr[3], "val_loss_total": va[0],
                        "val_loss_shape": va[1], "val_loss_freq": va[2], "val_freq_mae_hz": va[3]})
        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:5d} | train {tr[0]:.4e} | val {va[0]:.4e} | best {best_epoch} "
                  f"| {time.time() - t0 + elapsed0:.0f} s", flush=True)
        if va[0] < best_val:
            best_val, best_epoch = va[0], epoch
            save_checkpoint(ckpt_dir / "best_model.pt", model, model_cfg, scaler, epoch=epoch,
                            train_loss=tr[0], val_loss=va[0], training=meta)
        ran += 1
        done = epoch + 1 == cfg.epochs
        if (epoch + 1) % cfg.resume_every == 0 or done or (stop_after and ran >= stop_after):
            save_resume(epoch, elapsed0 + time.time() - t0)
        if stop_after and ran >= stop_after and not done:
            return {"stopped_after_epoch": epoch}

    save_checkpoint(ckpt_dir / "last_model.pt", model, model_cfg, scaler, epoch=cfg.epochs - 1,
                    train_loss=history[-1]["train_loss_total"], val_loss=history[-1]["val_loss_total"], training=meta)
    with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    summary = {**meta, "best_epoch": best_epoch, "best_val_loss": best_val,
               "final_val_loss": history[-1]["val_loss_total"], "threads": torch.get_num_threads(),
               "torch": torch.__version__, "train_time_s": elapsed0 + time.time() - t0,
               "freq_mean_hz": scaler.mean_hz.tolist(), "freq_std_hz": scaler.std_hz.tolist(),
               "train_config": {k: v for k, v in asdict(cfg).items() if k != "model"}, "model_config": asdict(model_cfg)}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
