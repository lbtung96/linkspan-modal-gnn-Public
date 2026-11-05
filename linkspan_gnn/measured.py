"""Applying the network to a measured deck.

A measured deck is described by three things (``MeasuredStructure``):
  * an FE parameter dict for its geometry (the same parameters as the synthetic
    population: span, girder count and spacing, bracing, bottom beam, balconies,
    non-structural mass - anything not given keeps ``fe_model.default_params``);
  * the plan positions of its vertical accelerometers, in mm, with x along the
    span from the hinge (x = 0) and y across, girder 0 at y = 0;
  * the recorded vertical accelerations, one row per sensor, and the sampling rate.
Optionally, reference mode shapes at the sensors and natural frequencies (from an
operational modal analysis such as SSI) for scoring and fine-tuning, in the order
B1, T1, B2, T2.

Graph.  The deck's FE graph (``graph.projected_graph``) gets one extra node per
sensor: each sensor's y is snapped onto the nearest y of any FE node (for a
sensor on a girder, that girder's line), and the sensor is inserted into the
edge it lies on (that edge is replaced by two edges through the sensor) or, if
another sensor already split that edge or it lies on no edge, joined to the
nearest FE node.

Features.  Only the sensor nodes are measured.  The recordings are band-passed
(1.5-20 Hz, as the training response); node PSDs are computed at the sensors,
diffused to every other node by feature propagation (50 iterations, sensors
clamped), mapped onto the 400-bin training grid and row-normalised.  For the
edges, the band-passed time series themselves are propagated to every node and
the complex coherency is computed per edge and mapped onto the grid.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import fe_model as fe
from .features import (N_FREQ_BINS, WELCH_NFFT, WELCH_NOVERLAP, WELCH_NPERSEG, bandpass_filter, edge_coherency,
                       interleave_re_im, normalize_rows_by_max_abs, propagate, to_grid, training_frequency_grid,
                       welch_psd)
from .gnn import EdgeGATModalGNN, FrequencyScaler, graph_from_arrays, load_checkpoint, mac, predict
from .graph import projected_graph

BANDPASS_HZ = (1.5, 20.0)


@dataclass
class MeasuredStructure:
    name: str
    fe_params: dict                      # overrides of fe_model.default_params()
    sensor_xy_mm: np.ndarray             # (n_sensors, 2)
    acc: np.ndarray                      # (n_sensors, n_samples), vertical acceleration (any unit)
    fs_hz: float
    ref_mode_shapes: np.ndarray | None = None   # (n_sensors, 4), B1 T1 B2 T2
    ref_freq_hz: np.ndarray | None = None       # (4,)


def load_measured(path: str | Path) -> MeasuredStructure:
    """Read a measured deck from .npz (keys documented in docs/measured_input_format.md)."""
    with np.load(path, allow_pickle=False) as d:
        get = (lambda k: np.asarray(d[k]) if k in d.files else None)
        return MeasuredStructure(
            name=str(d["name"]) if "name" in d.files else Path(path).stem,
            fe_params=json.loads(str(d["fe_params_json"])),
            sensor_xy_mm=np.asarray(d["sensor_xy_mm"], dtype=float),
            acc=np.asarray(d["acc"], dtype=float),
            fs_hz=float(d["fs_hz"]),
            ref_mode_shapes=None if get("ref_mode_shapes") is None else get("ref_mode_shapes").astype(float),
            ref_freq_hz=None if get("ref_freq_hz") is None else get("ref_freq_hz").astype(float).reshape(-1))


def save_measured(path: str | Path, s: MeasuredStructure) -> None:
    arrays = {"name": np.asarray(s.name), "fe_params_json": np.asarray(json.dumps(s.fe_params)),
              "sensor_xy_mm": np.asarray(s.sensor_xy_mm, dtype=float), "acc": np.asarray(s.acc),
              "fs_hz": np.asarray(float(s.fs_hz))}
    if s.ref_mode_shapes is not None:
        arrays["ref_mode_shapes"] = np.asarray(s.ref_mode_shapes, dtype=float)
    if s.ref_freq_hz is not None:
        arrays["ref_freq_hz"] = np.asarray(s.ref_freq_hz, dtype=float)
    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Graph with sensor nodes
# ---------------------------------------------------------------------------
def _bracketing_edge(sensor_xy: np.ndarray, coords: np.ndarray, undirected: set, y_tol: float = 1.0):
    """(left, right) FE nodes of the girder edge the sensor lies on, or None."""
    sx, gy = float(sensor_xy[0]), float(sensor_xy[1])
    on_girder = np.where(np.abs(coords[:, 1] - gy) < y_tol)[0]
    if len(on_girder) < 2:
        return None
    girder_nodes = on_girder[np.argsort(coords[on_girder, 0])]
    pos = int(np.searchsorted(coords[girder_nodes, 0], sx))
    if pos == 0 or pos >= len(girder_nodes):
        return None
    left, right = int(girder_nodes[pos - 1]), int(girder_nodes[pos])
    return (left, right) if frozenset([left, right]) in undirected else None


def measured_graph(fe_params: dict, sensor_xy_mm: np.ndarray) -> dict:
    """FE graph of the deck plus one node per sensor (sensor nodes last, in sensor order)."""
    model = fe.build_model(fe.params_from_overrides(fe_params))
    g = projected_graph(model)
    fe_xy = np.asarray(g["coords_xy_mm"], dtype=np.float32)
    fe_ei = np.asarray(g["edge_index"], dtype=np.int64)
    n_fe = fe_xy.shape[0]

    sensors = np.asarray(sensor_xy_mm, dtype=np.float32).reshape(-1, 2).copy()
    girder_ys = np.unique(fe_xy[:, 1])
    for i in range(sensors.shape[0]):                     # snap onto the nearest girder line
        sensors[i, 1] = girder_ys[np.argmin(np.abs(girder_ys - sensors[i, 1]))]
    nearest = np.argmin(np.linalg.norm(fe_xy[np.newaxis, :, :] - sensors[:, np.newaxis, :], axis=2), axis=1)
    sensor_idx = np.arange(n_fe, n_fe + sensors.shape[0], dtype=np.int64)

    undirected = {frozenset([int(u), int(v)]) for u, v in fe_ei.T.tolist()}
    split: set[frozenset] = set()
    bracket: dict[int, tuple[int, int] | None] = {}
    for i in range(sensors.shape[0]):
        b = _bracketing_edge(sensors[i], fe_xy, undirected)
        if b is not None and frozenset(b) not in split:
            split.add(frozenset(b))
            bracket[i] = b
        else:
            bracket[i] = None

    edges = [(int(u), int(v)) for u, v in fe_ei.T.tolist() if frozenset([u, v]) not in split]
    for i in range(sensors.shape[0]):
        s = int(sensor_idx[i])
        if bracket[i] is not None:
            left, right = bracket[i]
            edges += [(s, left), (left, s), (s, right), (right, s)]
        else:
            edges += [(s, int(nearest[i])), (int(nearest[i]), s)]
    return {"coords_xy_mm": np.vstack([fe_xy, sensors]), "edge_index": np.array(edges, dtype=np.int64).T,
            "sensor_idx": sensor_idx, "n_fe_nodes": int(n_fe), "fe_graph_node_ids": g["graph_node_ids"]}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def measured_features(graph: dict, acc: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Node features (n_nodes, 400) and edge features (E, 800) from the sensor recordings."""
    acc = np.asarray(acc, dtype=np.float64)
    sensor_idx, edge_index = graph["sensor_idx"], graph["edge_index"]
    n_nodes = int(graph["coords_xy_mm"].shape[0])
    if acc.shape[0] != sensor_idx.size:
        raise ValueError(f"{acc.shape[0]} acceleration channels for {sensor_idx.size} sensors")
    grid = training_frequency_grid()
    acc_bp = bandpass_filter(acc, fs_hz, f_low=BANDPASS_HZ[0], f_high=BANDPASS_HZ[1])
    nperseg = int(min(WELCH_NPERSEG, acc_bp.shape[1]))
    noverlap = int(min(WELCH_NOVERLAP, nperseg // 2))

    f_native, psd = welch_psd(acc_bp, fs_hz, nperseg=nperseg, noverlap=noverlap, nfft=WELCH_NFFT)
    x = np.zeros((n_nodes, psd.shape[1]), dtype=np.float32)
    x[sensor_idx] = psd.astype(np.float32)
    x = propagate(x, sensor_idx, edge_index, n_nodes).astype(np.float64)
    x = normalize_rows_by_max_abs(to_grid(f_native, x, grid)).astype(np.float32)

    ts = np.zeros((n_nodes, acc_bp.shape[1]), dtype=np.float32)
    ts[sensor_idx] = acc_bp.astype(np.float32)
    ts = propagate(ts, sensor_idx, edge_index, n_nodes).astype(np.float64)
    coh = edge_coherency(edge_index, ts, fs_hz, nperseg=nperseg, noverlap=noverlap, nfft=WELCH_NFFT)
    f_coh = np.fft.rfftfreq(int(WELCH_NFFT), d=1.0 / fs_hz)
    edge_x = interleave_re_im(to_grid(f_coh, np.real(coh).astype(np.float64), grid),
                              to_grid(f_coh, np.imag(coh).astype(np.float64), grid))
    assert x.shape[1] == N_FREQ_BINS and edge_x.shape == (edge_index.shape[1], 2 * N_FREQ_BINS)
    return x, edge_x


@dataclass
class Prepared:
    name: str
    graph: dict
    x: np.ndarray
    edge_x: np.ndarray
    ref_mode_shapes: np.ndarray | None
    ref_freq_hz: np.ndarray | None

    def dgl_graph(self):
        return graph_from_arrays(self.graph["edge_index"], self.x, self.edge_x)


def prepare(s: MeasuredStructure) -> Prepared:
    g = measured_graph(s.fe_params, s.sensor_xy_mm)
    x, ex = measured_features(g, s.acc, s.fs_hz)
    return Prepared(s.name, g, x, ex, s.ref_mode_shapes, s.ref_freq_hz)


# ---------------------------------------------------------------------------
# Zero-shot prediction
# ---------------------------------------------------------------------------
def zero_shot(model: EdgeGATModalGNN, scaler: FrequencyScaler, p: Prepared) -> dict:
    """Predicted mode shapes (every node and at the sensors), frequencies, and MAC against the reference."""
    shapes, freq = predict(model, scaler, p.dgl_graph())
    at = shapes[p.graph["sensor_idx"]]
    out = {"structure": p.name, "mode_shapes_all_nodes": shapes, "mode_shapes_at_sensors": at, "freq_hz": freq}
    if p.ref_mode_shapes is not None:
        out["mac"] = [mac(np.asarray(at[:, m], dtype=np.float64), np.asarray(p.ref_mode_shapes[:, m], dtype=np.float64))
                      for m in range(4)]
    return out


# ---------------------------------------------------------------------------
# Fine-tuning, leave-one-structure-out with a separate validation structure
# ---------------------------------------------------------------------------
ENCODER_LR = 1e-5                       # node/edge encoders and attention layers
HEAD_LR = 1e-4                          # decoder and frequency head
FT_WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
FT_MAX_EPOCHS = 500
FT_PATIENCE = 50
FT_FREQ_LOSS_WEIGHT = 0.25
COSINE_ETA_MIN = 1e-7
ENCODER_PARTS = ("node_encoder", "edge_encoder", "layers")
FT_SEED = 123


def _mac_ft(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.dot(a, a) * np.dot(b, b))
    return 0.0 if denom < 1e-24 else float(np.dot(a, b) ** 2 / denom)


def _sign_aligned_macs(pred_at: np.ndarray, ref: np.ndarray) -> list[float]:
    out = []
    for m in range(4):
        a = pred_at[:, m].copy()
        if np.dot(a, ref[:, m]) < 0:
            a = -a
        out.append(_mac_ft(a, ref[:, m]))
    return out


def finetune_fold(checkpoint: str | Path, structures: dict[str, Prepared], held_out: str, val: str,
                  train: list[str]) -> dict:
    """Fine-tune on ``train`` (all weights; sparse loss at the sensors), select the epoch on ``val``,
    score once on ``held_out``."""
    torch.manual_seed(FT_SEED)
    np.random.seed(FT_SEED)
    model, scaler, _ = load_checkpoint(checkpoint)
    enc, head = [], []
    for name, prm in model.named_parameters():
        (enc if any(x in name for x in ENCODER_PARTS) else head).append(prm)
    opt = torch.optim.Adam([{"params": enc, "lr": ENCODER_LR, "weight_decay": FT_WEIGHT_DECAY},
                            {"params": head, "lr": HEAD_LR, "weight_decay": FT_WEIGHT_DECAY}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_MAX_EPOCHS, eta_min=COSINE_ETA_MIN)
    f_mean = torch.tensor(scaler.mean_hz, dtype=torch.float32)
    f_std = torch.tensor(scaler.std_hz, dtype=torch.float32)
    data = {}
    for k in {*train, val, held_out}:
        p = structures[k]
        data[k] = {"g": p.dgl_graph(), "idx": torch.tensor(p.graph["sensor_idx"], dtype=torch.long),
                   "sensor_idx": p.graph["sensor_idx"].astype(int),
                   "y": torch.tensor(p.ref_mode_shapes, dtype=torch.float32),
                   "f": torch.tensor(np.asarray(p.ref_freq_hz).reshape(-1), dtype=torch.float32)}

    def predict_eval(k):
        model.eval()
        with torch.no_grad():
            g = data[k]["g"]
            o = model(g, g.ndata["x"], g.edata["edge_x"])
            return o.mode_shapes.cpu().numpy(), o.freq_z

    best, best_state, best_epoch, patience, epoch = -1.0, None, 0, 0, 0
    for epoch in range(1, FT_MAX_EPOCHS + 1):
        model.train()
        for k in train:
            d = data[k]
            opt.zero_grad()
            o = model(d["g"], d["g"].ndata["x"], d["g"].edata["edge_x"])
            loss = (F.mse_loss(o.mode_shapes[d["idx"], :], d["y"])
                    + FT_FREQ_LOSS_WEIGHT * F.mse_loss(o.freq_z.squeeze(0), (d["f"] - f_mean) / f_std))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
        sched.step()
        pv, _ = predict_eval(val)
        vm = float(np.mean(_sign_aligned_macs(pv[data[val]["sensor_idx"], :], data[val]["y"].numpy())))
        if vm > best:
            best, best_epoch, patience = vm, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= FT_PATIENCE:
                break

    model.load_state_dict(best_state)
    shapes, fz = predict_eval(held_out)
    freq = scaler.to_hz(fz).cpu().numpy().reshape(-1)
    at = shapes[data[held_out]["sensor_idx"], :]
    return {"held_out": held_out, "val": val, "n_train": len(train), "best_epoch": best_epoch, "stopped_epoch": epoch,
            "mac": _sign_aligned_macs(at, data[held_out]["y"].numpy()), "freq_hz": freq.tolist(),
            "mode_shapes_all_nodes": shapes}


def finetune_loocv(checkpoint: str | Path, structures: dict[str, Prepared]) -> list[dict]:
    """Nested leave-one-out: for structure i, validate on structure i + 1 (cyclically), train on the rest."""
    names = list(structures)
    if len(names) < 3:
        raise ValueError("need at least three labelled structures (train, validation, test)")
    results = []
    for i, held in enumerate(names):
        val = names[(i + 1) % len(names)]
        results.append(finetune_fold(checkpoint, structures, held, val, [n for n in names if n not in (held, val)]))
    return results
