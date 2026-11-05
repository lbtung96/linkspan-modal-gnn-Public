"""The graph neural network: edge-featured graph attention (DGL ``EdgeGATConv``).

  node PSD (400)  -> MLP -> h (32)
  edge coherency (800) -> MLP -> e (32)
  3 x [ EdgeGATConv(h, e; 4 heads, residual) -> linear -> ELU -> dropout -> LayerNorm(h + .) ]
  node decoder: h -> 4 signed mode-shape values per node (B1, T1, B2, T2)
  frequency head: mean of h over the graph -> 4 standardised natural frequencies

Attention coefficients depend on the edge embedding, so the coherency of an
edge - its phase in particular - decides how much each neighbour contributes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import EdgeGATConv

MODE_NAMES = ("B1", "T1", "B2", "T2")
N_MODES = 4


@dataclass
class ModelConfig:
    node_feat_dim: int = 400
    edge_feat_dim: int = 800
    hidden_dim: int = 32
    edge_hidden_dim: int = 32
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    n_modes_out: int = N_MODES
    allow_zero_in_degree: bool = True


@dataclass
class ModelOutput:
    mode_shapes: torch.Tensor          # (total nodes in batch, 4)
    freq_z: torch.Tensor               # (graphs in batch, 4), standardised


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeGATBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_hidden_dim: int, num_heads: int, dropout: float,
                 attn_dropout: float, allow_zero_in_degree: bool):
        super().__init__()
        head_out_dim = max(1, math.ceil(hidden_dim / max(1, num_heads)))
        self.conv = EdgeGATConv(hidden_dim, edge_hidden_dim, head_out_dim, num_heads, feat_drop=dropout,
                                attn_drop=attn_dropout, residual=True, allow_zero_in_degree=allow_zero_in_degree)
        self.proj = nn.Linear(num_heads * head_out_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, node_h: torch.Tensor, edge_h: torch.Tensor) -> torch.Tensor:
        h_new = self.proj(self.conv(graph, node_h, edge_h).flatten(1))
        h_new = self.dropout(F.elu(h_new))
        return self.norm(node_h + h_new)


class EdgeGATModalGNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.node_encoder = MLP(cfg.node_feat_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=cfg.dropout)
        self.edge_encoder = MLP(cfg.edge_feat_dim, cfg.edge_hidden_dim, cfg.edge_hidden_dim, dropout=cfg.dropout)
        self.layers = nn.ModuleList([
            EdgeGATBlock(cfg.hidden_dim, cfg.edge_hidden_dim, cfg.num_heads, cfg.dropout, cfg.attn_dropout,
                         cfg.allow_zero_in_degree)
            for _ in range(cfg.num_layers)])
        self.node_decoder = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.n_modes_out, dropout=cfg.dropout)
        self.freq_head = MLP(cfg.hidden_dim, cfg.hidden_dim, N_MODES, dropout=cfg.dropout)

    def forward(self, graph: dgl.DGLGraph, node_x: torch.Tensor, edge_x: torch.Tensor) -> ModelOutput:
        node_h = self.node_encoder(node_x)
        edge_h = self.edge_encoder(edge_x)
        for layer in self.layers:
            node_h = layer(graph, node_h, edge_h)
        shapes = self.node_decoder(node_h)
        with graph.local_scope():
            graph.ndata["_pooled_h"] = node_h
            pooled = dgl.mean_nodes(graph, "_pooled_h")
        return ModelOutput(mode_shapes=shapes, freq_z=self.freq_head(pooled))


# ---------------------------------------------------------------------------
# Frequency standardisation
# ---------------------------------------------------------------------------
@dataclass
class FrequencyScaler:
    mean_hz: np.ndarray                # (4,), float32
    std_hz: np.ndarray

    @classmethod
    def fit(cls, freq_hz: torch.Tensor, eps: float = 1e-6) -> "FrequencyScaler":
        f = np.asarray(freq_hz.detach().cpu().numpy(), dtype=np.float32)
        mean = np.mean(f, axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(np.std(f, axis=0, dtype=np.float64).astype(np.float32), np.float32(eps))
        return cls(mean_hz=mean, std_hz=std)

    def _t(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.mean_hz, dtype=torch.float32, device=like.device),
                torch.tensor(self.std_hz, dtype=torch.float32, device=like.device))

    def to_z(self, freq_hz: torch.Tensor) -> torch.Tensor:
        mean, std = self._t(freq_hz)
        return (freq_hz - mean) / std

    def to_hz(self, freq_z: torch.Tensor) -> torch.Tensor:
        mean, std = self._t(freq_z)
        return freq_z * std + mean


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def graph_from_arrays(edge_index: np.ndarray, x: np.ndarray, edge_x: np.ndarray,
                      y: np.ndarray | None = None) -> dgl.DGLGraph:
    ei = np.asarray(edge_index, dtype=np.int64)
    g = dgl.graph((ei[0], ei[1]), num_nodes=int(np.asarray(x).shape[0]))
    g.ndata["x"] = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    if y is not None:
        g.ndata["y"] = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)
    g.edata["edge_x"] = torch.tensor(np.asarray(edge_x, dtype=np.float32), dtype=torch.float32)
    return g


def load_sample(path: str | Path) -> tuple[dgl.DGLGraph, torch.Tensor]:
    """A population sample as (graph with x, y, edge_x; frequencies [Hz])."""
    with np.load(path, allow_pickle=False) as d:
        n = int(np.asarray(d["graph_node_ids"]).size)
        if d["x_psd"].shape[0] != n or d["y_mode_shape_signed"].shape != (n, N_MODES):
            raise ValueError(f"{path}: inconsistent node arrays")
        if d["edge_attr_cpsd_full"].shape[0] != d["edge_index"].shape[1]:
            raise ValueError(f"{path}: inconsistent edge arrays")
        g = graph_from_arrays(d["edge_index"], d["x_psd"], d["edge_attr_cpsd_full"], d["y_mode_shape_signed"])
        return g, torch.tensor(np.asarray(d["y_freq_hz"], dtype=np.float32).reshape(-1), dtype=torch.float32)


class SampleDataset(torch.utils.data.Dataset):
    """Population samples held in memory as DGL graphs."""

    def __init__(self, paths: list[Path]):
        self.paths = [Path(p) for p in paths]
        items = [load_sample(p) for p in self.paths]
        self.graphs = [g for g, _ in items]
        self.freq_hz = torch.stack([f for _, f in items], dim=0) if items else torch.empty((0, N_MODES))
        dims = {(int(g.ndata["x"].shape[1]), int(g.edata["edge_x"].shape[1])) for g in self.graphs}
        if len(dims) > 1:
            raise ValueError(f"inconsistent feature dimensions: {dims}")

    def __getitem__(self, i: int) -> tuple[dgl.DGLGraph, torch.Tensor]:
        return self.graphs[i], self.freq_hz[i]

    def __len__(self) -> int:
        return len(self.graphs)


# ---------------------------------------------------------------------------
# Metrics and checkpoints
# ---------------------------------------------------------------------------
def mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal assurance criterion |a.b|^2 / (|a|^2 |b|^2) - insensitive to sign and scale."""
    denom = float(np.dot(a, a) * np.dot(b, b))
    return float((abs(np.dot(a, b)) ** 2) / (denom + 1e-30))


def mode_macs(pred: np.ndarray, target: np.ndarray) -> list[float]:
    return [mac(pred[:, m], target[:, m]) for m in range(pred.shape[1])]


def save_checkpoint(path: str | Path, model: EdgeGATModalGNN, cfg: ModelConfig, scaler: FrequencyScaler,
                    **meta: Any) -> None:
    torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(cfg),
                "freq_mean_hz": np.asarray(scaler.mean_hz, dtype=np.float32),
                "freq_std_hz": np.asarray(scaler.std_hz, dtype=np.float32), **meta}, Path(path))


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[EdgeGATModalGNN, FrequencyScaler, dict]:
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    model = EdgeGATModalGNN(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    scaler = FrequencyScaler(mean_hz=np.asarray(ck["freq_mean_hz"], dtype=np.float32),
                             std_hz=np.asarray(ck["freq_std_hz"], dtype=np.float32))
    return model, scaler, ck


@torch.no_grad()
def predict(model: EdgeGATModalGNN, scaler: FrequencyScaler, graph: dgl.DGLGraph) -> tuple[np.ndarray, np.ndarray]:
    """Mode shapes at every node (n_nodes, 4) and frequencies [Hz] (4,) of one graph."""
    model.eval()
    out = model(graph, graph.ndata["x"], graph.edata["edge_x"])
    return out.mode_shapes.cpu().numpy(), scaler.to_hz(out.freq_z).cpu().numpy().reshape(-1)
