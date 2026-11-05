"""The graph the network sees: the FE model projected onto the deck plane.

Every rigid-link slave node is collapsed onto its master (the girder node it is
offset from), so cross beams, braces, the bottom and front beams become edges
between girder nodes.  Free nodes (balcony tips, bottom-beam overhangs) stay as
nodes of their own.  Every structural element becomes one undirected edge;
duplicates and self-loops are dropped.

Node order is ascending FE node id.  Directed edges are the undirected edges
(i < j) followed by the same edges reversed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fe_model import G_RIGID

STRUCTURAL_GROUPS = frozenset({1, 2, 3, 5, 6, 7, 8, 9})     # everything except rigid links


def _representative(slave_to_master: dict[int, int], nid: int) -> int:
    cur, seen = int(nid), set()
    while cur in slave_to_master:
        if cur in seen:
            raise RuntimeError(f"rigid-link loop at node {cur}")
        seen.add(cur)
        cur = int(slave_to_master[cur])
    return cur


def projected_graph(model: dict) -> dict[str, Any]:
    """Projected graph of a model from ``fe_model.build_model``.

    Returns ``graph_node_ids`` (FE ids of the graph nodes), ``coords_xy_mm``,
    ``edge_index_undirected`` (2, E), ``edge_index`` (2, 2E), and per directed
    edge ``edge_length_mm`` and ``edge_theta_rad`` (plan angle).
    """
    nodes = np.asarray(model["nodes"], dtype=float)
    elems = np.asarray(model["elems"], dtype=float)
    slave_to_master = {int(s): int(m) for m, s in np.asarray(model["mpc_pairs"], dtype=int)}

    edge_set = set()
    for e in elems:
        grp = int(e[3])
        if grp == G_RIGID or grp not in STRUCTURAL_GROUPS:
            continue
        r1, r2 = _representative(slave_to_master, int(e[1])), _representative(slave_to_master, int(e[2]))
        if r1 != r2:
            edge_set.add((r1, r2) if r1 < r2 else (r2, r1))
    edges = sorted(edge_set)

    graph_node_ids = np.sort(np.unique(np.asarray(edges, dtype=int).reshape(-1)).astype(int))
    rep_to_idx = {int(rep): i for i, rep in enumerate(graph_node_ids.tolist())}
    und = np.zeros((2, len(edges)), dtype=int)
    for k, (a, b) in enumerate(edges):
        und[0, k], und[1, k] = rep_to_idx[int(a)], rep_to_idx[int(b)]

    coords_xy = np.zeros((graph_node_ids.size, 2), dtype=float)
    for i, rep in enumerate(graph_node_ids.astype(int)):
        coords_xy[i, 0], coords_xy[i, 1] = float(nodes[rep - 1, 1]), float(nodes[rep - 1, 2])

    dxy = coords_xy[und[1]] - coords_xy[und[0]]
    length_u = np.sqrt(np.sum(dxy * dxy, axis=1))
    theta_u = np.arctan2(dxy[:, 1], dxy[:, 0])

    n_und = und.shape[1]
    edge_index = np.zeros((2, 2 * n_und), dtype=int)
    edge_index[:, :n_und] = und
    edge_index[0, n_und:], edge_index[1, n_und:] = und[1], und[0]
    edge_length = np.r_[length_u, length_u]
    edge_theta = np.r_[theta_u, theta_u + np.pi]
    return {
        "graph_node_ids": graph_node_ids.astype(np.int64),
        "coords_xy_mm": coords_xy,
        "edge_index_undirected": und.astype(np.int64),
        "edge_index": edge_index.astype(np.int64),
        "edge_length_mm": np.asarray(edge_length, dtype=float),
        "edge_theta_rad": np.asarray(edge_theta, dtype=float),
    }
