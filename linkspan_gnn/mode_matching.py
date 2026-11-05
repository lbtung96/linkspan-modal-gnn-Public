"""Which of the solved FE modes are B1, T1, B2 and T2?

The FE solver returns modes ordered by frequency, and the order of bending and
torsion modes changes from one deck to the next.  Each structure's candidates
(modes 3 to 15; the first two are excluded as near-rigid-body modes) are scored
against a template of each target mode, computed once on a nominal deck of the
same girder count (``templates/``; two-girder decks have separate templates with
and without balconies).  Templates and candidates are compared at 20 stations
(x = 0.05 L ... L) on every girder.

Stage 1, every target:
    score = (1 - g) * MAC_u3 - g * |f - f_template| / f_template,   g = 0.5
    a candidate is feasible only if MAC_u3 >= 0.85.
Stage 2, torsion targets only: the three best stage-1 candidates are re-ranked
    T1  by girder-to-balcony motion ratio and low balcony energy share
        (a torsion mode of the deck, not a balcony mode),
    T2  by low lateral (u2) contamination and low lateral participation
        (a torsion mode, not a lateral/sway mode).
The four targets are then assigned jointly (Hungarian algorithm) so no FE mode
is used twice.  The generator additionally rejects any structure whose assigned
modes do not all reach MAC 0.85 (``population.MAC_GATE``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .fe_model import G_BALCONY, G_BBRACE, G_BOTTOM, G_CROSS, G_DIAG, G_EDGE, G_FRONT, G_GIRDER

TARGETS = ("B1", "T1", "B2", "T2")


@dataclass(frozen=True)
class MatchConfig:
    min_mac: float = 0.85
    gamma: float = 0.5                     # weight of the frequency penalty in stage 1
    torsion_shortlist_k: int = 3


@dataclass
class TemplateLibrary:
    ng: int
    x_fracs: np.ndarray
    u3: dict[str, np.ndarray]              # target -> (n_stations_total, n_templates)
    freq: dict[str, np.ndarray]            # target -> (n_templates,)
    primary_u3: np.ndarray                 # (n_stations_total, 4), used for sign alignment
    source: str


@dataclass
class MatchResult:
    selected_idx: np.ndarray               # FE mode index of B1, T1, B2, T2
    selected_mac: np.ndarray               # u3 MAC of each against its template


def load_templates(path: str | Path) -> TemplateLibrary:
    z = np.load(Path(path), allow_pickle=False)
    lib_u3 = np.asarray(z["u3"], dtype=float)                  # (n, 4, n_templates)
    lib_f = np.asarray(z["freq_hz"], dtype=float)              # (4, n_templates)
    return TemplateLibrary(
        ng=int(np.asarray(z["Ng"]).item()),
        x_fracs=np.asarray(z["x_fracs"], dtype=float).reshape(-1),
        u3={t: np.asarray(lib_u3[:, j, :], dtype=float) for j, t in enumerate(TARGETS)},
        freq={t: np.asarray(lib_f[j, :], dtype=float).reshape(-1) for j, t in enumerate(TARGETS)},
        primary_u3=np.asarray(lib_u3[:, :, 0], dtype=float),
        source=str(path),
    )


def _mac_to_library(phi: np.ndarray, library: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(phi, dtype=float).reshape(-1)
    R = np.asarray(library, dtype=float)
    den = float(v @ v) * np.sum(R * R, axis=0)
    num = np.abs(v @ R) ** 2
    out = np.zeros(R.shape[1], dtype=float)
    valid = den > eps
    out[valid] = num[valid] / den[valid]
    return np.clip(out, 0.0, 1.0)


def _relative_freq_error(f_ref: np.ndarray, f_cand: float, eps: float = 1e-12) -> np.ndarray:
    ref = np.asarray(f_ref, dtype=float).reshape(-1)
    out = np.zeros(ref.size, dtype=float)
    valid = np.isfinite(ref) & (np.abs(ref) > eps)
    out[valid] = np.abs(float(f_cand) - ref[valid]) / np.abs(ref[valid])
    return out


def _group_nodes(model: dict, groups: set[int]) -> np.ndarray:
    elems = np.asarray(model["elems"], dtype=float)
    mask = np.isin(elems[:, 3].astype(int), np.asarray(sorted(groups), dtype=int))
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    return np.asarray(np.unique(np.r_[elems[mask, 1].astype(int), elems[mask, 2].astype(int)]), dtype=int)


def _dof_values(phi: np.ndarray, node_ids: np.ndarray, dofs: tuple[int, ...]) -> list[np.ndarray]:
    vals = []
    for dof in dofs:
        idx = 6 * (np.asarray(node_ids, dtype=int) - 1) + (int(dof) - 1)
        idx = idx[(idx >= 0) & (idx < phi.size)]
        if idx.size:
            vals.append(phi[idx])
    return vals


def _balcony_metrics(model: dict, phi_full: np.ndarray, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate: balcony share of (u2, u3) energy, and girder-u3 / balcony-motion norm ratio."""
    balcony = _group_nodes(model, {G_BALCONY, G_EDGE, G_BBRACE})
    girder = _group_nodes(model, {G_GIRDER})
    deck = np.setdiff1d(_group_nodes(model, {G_GIRDER, G_CROSS, G_DIAG, G_BOTTOM, G_FRONT,
                                             G_BALCONY, G_EDGE, G_BBRACE}), balcony, assume_unique=False)
    share = np.zeros(modes.size, dtype=float)
    ratio = np.full(modes.size, np.nan, dtype=float)
    for k, m in enumerate(np.asarray(modes, dtype=int)):
        v = np.asarray(phi_full[:, int(m)], dtype=float).reshape(-1)
        if balcony.size and deck.size:
            e_bal = sum(float(np.sum(x ** 2)) for x in _dof_values(v, balcony, (2, 3)))
            e_deck = sum(float(np.sum(x ** 2)) for x in _dof_values(v, deck, (2, 3)))
            share[k] = 0.0 if e_bal + e_deck <= 1e-12 else float(e_bal / (e_bal + e_deck))
        if balcony.size:
            g = _dof_values(v, girder, (3,))
            b = _dof_values(v, balcony, (2, 3))
            g_norm = float(np.linalg.norm(np.concatenate(g))) if g else 0.0
            b_norm = float(np.linalg.norm(np.concatenate(b))) if b else 0.0
            ratio[k] = float(g_norm / max(b_norm, 1e-12))
    return share, ratio


def _lateral_contamination(phi_u2: np.ndarray, phi_u3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u2 / (u2 + u3), by vector norm and by peak, per candidate."""
    norm_u2, norm_u3 = np.linalg.norm(phi_u2, axis=0), np.linalg.norm(phi_u3, axis=0)
    peak_u2, peak_u3 = np.max(np.abs(phi_u2), axis=0), np.max(np.abs(phi_u3), axis=0)
    norm_ratio = np.full(phi_u3.shape[1], np.nan, dtype=float)
    peak_ratio = np.full(phi_u3.shape[1], np.nan, dtype=float)
    vn, vp = norm_u2 + norm_u3 > 1e-12, peak_u2 + peak_u3 > 1e-12
    norm_ratio[vn] = norm_u2[vn] / (norm_u2 + norm_u3)[vn]
    peak_ratio[vp] = peak_u2[vp] / (peak_u2 + peak_u3)[vp]
    return norm_ratio, peak_ratio


def _lateral_participation(part: dict[str, np.ndarray], mode: int) -> tuple[float, float]:
    """Lateral share of participation |G_y| / (|G_y| + |G_z|) and of effective mass."""
    def pick(key: str) -> float:
        arr = np.asarray(part[key], dtype=float).reshape(-1)
        return float(arr[mode]) if 0 <= mode < arr.size else np.nan

    g_y, g_z = abs(pick("gamma_uy")), abs(pick("gamma_uz"))
    m_y = max(0.0, pick("meff_uy")) if np.isfinite(pick("meff_uy")) else np.nan
    m_z = max(0.0, pick("meff_uz")) if np.isfinite(pick("meff_uz")) else np.nan
    g_ratio = m_ratio = np.nan
    if np.isfinite(g_y) and np.isfinite(g_z) and (g_y + g_z) > 1e-12:
        g_ratio = g_y / (g_y + g_z)
    if np.isfinite(m_y) and np.isfinite(m_z) and (m_y + m_z) > 1e-12:
        m_ratio = m_y / (m_y + m_z)
    if not (np.isfinite(g_ratio) or np.isfinite(m_ratio)):
        raise ValueError("participation factors unavailable or degenerate")
    return float(g_ratio), float(m_ratio)


def match_modes(phi_u3: np.ndarray, phi_u2: np.ndarray, freq_hz: np.ndarray, templates: TemplateLibrary, *,
                model: dict, phi_full: np.ndarray, participation: dict[str, np.ndarray],
                config: MatchConfig = MatchConfig()) -> MatchResult:
    """Assign FE modes to B1, T1, B2, T2.

    ``phi_u3`` / ``phi_u2``: candidate modes at the template stations,
    (n_stations_total, n_modes); ``phi_full``: every DOF; ``participation``:
    ``gamma_uy``, ``gamma_uz``, ``meff_uy``, ``meff_uz`` per mode.
    Raises ValueError if some target has no feasible candidate.
    """
    phi_u3 = np.asarray(phi_u3, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float).reshape(-1)
    cand = np.arange(phi_u3.shape[1], dtype=int)
    cand = cand[cand >= 2]
    if cand.size < len(TARGETS):
        raise ValueError(f"need at least {len(TARGETS)} candidate modes beyond the first two")
    u3, u2, f = phi_u3[:, cand], np.asarray(phi_u2, dtype=float)[:, cand], freq_hz[cand]
    g = float(np.clip(config.gamma, 0.0, 1.0))

    balcony_share, girder_balcony_ratio = _balcony_metrics(model, np.asarray(phi_full, dtype=float), cand)
    u2_norm_ratio, u2_peak_ratio = _lateral_contamination(u2, u3)

    n, nt = cand.size, len(TARGETS)
    best_mac = np.full((n, nt), -np.inf)
    best_base = np.full((n, nt), -np.inf)
    part_g = np.full(n, np.nan)
    part_m = np.full(n, np.nan)
    for j, name in enumerate(TARGETS):
        for i in range(n):
            mac_vals = _mac_to_library(u3[:, i], templates.u3[name])
            if name.startswith("T"):
                part_g[i], part_m[i] = _lateral_participation(participation, int(cand[i]))
            base = (1.0 - g) * mac_vals - g * _relative_freq_error(templates.freq[name], f[i])
            k = int(np.argmax(base))
            best_mac[i, j] = float(mac_vals[k])
            best_base[i, j] = float(base[k])

    feasible = np.isfinite(best_base) & (best_mac >= float(config.min_mac))
    base_scores = np.where(feasible, best_base, -np.inf)
    scores = base_scores.copy()

    for name in ("T1", "T2"):              # stage 2: re-rank the torsion shortlist
        j = TARGETS.index(name)
        col = base_scores[:, j]
        finite = np.flatnonzero(np.isfinite(col))
        order = finite[np.argsort(col[finite])[::-1]] if finite.size else np.zeros(0, dtype=int)
        scores[:, j] = -np.inf
        for rank, i in enumerate(order[: min(max(1, int(config.torsion_shortlist_k)), order.size)], start=1):
            comps: list[float] = []
            if name == "T2":
                if np.isfinite(u2_norm_ratio[i]):
                    comps.append(1.0 - float(np.clip(u2_norm_ratio[i], 0.0, 1.0)))
                if np.isfinite(u2_peak_ratio[i]):
                    comps.append(1.0 - float(np.clip(u2_peak_ratio[i], 0.0, 1.0)))
                pt = [v for v in (float(part_g[i]), float(part_m[i])) if np.isfinite(v)]
                if pt:
                    comps.append(1.0 - float(np.clip(np.mean(pt), 0.0, 1.0)))
            else:
                if np.isfinite(girder_balcony_ratio[i]):
                    comps.append(float(girder_balcony_ratio[i] / (1.0 + girder_balcony_ratio[i])))
                if np.isfinite(balcony_share[i]):
                    comps.append(1.0 - float(np.clip(balcony_share[i], 0.0, 1.0)))
            stage2 = float(np.mean(comps)) if comps else 0.0
            scores[i, j] = stage2 + 1e-3 * float(best_base[i, j]) - 1e-6 * float(rank)

    for j, name in enumerate(TARGETS):
        if not np.any(np.isfinite(scores[:, j])):
            raise ValueError(f"no feasible candidate for target {name} (MAC >= {config.min_mac:.2f})")

    finite = np.isfinite(scores)
    profit = np.where(finite, scores, float(np.nanmin(scores[finite])) - 1e6)
    rows, cols = linear_sum_assignment(-profit)
    assigned = np.full(nt, -1, dtype=int)
    for r, c in zip(rows, cols):
        assigned[int(c)] = int(r)
    return MatchResult(selected_idx=np.asarray(cand[assigned], dtype=int),
                       selected_mac=np.asarray([best_mac[assigned[j], j] for j in range(nt)], dtype=float))
