"""Synthetic population of linkspan decks, one training sample per deck.

For every deck:
  1. draw geometry, material, support stiffness and non-structural mass
     (``sample_params``);
  2. build the FE model and solve the first 15 modes (``fe_model``);
  3. label B1, T1, B2, T2 among them (``mode_matching``); reject the deck if any
     label matches its template with MAC < 0.85;
  4. simulate 60 s of ambient vertical acceleration at every node with all 15
     modes, band-pass 1.5-20 Hz (``excitation``);
  5. project the FE model onto a graph and compute node PSDs and edge coherency
     (``graph``, ``features``);
  6. store inputs and targets: signed, max-abs-normalised mode shapes at every
     graph node and the four natural frequencies.

Decks are drawn girder count by girder count (Ng = 2, 3, 4, 5), each from its
own random stream spawned from one seed, so the first k decks of a girder
count do not depend on how many decks are generated in total.
"""

from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import fe_model as fe
from .excitation import BANDPASS_HZ, FS_HZ, vertical_acceleration
from .features import N_FREQ_BINS, edge_coherency, interleave_re_im, normalize_cols_by_max_abs, \
    normalize_rows_by_max_abs, welch_psd
from .graph import projected_graph
from .mode_matching import TARGETS, MatchConfig, MatchResult, TemplateLibrary, load_templates, match_modes

SEED = 20260507
NG_CHOICES = (2, 3, 4, 5)
N_CANDIDATE_MODES = 15
MAC_GATE = 0.85
X_FRACS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]     # template stations, fraction of L
MAX_ATTEMPTS_PER_NG = 20000
MAX_BALCONY_TRIES = 40

# sampling ranges
L_RANGE_MM = (14000.0, 22000.0)
NX_RANGE = (15, 26)                       # integers, upper bound exclusive
DY_NG2_MM = (4000.0, 4500.0)
DY_MAX_MM = 3000.0                        # Ng >= 3: dy in [max(2000, 4500/(Ng-1)), min(3000, 12000/(Ng-1))]
DIAG_CHOICES_NG4 = ["global_x", "staggered_x", "single_partial_x", "None"]
DIAG_CHOICES = ["global_x", "staggered_x", "None"]
E_RANGE = (160000.0, 230000.0)            # MPa
RHO_RANGE = (7.0e-9, 8.0e-9)              # tonne/mm^3
KX_RANGE = (1750.0, 2250.0)               # kN/m, hinge springs
KZ_RANGE = (17500.0, 22500.0)             # kN/m, bottom-beam springs
BOTTOM_X_FRAC = (0.55, 0.70)
BOTTOM_OVERHANG_MM = (500.0, 600.0)
NSM_TONNE = (25.0, 35.0)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_balcony(rng: np.random.Generator, L_mm: float) -> dict[str, Any]:
    """Half of the decks get balconies near the free end, on one or both sides."""
    if not bool(rng.random() < 0.5):
        return {"enable": False, "side": "none", "x_center_frac": np.nan, "nBeams": 0,
                "dx_mm": np.nan, "length_mm": np.nan, "edge_beam_enable": False, "brace_enable": False}
    side = str(rng.choice(["left", "right", "both"], p=[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]))
    for _ in range(MAX_BALCONY_TRIES):
        x_center_frac = float(rng.uniform(0.85, 0.92))
        n_beams = int(rng.integers(1, 4))
        dx_mm = float(rng.uniform(500.0, 1000.0))
        length_mm = float(rng.uniform(2000.0, 3000.0))
        centers = x_center_frac * L_mm + (np.arange(n_beams, dtype=float) - (n_beams - 1) / 2.0) * dx_mm
        margin = 100.0
        if not (np.all(centers - 0.5 * length_mm >= margin) and np.all(centers + 0.5 * length_mm <= L_mm - margin)):
            continue
        multi = n_beams > 1                   # several balcony beams always get an edge beam and a brace
        return {"enable": True, "side": side, "x_center_frac": x_center_frac, "nBeams": n_beams,
                "dx_mm": dx_mm, "length_mm": length_mm, "edge_beam_enable": multi, "brace_enable": multi}
    raise ValueError("balcony does not fit the span after retries")


def sample_params(ng: int, rng: np.random.Generator) -> dict[str, Any]:
    """One deck with ``ng`` girders.  The order of the random draws is part of the population."""
    L_mm = float(rng.uniform(*L_RANGE_MM))
    Nx = int(rng.integers(*NX_RANGE))
    if ng <= 1:
        raise ValueError(f"invalid Ng={ng}")
    if ng == 2:
        dy_low, dy_high = DY_NG2_MM               # two-girder decks need wider bays
    else:
        dy_low = max(2000.0, 4500.0 / (ng - 1))
        dy_high = min(DY_MAX_MM, 12000.0 / (ng - 1))
    dy_mm = float(rng.uniform(dy_low, dy_high))
    diag_pattern = str(rng.choice(DIAG_CHOICES_NG4 if ng == 4 else DIAG_CHOICES))
    balcony = sample_balcony(rng, L_mm)
    return {
        "L_mm": L_mm,
        "E": float(rng.uniform(*E_RANGE)),
        "rho": float(rng.uniform(*RHO_RANGE)),
        "Nx": Nx,
        "Ng": int(ng),
        "dy_mm": dy_mm,
        "diag_pattern": diag_pattern,
        "spring": {
            "kx_kN_per_m": float(rng.uniform(*KX_RANGE)),
            "kz_kN_per_m": float(rng.uniform(*KZ_RANGE)),
        },
        "bottom_beam": {
            "x_frac": float(rng.uniform(*BOTTOM_X_FRAC)),
            "overhang_mm": float(rng.uniform(*BOTTOM_OVERHANG_MM)),
        },
        "balcony": balcony,
        "nsm": {
            "enable": True,
            "total_mass_tonne": float(rng.uniform(*NSM_TONNE)),
        },
    }


def is_valid(sampled: dict[str, Any]) -> bool:
    ng, dy = int(sampled["Ng"]), float(sampled["dy_mm"])
    if ng == 2:
        if not (DY_NG2_MM[0] <= dy <= DY_NG2_MM[1]):
            return False
    elif not (2000.0 <= dy <= DY_MAX_MM) or not (4500.0 <= (ng - 1) * dy <= 12000.0):
        return False
    if ng != 4 and str(sampled["diag_pattern"]) == "single_partial_x":
        return False
    b = sampled["balcony"]
    if bool(b["enable"]):
        n = int(b["nBeams"])
        if n > 1 and not (bool(b["edge_beam_enable"]) and bool(b["brace_enable"])):
            return False
        if n == 1 and (bool(b["edge_beam_enable"]) or bool(b["brace_enable"])):
            return False
    return True


def solver_params(sampled: dict[str, Any], n_modes: int = N_CANDIDATE_MODES) -> dict[str, Any]:
    """Full FE parameter dict for a sampled deck (defaults for everything not sampled)."""
    p = deepcopy(fe.default_params())
    p["L_mm"] = float(sampled["L_mm"])
    p["E"] = float(sampled["E"])
    p["rho"] = float(sampled["rho"])
    p["G"] = p["E"] / (2 * (1 + p["nu"]))
    p["Nx"] = int(sampled["Nx"])
    p["Ng"] = int(sampled["Ng"])
    p["dy_mm"] = float(sampled["dy_mm"])
    p["diag_pattern"] = str(sampled["diag_pattern"])
    p["spring"]["kx_kN_per_m"] = float(sampled["spring"]["kx_kN_per_m"])
    p["spring"]["kz_kN_per_m"] = float(sampled["spring"]["kz_kN_per_m"])
    p["spring"]["kx_N_per_mm"] = float(fe.kN_per_m_to_N_per_mm(p["spring"]["kx_kN_per_m"]))
    p["spring"]["kz_N_per_mm"] = float(fe.kN_per_m_to_N_per_mm(p["spring"]["kz_kN_per_m"]))
    p["bottom_beam"]["x_frac"] = float(sampled["bottom_beam"]["x_frac"])
    p["bottom_beam"]["overhang_mm"] = float(sampled["bottom_beam"]["overhang_mm"])
    p["balcony"]["enable"] = bool(sampled["balcony"]["enable"])
    if p["balcony"]["enable"]:
        for k, cast in (("side", str), ("x_center_frac", float), ("nBeams", int), ("dx_mm", float),
                        ("length_mm", float), ("edge_beam_enable", bool), ("brace_enable", bool)):
            p["balcony"][k] = cast(sampled["balcony"][k])
    else:
        p["balcony"]["side"] = "both"
    p["nsm"]["enable"] = True
    p["nsm"]["total_mass_tonne"] = float(sampled["nsm"]["total_mass_tonne"])
    p["nmodes"] = int(n_modes)
    return p


# ---------------------------------------------------------------------------
# One deck
# ---------------------------------------------------------------------------
@dataclass
class Solved:
    model: dict
    freq_hz: np.ndarray            # (n_modes,)
    phi_full: np.ndarray           # (6 * n_nodes, n_modes), mass-normalised
    M_full: Any
    station_nodes: np.ndarray      # girder nodes at the template stations, girder-major
    phi_u3: np.ndarray             # (n_stations_total, n_modes)
    phi_u2: np.ndarray
    participation: dict[str, np.ndarray]


def solve(params: dict[str, Any]) -> Solved:
    model = fe.build_model(params)
    freq_hz, phi_full, M_full = fe.solve_modes(model, int(params["nmodes"]))
    stations = fe.girder_station_nodes(model, X_FRACS)
    supports = fe.support_nodes(model)
    gamma_uy, meff_uy = fe.modal_participation(np.asarray(phi_full, dtype=float), M_full,
                                               fe.influence_vector(model, supports, 2))
    gamma_uz, meff_uz = fe.modal_participation(np.asarray(phi_full, dtype=float), M_full,
                                               fe.influence_vector(model, supports, 3))
    return Solved(model=model, freq_hz=np.asarray(freq_hz, dtype=float), phi_full=np.asarray(phi_full, dtype=float),
                  M_full=M_full, station_nodes=stations,
                  phi_u3=fe.mode_rows(phi_full, stations, 3), phi_u2=fe.mode_rows(phi_full, stations, 2),
                  participation={"gamma_uy": gamma_uy, "gamma_uz": gamma_uz, "meff_uy": meff_uy, "meff_uz": meff_uz})


def template_name(ng: int, balcony_enabled: bool) -> str:
    if ng == 2:
        return "Ng2_balcony" if balcony_enabled else "Ng2_no_balcony"
    return f"Ng{ng}"


def load_all_templates(template_dir: Path = TEMPLATE_DIR) -> dict[str, TemplateLibrary]:
    out = {}
    for ng in NG_CHOICES:
        for bal in ((True, False) if ng == 2 else (False,)):
            name = template_name(ng, bal)
            lib = load_templates(template_dir / f"{name}.npz")
            if lib.ng != ng or not np.allclose(lib.x_fracs, np.asarray(X_FRACS), atol=1e-12):
                raise ValueError(f"template {name} does not match Ng={ng} / the station layout")
            out[name] = lib
    return out


def label_modes(solved: Solved, templates: TemplateLibrary) -> MatchResult:
    """B1/T1/B2/T2 among the solved modes; raises ValueError if any label has MAC < MAC_GATE."""
    res = match_modes(solved.phi_u3, solved.phi_u2, solved.freq_hz, templates, model=solved.model,
                      phi_full=solved.phi_full, participation=solved.participation, config=MatchConfig(min_mac=MAC_GATE))
    if np.any(res.selected_mac < MAC_GATE):
        raise ValueError(f"selected-mode MAC below {MAC_GATE}: {np.round(res.selected_mac, 4).tolist()}")
    return res


def sign_flips(phi_stations: np.ndarray, template: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """+1/-1 per mode so that each mode correlates positively with its template.

    If the dot product is ~0, the template's largest station decides instead.
    """
    cand, ref = np.asarray(phi_stations, dtype=float), np.asarray(template, dtype=float)
    dots = np.sum(cand * ref, axis=0)
    anchor = np.argmax(np.abs(ref), axis=0)
    flips = np.ones(cand.shape[1], dtype=float)
    for j in range(cand.shape[1]):
        if abs(float(dots[j])) > eps:
            if float(dots[j]) < 0.0:
                flips[j] = -1.0
        else:
            r, c = float(ref[anchor[j], j]), float(cand[anchor[j], j])
            if abs(r) > eps and abs(c) > eps and c * r < 0.0:
                flips[j] = -1.0
    return flips


def make_sample(sample_id: int, sampled: dict[str, Any], solved: Solved, match: MatchResult,
                templates: TemplateLibrary) -> dict[str, np.ndarray]:
    """Network inputs and targets of one deck."""
    sel = np.asarray(match.selected_idx, dtype=int)
    graph = projected_graph(solved.model)
    rows = np.asarray(graph["graph_node_ids"], dtype=np.int64) - 1          # FE node -> row

    a3 = vertical_acceleration(solved.model, solved.phi_full, solved.freq_hz, solved.M_full, bandpass_hz=BANDPASS_HZ)
    a3_graph = np.asarray(a3, dtype=float)[rows, :]
    f, psd = welch_psd(a3_graph, FS_HZ)
    x_psd = normalize_rows_by_max_abs(psd)[:, :N_FREQ_BINS]               # normalised over the full spectrum
    coh = edge_coherency(graph["edge_index"], a3_graph, FS_HZ)[:, :N_FREQ_BINS]

    n_nodes = int(solved.model["nodes"].shape[0])
    phi_u3_all = fe.mode_rows(solved.phi_full[:, sel], np.arange(1, n_nodes + 1), 3)
    flips = sign_flips(solved.phi_u3[:, sel], templates.primary_u3).reshape(1, -1)
    y_signed = normalize_cols_by_max_abs(phi_u3_all * flips)[rows, :]

    return {
        "sample_id": np.asarray(int(sample_id), dtype=np.int64),
        "Ng": np.asarray(int(sampled["Ng"]), dtype=np.int64),
        "Nx": np.asarray(int(sampled["Nx"]), dtype=np.int64),
        "params_json": np.asarray(to_json(sampled)),
        "graph_node_ids": np.asarray(graph["graph_node_ids"], dtype=np.int64),
        "coords_xy_mm": np.asarray(graph["coords_xy_mm"], dtype=float),
        "edge_index": np.asarray(graph["edge_index"], dtype=np.int64),
        "edge_length_mm": np.asarray(graph["edge_length_mm"], dtype=float),
        "edge_theta_rad": np.asarray(graph["edge_theta_rad"], dtype=float),
        "freq_bins_hz": np.asarray(f[:N_FREQ_BINS], dtype=float),
        "x_psd": np.asarray(x_psd, dtype=np.float32),
        "edge_attr_cpsd_full": interleave_re_im(np.real(coh), np.imag(coh)),
        "target_mode_names": np.asarray(TARGETS, dtype="<U2"),
        "y_mode_shape_signed": np.asarray(y_signed, dtype=float),
        "y_freq_hz": np.asarray(solved.freq_hz[sel], dtype=np.float32),
        "matched_mode_indices": sel.astype(np.int64),
        "matched_mac": np.asarray(match.selected_mac, dtype=float),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def to_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default)


def sample_filename(sample_id: int, ng: int) -> str:
    return f"sample_{sample_id:04d}_Ng{ng}.npz"


def _summary_row(sample_id: int, sampled: dict[str, Any], sample: dict[str, np.ndarray]) -> dict[str, Any]:
    b = sampled["balcony"]
    row = {"sample_id": sample_id, "file": sample_filename(sample_id, int(sampled["Ng"])),
           "Ng": int(sampled["Ng"]), "L_mm": sampled["L_mm"], "Nx": sampled["Nx"], "dy_mm": sampled["dy_mm"],
           "diag_pattern": sampled["diag_pattern"], "E": sampled["E"], "rho": sampled["rho"],
           "kx_kN_per_m": sampled["spring"]["kx_kN_per_m"], "kz_kN_per_m": sampled["spring"]["kz_kN_per_m"],
           "bottom_x_frac": sampled["bottom_beam"]["x_frac"], "bottom_overhang_mm": sampled["bottom_beam"]["overhang_mm"],
           "balcony": int(bool(b["enable"])), "balcony_side": b["side"], "balcony_nBeams": b["nBeams"],
           "nsm_tonne": sampled["nsm"]["total_mass_tonne"], "n_graph_nodes": int(sample["graph_node_ids"].size)}
    for k, name in enumerate(TARGETS):
        row[f"freq_{name}_hz"] = float(sample["y_freq_hz"][k])
        row[f"mac_{name}"] = float(sample["matched_mac"][k])
        row[f"fe_mode_{name}"] = int(sample["matched_mode_indices"][k])
    return row


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def generate(out_dir: str | Path, n_per_ng: int, *, ngs: tuple[int, ...] = NG_CHOICES, seed: int = SEED,
             labels_only: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Generate ``n_per_ng`` accepted decks for each girder count in ``ngs``.

    Sample ids run over the full population (Ng = 2 first), so generating one
    girder count on its own gives the same ids and decks as generating them all.
    ``labels_only`` skips the response simulation and writes only summary.csv
    (to check the structure list and the labels quickly).  Runs for a subset of
    girder counts name their summary and log after it (``summary_Ng3.csv``), so
    several can write into the same folder at once.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = run_tag(ngs)
    templates = load_all_templates()
    streams = dict(zip(NG_CHOICES, np.random.SeedSequence(seed).spawn(len(NG_CHOICES))))
    summary: list[dict[str, Any]] = []
    counts: dict[int, dict[str, int]] = {}
    t0 = time.time()
    with (out_dir / f"failures{tag}.log").open("w", encoding="utf-8") as log:
        for ng in ngs:
            rng = np.random.default_rng(streams[ng])
            sample_id = NG_CHOICES.index(ng) * n_per_ng
            c = counts[ng] = {"accepted": 0, "rejected": 0, "attempts": 0}
            while c["accepted"] < n_per_ng:
                c["attempts"] += 1
                if c["attempts"] > MAX_ATTEMPTS_PER_NG:
                    raise RuntimeError(f"Ng={ng}: more than {MAX_ATTEMPTS_PER_NG} attempts")
                sampled = None
                try:
                    sampled = sample_params(ng, rng)
                    if not is_valid(sampled):
                        c["rejected"] += 1
                        continue
                    solved = solve(solver_params(sampled))
                    lib = templates[template_name(ng, bool(sampled["balcony"]["enable"]))]
                    match = label_modes(solved, lib)
                    if labels_only:
                        sample = {"graph_node_ids": projected_graph(solved.model)["graph_node_ids"],
                                  "y_freq_hz": np.asarray(solved.freq_hz[match.selected_idx], dtype=np.float32),
                                  "matched_mac": match.selected_mac, "matched_mode_indices": match.selected_idx}
                    else:
                        sample = make_sample(sample_id, sampled, solved, match, lib)
                        np.savez(out_dir / sample_filename(sample_id, ng), **sample)
                except Exception as exc:          # a rejected deck: log it and draw the next one
                    c["rejected"] += 1
                    log.write(f"Ng={ng} attempt={c['attempts']} {type(exc).__name__}: {exc}\n")
                    log.write(f"sampled={to_json(sampled) if sampled is not None else '<none>'}\n---\n")
                    continue
                summary.append({"params_json": to_json(sampled), **_summary_row(sample_id, sampled, sample)})
                c["accepted"] += 1
                sample_id += 1
                if verbose and (c["accepted"] % 25 == 0 or c["accepted"] == n_per_ng):
                    print(f"[Ng={ng}] {c['accepted']}/{n_per_ng} accepted, {c['rejected']} rejected, "
                          f"{time.time() - t0:.0f} s", flush=True)
    with (out_dir / f"summary{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    return {"counts": counts, "seconds": round(time.time() - t0, 1)}


def run_tag(ngs: tuple[int, ...]) -> str:
    """'' for the full population, '_Ng3' (or '_Ng34', ...) for a subset of girder counts."""
    return "" if tuple(sorted(ngs)) == NG_CHOICES else "_Ng" + "".join(str(g) for g in sorted(ngs))
