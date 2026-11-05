"""Parametric 3-D beam finite-element model of a ferry linkspan deck.

The deck is a grid of steel beams in plan: ``Ng`` longitudinal girders at a
spacing ``dy_mm``, transverse cross beams, optional diagonal bracing, a bottom
(support) beam under the deck, a front beam at the free end and optional
balconies on the outer girders.  Cross beams, bracing, the bottom beam, the
front beam and the balcony beams sit at vertical offsets from the girder plane;
each offset node is tied to its girder node by a rigid link, imposed as a
multi-point constraint (MPC) by condensation.

Supports (the shore-side hinge at x = 0 and the two ends of the bottom beam):
  * every girder node at x = 0 is fixed in uy and uz and has a longitudinal
    spring kx;
  * the two ends of the bottom beam have vertical springs kz;
  * a small rotational spring about z at one hinge node removes the in-plane
    rigid-body rotation.

Elements are 3-D Timoshenko beams (consistent Euler-Bernoulli mass), with end
releases for the diagonal braces.  Non-structural mass (surfacing, equipment) is
smeared per unit length over the girders, cross beams and balcony beams.

Units: N, mm, tonne, s (so E is in MPa and rho in tonne/mm^3).
Degrees of freedom per node, in order: ux, uy, uz, rx, ry, rz.
Node ids are 1-based throughout, as in the element table.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh

# Element groups (column 3 of the element table).
G_GIRDER, G_CROSS, G_DIAG, G_RIGID, G_BOTTOM, G_FRONT, G_BALCONY, G_EDGE, G_BBRACE = range(1, 10)
EIGSH_START_SEED = 0


@dataclass
class Section:
    name: str
    A: float
    Iy: float
    Iz: float
    J: float
    ky: float
    kz: float


@dataclass
class Material:
    E: float
    G: float
    nu: float
    rho: float


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def default_params() -> dict:
    """Parameters of a mid-range deck, with standard steel properties.

    The synthetic population overrides the geometry, the material, the support
    springs and the non-structural mass of every deck (``population.solver_params``);
    a measured deck supplies its own geometry.  The cross sections, offsets, end
    releases and the rotational stabiliser below are kept for every structure.
    """
    p: dict = {}
    p["L_mm"] = 18000
    p["Nx"] = 20
    p["Ng"] = 4
    p["dy_mm"] = 2500
    p["crossbeam_include_ends"] = False
    p["x_crossbeam_mm_list"] = None          # list of x [mm] to override uniform cross-beam spacing
    p["diag_pattern"] = "None"               # None | global_x | staggered_x | single_partial_x

    p["E"] = 210000.0
    p["nu"] = 0.30
    p["rho"] = 7.85e-9
    p["G"] = p["E"] / (2 * (1 + p["nu"]))

    p["sec"] = {
        "girder": dict(A=2.0e4, Iy=1.6e9, Iz=5.0e8, J=3.0e7, ky=1.2, kz=1.2),
        "cross": dict(A=8.0e3, Iy=1.2e8, Iz=4.0e7, J=5.0e6, ky=1.2, kz=1.2),
        "diag": dict(A=4.5e3, Iy=1.0e7, Iz=1.0e7, J=2.0e6, ky=1.1, kz=1.1),
        "bottom": dict(A=1.35e4, Iy=7.0e8, Iz=2.5e8, J=2.0e7, ky=1.2, kz=1.2),
    }

    b_x, h_z = 890.0, 15.0                   # front plate
    p["sec"]["front"] = dict(
        A=b_x * h_z,
        Iy=(h_z * b_x**3) / 12,
        Iz=(b_x * h_z**3) / 12,
        J=(1 / 3) * b_x * h_z**3,
        ky=5 / 6,
        kz=5 / 6,
    )

    h, b, tw, tf = 380.0, 150.0, 11.7, 17.5  # balcony beams: I-section
    hw = h - 2 * tf
    p["sec"]["balcony"] = dict(
        A=2 * (b * tf) + (tw * hw),
        Iz=2 * ((b * tf**3) / 12 + (b * tf) * ((h / 2 - tf / 2) ** 2)) + (tw * hw**3) / 12,
        Iy=2 * ((tf * b**3) / 12) + (hw * tw**3) / 12,
        J=(1 / 3) * (2 * b * tf**3 + hw * tw**3),
        ky=5 / 6,
        kz=5 / 6,
    )

    t, h = 15.0, 380.0                       # balcony edge beams and braces: flat plate
    p["sec"]["edge"] = dict(
        A=t * h,
        Iy=(t * h**3) / 12,
        Iz=(h * t**3) / 12,
        J=(1 / 3) * h * t**3,
        ky=5 / 6,
        kz=5 / 6,
    )

    # vertical offsets [mm] of each member family from the girder plane
    p["offset"] = dict(cross_z=100.0, diag_z=-127.5, bottom_z=-472.0, front_z=115.0, balcony_z=70.0)
    p["diag_release"] = True                 # braces are pin-ended
    p["spring"] = dict(kx_kN_per_m=2000, kz_kN_per_m=20000)      # hinge (x) and bottom-beam (z) supports
    p["spring"]["kx_N_per_mm"] = kN_per_m_to_N_per_mm(p["spring"]["kx_kN_per_m"])
    p["spring"]["kz_N_per_mm"] = kN_per_m_to_N_per_mm(p["spring"]["kz_kN_per_m"])

    p["bottom_beam"] = dict(x_frac=0.62, overhang_mm=550)
    p["stabilize"] = dict(enable=True, k_rz_Nmm_per_rad=1e6, ref_girder_j=0, ref_x=0.0)
    p["balcony"] = dict(
        enable=False,
        side="both",                         # left | right | both
        x_center_frac=0.88,
        nBeams=2,
        dx_mm=750,
        length_mm=2500,
        edge_beam_enable=True,
        brace_enable=True,
    )
    p["nsm"] = dict(
        enable=True,
        total_mass_tonne=30,
        to_girders=True,
        to_cross=True,
        to_balcony=True,
        to_edge=False,
        to_brace=False,
        to_front=False,
        to_bottom=False,
        to_diagonals=False,
    )
    p["nmodes"] = 20
    p["geom_key_mm"] = 1e-6
    return p


def kN_per_m_to_N_per_mm(k_kN_per_m: float) -> float:
    return k_kN_per_m                        # 1 kN/m == 1 N/mm


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update ``base`` in place with ``overrides``."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def params_from_overrides(overrides: dict) -> dict:
    """Default parameters updated with a (possibly nested) dict of overrides."""
    if not overrides:
        raise ValueError("empty parameter overrides: the defaults describe no particular structure")
    params = default_params()
    deep_update(params, deepcopy(overrides))
    return params


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def _unique_tol(xs: np.ndarray, tol: float) -> np.ndarray:
    xs = np.sort(np.ravel(xs).astype(float))
    if xs.size == 0:
        return xs
    keep = np.ones(xs.size, dtype=bool)
    for k in range(1, xs.size):
        if abs(xs[k] - xs[k - 1]) < tol:
            keep[k] = False
    return xs[keep]


def _interp_x_on_line(x0: float, y0: float, x1: float, y1: float, yq: float) -> float:
    t = (yq - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)


def _node_key(j_girder: int, x: float, key_scale_mm: float) -> int:
    return int(j_girder) * 10**13 + int(round(x / key_scale_mm))


def _section(s: dict, name: str) -> Section:
    return Section(name=name, A=s["A"], Iy=s["Iy"], Iz=s["Iz"], J=s["J"], ky=s["ky"], kz=s["kz"])


def _release_code(rx1: bool, ry1: bool, rz1: bool, rx2: bool, ry2: bool, rz2: bool) -> int:
    """Bit mask of released rotations at the two element ends."""
    rel = 0
    for k, bit in enumerate([rx1, ry1, rz1, rx2, ry2, rz2]):
        if bit:
            rel |= 1 << k
    return rel


_NSM_FLAGS = (("to_girders", G_GIRDER), ("to_cross", G_CROSS), ("to_balcony", G_BALCONY),
              ("to_edge", G_EDGE), ("to_brace", G_BBRACE), ("to_front", G_FRONT),
              ("to_bottom", G_BOTTOM), ("to_diagonals", G_DIAG))


def _nsm_distribution(params: dict, nodes: np.ndarray, elems: np.ndarray) -> dict:
    """Spread the non-structural mass uniformly over the length of the chosen groups."""
    nsm = dict(enable=bool(params.get("nsm", {}).get("enable", False)), total_mass_tonne=0.0,
               m_add_per_length_tonne_per_mm=0.0, groups=np.zeros(20, dtype=bool), total_target_length_mm=0.0)
    if not nsm["enable"]:
        return nsm
    npm = params["nsm"]
    nsm["total_mass_tonne"] = npm["total_mass_tonne"]
    gmask = np.zeros(20, dtype=bool)
    for key, g in _NSM_FLAGS:
        if npm.get(key, False):
            gmask[g - 1] = True

    ltot = 0.0
    for e in elems:
        grp, mat_id = int(e[3]), int(e[5])
        if mat_id != 1 or grp < 1 or grp > gmask.size or not gmask[grp - 1]:
            continue
        n1, n2 = int(e[1]) - 1, int(e[2]) - 1
        ltot += np.linalg.norm(nodes[n2, 1:4] - nodes[n1, 1:4])
    if ltot <= 0:
        return nsm
    nsm["groups"] = gmask
    nsm["total_target_length_mm"] = ltot
    nsm["m_add_per_length_tonne_per_mm"] = nsm["total_mass_tonne"] / ltot
    return nsm


def _normalize_diag_pattern(diag_pattern: Optional[object]) -> Optional[str]:
    if diag_pattern is None or str(diag_pattern).strip().lower() in ("none", ""):
        return None
    alias = {"globalX_attach": "global_x", "global_x": "global_x",
             "staggered_multi_x": "staggered_multi_x", "staggered_x": "staggered_multi_x",
             "single_partial_x_ng4": "single_partial_x_ng4", "single_partial_x": "single_partial_x_ng4"}
    pattern = str(diag_pattern)
    if pattern not in alias:
        raise ValueError(f"unknown diag_pattern {pattern!r}; use None, global_x, staggered_x or single_partial_x")
    return alias[pattern]


def _diag_lines(params: dict, x_cb: np.ndarray, x_bottom: float, x0: float, x1: float,
                Ng: int) -> list[tuple[int, float, int, float]]:
    """Diagonal brace lines as (girder_start, x_start, girder_end, x_end)."""
    pattern = _normalize_diag_pattern(params.get("diag_pattern", "global_x"))
    if pattern is None:
        return []
    if x_cb.size == 0:
        raise ValueError("no cross-beam positions available for diagonal bracing")
    d = np.abs(x_cb - x_bottom)
    x_near, x_far = float(x_cb[int(np.argmin(d))]), float(x_cb[int(np.argmax(d))])
    x_last = float(np.max(x_cb))
    j_top, j_bot, j_mid = Ng - 1, 0, Ng // 2

    if pattern == "global_x":                # one X over the whole deck
        return [(j_bot, x0, j_top, x1), (j_bot, x1, j_top, x0)]
    if pattern == "staggered_multi_x":
        lines = [(j_top, x_far, j_bot, x_near), (j_top, x_near, j_bot, x_far)]
        if Ng % 2 == 1:                      # odd Ng: two extra branches meeting at the middle girder
            lines += [(j_top, x_near, j_mid, x_last), (j_bot, x_near, j_mid, x_last)]
        else:                                # even Ng: a second full X
            lines += [(j_top, x_near, j_bot, x_last), (j_bot, x_near, j_top, x_last)]
        return lines
    if pattern == "single_partial_x_ng4":    # one brace from the last girder to the second
        if Ng != 4:
            raise ValueError("single_partial_x is defined only for Ng == 4")
        return [(j_top, x_last, Ng - 3, x_near)]
    raise ValueError(pattern)


def build_model(params: dict) -> dict:
    """Nodes, elements, supports, springs and rigid links of one deck.

    Returns a dict with ``nodes`` (N x 4: id, x, y, z), ``elems`` (E x 7: id, n1,
    n2, group, section, material, release code), ``bc``, ``springs``,
    ``mpc_pairs`` (master, slave) and the section/material tables.
    """
    L, Nx, Ng, dy = params["L_mm"], params["Nx"], params["Ng"], params["dy_mm"]
    dx = L / Nx
    x_base = np.arange(Nx + 1) * dx
    i0, i1 = (0, Nx) if params["crossbeam_include_ends"] else (1, Nx - 1)
    x0, x1 = i0 * dx, i1 * dx
    x_bottom = params["bottom_beam"]["x_frac"] * L

    b = params.get("balcony", {})
    x_balcony = np.array([])
    if b.get("enable", False):
        xc, nb, dxB = b["x_center_frac"] * L, b["nBeams"], b["dx_mm"]
        if nb == 1:
            x_balcony = np.array([xc])
        elif nb > 1:
            x_balcony = xc + (np.arange(nb) - (nb - 1) / 2) * dxB
    x_brace_anchor = np.array([])
    if b.get("enable", False) and b.get("brace_enable", False) and x_balcony.size >= 1:
        x_brace_anchor = np.array([x_balcony.min() - b["dx_mm"]])

    if params.get("x_crossbeam_mm_list") is not None:
        x_cb = _unique_tol(np.asarray(params["x_crossbeam_mm_list"], dtype=float), 1e-9)
    else:
        x_cb = x_base if params["crossbeam_include_ends"] else x_base[1:-1]

    # girder stations: grid, cross beams, bottom beam, balcony beams, brace anchors ...
    x_list = [_unique_tol(np.r_[x_base, x_cb, x_bottom, x_balcony, x_brace_anchor], 1e-9) for _ in range(Ng)]
    # ... and every point where a diagonal crosses a girder
    single_partial = _normalize_diag_pattern(params.get("diag_pattern", "global_x")) == "single_partial_x_ng4"
    diag_lines = _diag_lines(params, x_cb, x_bottom, x0, x1, Ng)
    for j0, xs, j1, xe in diag_lines:
        if j0 == j1:
            continue
        y0, y1 = j0 * dy, j1 * dy
        j_ids = [j0, j1] if single_partial else list(range(min(j0, j1), max(j0, j1) + 1))
        for j in j_ids:
            x_list[j] = _unique_tol(np.r_[x_list[j], _interp_x_on_line(xs, y0, xe, y1, j * dy)], 1e-9)

    key_scale = max(params["geom_key_mm"], 1e-6)
    node_map: dict[int, int] = {}
    nodes: list[list[float]] = []
    for j in range(Ng):
        for x in x_list[j]:
            nid = len(nodes) + 1
            nodes.append([nid, float(x), float(j * dy), 0.0])
            node_map[_node_key(j, float(x), key_scale)] = nid

    def girder_node(j: int, x: float) -> int:
        return node_map[_node_key(j, float(x), key_scale)]

    sections = {
        1: _section(params["sec"]["girder"], "girder"),
        2: _section(params["sec"]["cross"], "cross"),
        3: _section(params["sec"]["diag"], "diag"),
        5: _section(params["sec"]["bottom"], "bottom"),
        6: _section(params["sec"]["front"], "front"),
        7: _section(params["sec"]["balcony"], "balcony"),
        8: _section(params["sec"]["edge"], "edge"),
        9: _section(params["sec"]["edge"], "balcony_brace"),
    }
    materials = {1: Material(params["E"], params["G"], params["nu"], params["rho"]),
                 2: Material(params["E"], params["G"], params["nu"], 0.0)}

    elems: list[list[float]] = []
    mpcs: list[list[int]] = []
    caches: dict[str, dict[int, int]] = {k: {} for k in ["cross", "diag", "bottom", "front", "balcony"]}
    nodes_arr = np.array(nodes, dtype=float)
    extra_nodes: list[list[float]] = []

    def add_elem(n1: int, n2: int, grp: int, sec: int, mat: int, rel: int = 0) -> None:
        elems.append([len(elems) + 1, n1, n2, grp, sec, mat, rel])

    def new_node(x: float, y: float, z: float) -> int:
        nid = nodes_arr.shape[0] + len(extra_nodes) + 1
        extra_nodes.append([nid, x, y, z])
        return nid

    def offset_node(base_nid: int, zoff: float, cache: dict) -> int:
        """Node vertically offset from a girder node, rigidly linked to it (one per family)."""
        if base_nid in cache:
            return cache[base_nid]
        xb, yb, zb = nodes_arr[base_nid - 1, 1:4]
        off_id = new_node(xb, yb, zb + zoff)
        cache[base_nid] = off_id
        mpcs.append([base_nid, off_id])
        return off_id

    off = params["offset"]
    for j in range(Ng):                      # girders
        xs = x_list[j]
        for k in range(xs.size - 1):
            add_elem(girder_node(j, xs[k]), girder_node(j, xs[k + 1]), G_GIRDER, 1, 1)

    for xi in x_cb:                          # cross beams
        for j in range(Ng - 1):
            nA = offset_node(girder_node(j, xi), off["cross_z"], caches["cross"])
            nB = offset_node(girder_node(j + 1, xi), off["cross_z"], caches["cross"])
            add_elem(nA, nB, G_CROSS, 2, 1)

    rel_diag = _release_code(True, True, True, True, True, True) if params["diag_release"] else 0
    for j0, xs, j1, xe in diag_lines:        # diagonal braces, split at every girder they cross
        if j0 == j1:
            continue
        y0, y1 = j0 * dy, j1 * dy
        j_ids = [j0, j1] if single_partial else list(range(min(j0, j1), max(j0, j1) + 1))
        pts = np.array([(_interp_x_on_line(xs, y0, xe, y1, jg * dy), jg * dy) for jg in j_ids], dtype=float)
        off_nodes = []
        for xk, yk in pts[np.argsort(pts[:, 1])]:
            base_n = girder_node(int(round(yk / dy)), xk)
            off_nodes.append(caches["diag"][base_n] if base_n in caches["diag"]
                             else offset_node(base_n, off["diag_z"], caches["diag"]))
        for k in range(len(off_nodes) - 1):
            add_elem(off_nodes[k], off_nodes[k + 1], G_DIAG, 3, 1, rel_diag)

    # bottom (support) beam, overhanging both outer girders
    y_outer_min, y_outer_max = 0.0, (Ng - 1) * dy
    yL = y_outer_min - params["bottom_beam"]["overhang_mm"]
    yR = y_outer_max + params["bottom_beam"]["overhang_mm"]
    bottom_node_ids: list[int] = []
    for yk in np.r_[yL, np.arange(Ng) * dy, yR]:
        if abs(yk - yL) < 1e-9 or abs(yk - yR) < 1e-9:
            bottom_node_ids.append(new_node(x_bottom, yk, off["bottom_z"]))
        else:
            bottom_node_ids.append(offset_node(girder_node(int(round(yk / dy)), x_bottom), off["bottom_z"],
                                               caches["bottom"]))
    for k in range(len(bottom_node_ids) - 1):
        add_elem(bottom_node_ids[k], bottom_node_ids[k + 1], G_BOTTOM, 5, 1)

    front_nodes = [offset_node(girder_node(j, L), off["front_z"], caches["front"]) for j in range(Ng)]
    for j in range(Ng - 1):                  # front beam at the free end
        add_elem(front_nodes[j], front_nodes[j + 1], G_FRONT, 6, 1)

    if b.get("enable", False) and x_balcony.size:   # balconies: cantilevers off the outer girders
        add_left = b["side"] in ("left", "both")
        add_right = b["side"] in ("right", "both")
        len_left = float(b.get("length_left_mm", b.get("length_mm", 0.0)))
        len_right = float(b.get("length_right_mm", b.get("length_mm", 0.0)))
        left_out, right_out = [], []
        for xi in x_balcony:
            if add_left:
                n_in = offset_node(girder_node(0, xi), off["balcony_z"], caches["balcony"])
                n_out = new_node(xi, y_outer_min - len_left, off["balcony_z"])
                add_elem(n_in, n_out, G_BALCONY, 7, 1)
                left_out.append((xi, n_out))
            if add_right:
                n_in = offset_node(girder_node(Ng - 1, xi), off["balcony_z"], caches["balcony"])
                n_out = new_node(xi, y_outer_max + len_right, off["balcony_z"])
                add_elem(n_in, n_out, G_BALCONY, 7, 1)
                right_out.append((xi, n_out))
        if b.get("edge_beam_enable", False):  # edge beam joining the balcony tips
            for side in (sorted(left_out), sorted(right_out)):
                for k in range(len(side) - 1):
                    add_elem(side[k][1], side[k + 1][1], G_EDGE, 8, 1)
        if b.get("brace_enable", False):      # brace from the first balcony tip back to the girder
            xmin, x_target = float(x_balcony.min()), float(x_balcony.min() - b["dx_mm"])
            for lst, j in [(left_out, 0), (right_out, Ng - 1)]:
                matches = [n for x, n in lst if abs(x - xmin) < 1e-6]
                if matches:
                    n_in = offset_node(girder_node(j, x_target), off["balcony_z"], caches["balcony"])
                    add_elem(n_in, matches[0], G_BBRACE, 9, 1)

    if extra_nodes:
        nodes_arr = np.vstack([nodes_arr, np.array(extra_nodes, dtype=float)])
    n_nodes = nodes_arr.shape[0]

    bc = []                                  # hinge: uy, uz fixed at every girder end at x = 0
    for j in range(Ng):
        n0 = girder_node(j, 0.0)
        bc += [[n0, 2, 0], [n0, 3, 0]]
    springs = [[girder_node(j, 0.0), 1, params["spring"]["kx_N_per_mm"]] for j in range(Ng)]
    springs += [[bottom_node_ids[0], 3, params["spring"]["kz_N_per_mm"]],
                [bottom_node_ids[-1], 3, params["spring"]["kz_N_per_mm"]]]
    if params.get("stabilize", {}).get("enable", False):
        nref = girder_node(params["stabilize"]["ref_girder_j"], params["stabilize"]["ref_x"])
        springs.append([nref, 6, params["stabilize"]["k_rz_Nmm_per_rad"]])

    elems_arr = np.array(elems, dtype=float)
    return dict(
        nodes=nodes_arr,
        elems=elems_arr,
        Ng=int(Ng), Nx=int(Nx), L_mm=float(L), dy_mm=float(dy),
        sections=sections,
        materials=materials,
        bc=np.array(bc, dtype=float),
        ndof=6 * n_nodes,
        springs=np.array(springs, dtype=float),
        nsm=_nsm_distribution(params, nodes_arr, elems_arr),
        bottom_node_ids=np.array(bottom_node_ids, dtype=int),
        mpc_pairs=np.array(mpcs, dtype=int) if len(mpcs) else np.zeros((0, 2), dtype=int),
        vref_primary=np.array([0.0, 0.0, 1.0]),
        vref_fallback=np.array([0.0, 1.0, 0.0]),
    )


# ---------------------------------------------------------------------------
# Element matrices and assembly
# ---------------------------------------------------------------------------
def _timoshenko_bending_2d(EI: float, L: float, phi: float) -> np.ndarray:
    fac = EI / (L**3 * (1 + phi))
    return fac * np.array([
        [12, 6 * L, -12, 6 * L],
        [6 * L, (4 + phi) * L**2, -6 * L, (2 - phi) * L**2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, (2 - phi) * L**2, -6 * L, (4 + phi) * L**2],
    ])


def _bending_mass_2d(m: float, L: float) -> np.ndarray:
    return (m / 420) * np.array([
        [156, 22 * L, 54, -13 * L],
        [22 * L, 4 * L**2, 13 * L, -3 * L**2],
        [54, 13 * L, 156, -22 * L],
        [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
    ])


def beam_KM_local(sec: Section, mat: Material, L: float, m_add_per_len: float) -> tuple[np.ndarray, np.ndarray]:
    """12 x 12 local stiffness (Timoshenko) and consistent mass of a 3-D beam."""
    E, G, rho = mat.E, mat.G, mat.rho
    A, Iy, Iz, J, ky, kz = sec.A, sec.Iy, sec.Iz, sec.J, sec.ky, sec.kz
    ke = np.zeros((12, 12))
    me = np.zeros((12, 12))
    ke[np.ix_([0, 6], [0, 6])] += np.array([[1, -1], [-1, 1]]) * (E * A / L)       # axial
    ke[np.ix_([3, 9], [3, 9])] += np.array([[1, -1], [-1, 1]]) * (G * J / L)       # torsion
    idx = [1, 5, 7, 11]                                                             # bending in x-y
    ke[np.ix_(idx, idx)] += _timoshenko_bending_2d(E * Iz, L, (12 * E * Iz) / (ky * G * A * L**2))
    idx = [2, 4, 8, 10]                                                             # bending in x-z
    ke[np.ix_(idx, idx)] += _timoshenko_bending_2d(E * Iy, L, (12 * E * Iy) / (kz * G * A * L**2))
    if rho <= 0 and m_add_per_len <= 0:
        return ke, me
    m = (rho * A + max(0.0, m_add_per_len)) * L
    me[np.ix_([0, 6], [0, 6])] += (m / 6) * np.array([[2, 1], [1, 2]])
    me[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += _bending_mass_2d(m, L)
    me[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += _bending_mass_2d(m, L)
    if rho > 0:
        Jm = rho * (Iy + Iz) * L
        me[np.ix_([3, 9], [3, 9])] += (Jm / 6) * np.array([[2, 1], [1, 2]])
    return ke, me


def _apply_end_releases(K: np.ndarray, M: np.ndarray, rel_code: int) -> tuple[np.ndarray, np.ndarray]:
    """Statically condense the released end rotations out of K and M."""
    rel = np.zeros(12, dtype=bool)
    for k, dof in enumerate([3, 4, 5, 9, 10, 11]):
        if rel_code & (1 << k):
            rel[dof] = True
    kept, rels = np.where(~rel)[0], np.where(rel)[0]
    if rels.size == 0:
        return K, M
    Kkk, Mkk = K[np.ix_(kept, kept)], M[np.ix_(kept, kept)]
    Kkr, Krk, Krr = K[np.ix_(kept, rels)], K[np.ix_(rels, kept)], K[np.ix_(rels, rels)]
    Mkr, Mrk, Mrr = M[np.ix_(kept, rels)], M[np.ix_(rels, kept)], M[np.ix_(rels, rels)]
    if np.linalg.cond(Krr) > 1e12:
        Krr = Krr + (1e-9 * max(1.0, np.linalg.norm(Krr, "fro"))) * np.eye(Krr.shape[0])
    Kcond = Kkk - Kkr @ np.linalg.solve(Krr, Krk)
    if Mrr.size == 0 or np.all(np.abs(Mrr) < 1e-30):
        Mcond = Mkk
    else:
        if np.linalg.cond(Mrr) > 1e12:
            Mrr = Mrr + (1e-12 * max(1.0, np.linalg.norm(Mrr, "fro"))) * np.eye(Mrr.shape[0])
        Mcond = Mkk - Mkr @ np.linalg.solve(Mrr, Mrk)
    Kc, Mc = np.zeros((12, 12)), np.zeros((12, 12))
    Kc[np.ix_(kept, kept)] = Kcond
    Mc[np.ix_(kept, kept)] = Mcond
    return Kc, Mc


def _local_axes(d: np.ndarray, vref: np.ndarray, vref_fallback: np.ndarray) -> np.ndarray:
    """Rotation matrix [x y z] of an element along ``d``, with y = vref x d."""
    xhat = d / np.linalg.norm(d)
    ytmp = np.cross(vref.reshape(-1), xhat)
    if np.linalg.norm(ytmp) < 1e-12:
        ytmp = np.cross(vref_fallback.reshape(-1), xhat)
        if np.linalg.norm(ytmp) < 1e-12:
            raise ValueError("failed to define element local axes")
    yhat = ytmp / np.linalg.norm(ytmp)
    zhat = np.cross(xhat, yhat)
    zhat = zhat / np.linalg.norm(zhat)
    return np.column_stack([xhat, yhat, zhat])


def _element_vref(grp: int, model: dict) -> np.ndarray:
    if grp in (G_GIRDER, G_CROSS, G_BBRACE):
        return np.array([0.0, 0.0, 1.0])
    if grp in (G_BALCONY, G_EDGE):
        return np.array([1.0, 0.0, 0.0])
    return model["vref_primary"]


def _T_beam(R: np.ndarray) -> np.ndarray:
    Z = np.zeros((3, 3))
    Tn = np.block([[R, Z], [Z, R]])
    return np.block([[Tn, np.zeros((6, 6))], [np.zeros((6, 6)), Tn]])


def assemble_KM(model: dict) -> tuple[csc_matrix, csc_matrix]:
    """Global stiffness and mass matrices (all nodes, before supports and rigid links)."""
    nodes, elems, ndof = model["nodes"], model["elems"], model["ndof"]
    rows: list[int] = []
    cols: list[int] = []
    valsK: list[float] = []
    valsM: list[float] = []
    nsm_enable = model.get("nsm", {}).get("enable", False)
    m_add = model.get("nsm", {}).get("m_add_per_length_tonne_per_mm", 0.0)
    gmask = model.get("nsm", {}).get("groups", np.zeros(20, dtype=bool))

    for e in elems:
        n1, n2 = int(e[1]), int(e[2])
        grp, sec_id, mat_id, rel_code = int(e[3]), int(e[4]), int(e[5]), int(e[6])
        sec, mat = model["sections"][sec_id], model["materials"][mat_id]
        d = nodes[n2 - 1, 1:4] - nodes[n1 - 1, 1:4]
        Le = np.linalg.norm(d)
        if Le <= 1e-12:
            continue
        R = _local_axes(d, _element_vref(grp, model), model["vref_fallback"])
        madd = m_add if (nsm_enable and mat_id == 1 and 1 <= grp <= gmask.size and gmask[grp - 1]) else 0.0
        keL, meL = beam_KM_local(sec, mat, Le, madd)
        if rel_code != 0:
            keL, meL = _apply_end_releases(keL, meL, rel_code)
        T = _T_beam(R)
        keG, meG = T.T @ keL @ T, T.T @ meL @ T
        dofs = np.r_[np.arange(6 * (n1 - 1), 6 * n1), np.arange(6 * (n2 - 1), 6 * n2)]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        valsK.extend(keG.ravel().tolist())
        valsM.extend(meG.ravel().tolist())

    K = coo_matrix((valsK, (rows, cols)), shape=(ndof, ndof)).tocsc()
    M = coo_matrix((valsM, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return ((K + K.T) * 0.5).tocsc(), ((M + M.T) * 0.5).tocsc()


def add_springs(K: csc_matrix, model: dict) -> csc_matrix:
    K = K.tolil(copy=True)
    for nid, dof, kval in model.get("springs", np.zeros((0, 3))):
        g = gdof(int(nid), int(dof))
        K[g, g] += float(kval)
    return K.tocsc()


def rigid_link_transform(model: dict) -> tuple[csc_matrix, np.ndarray]:
    """T with u_full = T u_reduced: every slave node follows its master rigidly."""
    ndof_full = int(model["ndof"])
    mpc_pairs = np.asarray(model["mpc_pairs"], dtype=int)
    nodes = model["nodes"]
    is_slave = np.zeros(ndof_full, dtype=bool)
    for sn in np.unique(mpc_pairs[:, 1]).astype(int):
        is_slave[6 * (sn - 1): 6 * sn] = True
    keep = np.where(~is_slave)[0]
    full_to_red = np.full(ndof_full, -1, dtype=int)
    full_to_red[keep] = np.arange(keep.size, dtype=int)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for g in keep:
        rows.append(int(g))
        cols.append(int(full_to_red[g]))
        vals.append(1.0)
    for master, slave in mpc_pairs:
        r = nodes[int(slave) - 1, 1:4] - nodes[int(master) - 1, 1:4]
        dx, dy, dz = float(r[0]), float(r[1]), float(r[2])
        C = np.array([[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]])
        A = np.block([[np.eye(3), C], [np.zeros((3, 3)), np.eye(3)]])
        mb, sb = 6 * (int(master) - 1), 6 * (int(slave) - 1)
        for lr in range(6):
            for lc in range(6):
                coeff = float(A[lr, lc])
                if abs(coeff) < 1e-18:
                    continue
                red_col = full_to_red[mb + lc]
                if red_col < 0:
                    raise ValueError(f"invalid rigid link: master node {int(master)} is itself a slave")
                rows.append(sb + lr)
                cols.append(int(red_col))
                vals.append(coeff)
    T = coo_matrix((vals, (rows, cols)), shape=(ndof_full, keep.size)).tocsc()
    return T, keep


def solve_modes(model: dict, n_modes: int) -> tuple[np.ndarray, np.ndarray, csc_matrix]:
    """Natural frequencies [Hz] and mass-normalised mode shapes at every DOF.

    Returns ``(freq_hz, Phi_full, M_full)``: ``Phi_full`` is (6 * n_nodes, n_modes),
    slave DOFs recovered through the rigid links, normalised so that
    phi^T M_full phi = 1.  The eigensolver is shift-invert Lanczos about 0.5 Hz.
    """
    K_full, M_full = assemble_KM(model)
    K_full = add_springs(K_full, model)
    ndof_full = int(model["ndof"])
    if np.asarray(model["mpc_pairs"]).size:
        T, keep = rigid_link_transform(model)
        full_to_red = np.full(ndof_full, -1, dtype=int)
        full_to_red[keep] = np.arange(keep.size, dtype=int)
        K = (T.T @ K_full @ T).tocsc()
        M = (T.T @ M_full @ T).tocsc()
    else:
        T = None
        full_to_red = np.arange(ndof_full, dtype=int)
        K, M = K_full, M_full

    fixed = np.zeros(K.shape[0], dtype=bool)
    for nid, dof, _ in model["bc"]:
        g_red = int(full_to_red[gdof(int(nid), int(dof))])
        if g_red >= 0:                       # a support on a slave DOF would be ignored
            fixed[g_red] = True
    free = np.where(~fixed)[0]
    Kff, Mff = K[free][:, free], M[free][:, free]
    if Kff.shape[0] <= 2:
        raise ValueError("not enough free DOFs to solve for modes")
    k_use = min(int(n_modes), Kff.shape[0] - 2)
    # A seeded start vector makes the eigensolver deterministic.  Left to itself,
    # ARPACK draws a random one in every process, which moves mode shapes by ~1e-11
    # between otherwise identical runs.
    v0 = np.random.default_rng(EIGSH_START_SEED).uniform(-1.0, 1.0, Kff.shape[0])
    vals, vecs = eigsh(Kff, k=k_use, M=Mff, sigma=(2 * np.pi * 0.5) ** 2, which="LM", tol=1e-10, maxiter=6000,
                       v0=v0)

    idx = np.argsort(np.real(vals))
    lam = np.real(vals[idx])
    lam[lam < 0] = 0
    freq_hz = np.sqrt(lam) / (2 * np.pi)
    Phi_red = np.zeros((K.shape[0], k_use))
    Phi_red[free, :] = vecs[:, idx]
    Phi = T @ Phi_red if T is not None else Phi_red
    for k in range(k_use):
        phi = Phi[:, k]
        mn = np.real(phi @ (M_full @ phi))
        if mn > 0:
            Phi[:, k] = phi / np.sqrt(mn)
    return freq_hz, Phi, M_full


# ---------------------------------------------------------------------------
# Node selection and mode-shape reduction
# ---------------------------------------------------------------------------
def gdof(nid: int, dof_1based: int) -> int:
    """0-based global index of DOF ``dof_1based`` (1..6) of node ``nid`` (1-based)."""
    return 6 * (nid - 1) + (dof_1based - 1)


def nodes_at(model: dict, *, x=None, y=None, z=None, tol=1e-6) -> np.ndarray:
    coords = model["nodes"][:, 1:4]
    ids = model["nodes"][:, 0].astype(int)
    mask = np.ones(ids.size, dtype=bool)
    for k, v in enumerate((x, y, z)):
        if v is not None:
            mask &= np.abs(coords[:, k] - float(v)) <= tol
    return ids[mask]


def support_nodes(model: dict) -> np.ndarray:
    """Hinge nodes at x = 0 and the two ends of the bottom beam, sorted."""
    bb = np.asarray(model["bottom_node_ids"], dtype=int).reshape(-1)
    bb_ends = np.array([int(bb[0]), int(bb[-1])], dtype=int) if bb.size >= 2 else np.zeros(0, dtype=int)
    return np.unique(np.r_[nodes_at(model, x=0.0, tol=1e-6), bb_ends]).astype(int)


def influence_vector(model: dict, node_ids: np.ndarray, dof_1based: int) -> np.ndarray:
    r = np.zeros(int(model["ndof"]), dtype=np.float64)
    for nid in np.asarray(node_ids, dtype=int):
        r[gdof(int(nid), int(dof_1based))] = 1.0
    return r


def influence_matrix(model: dict, node_ids: np.ndarray, dofs_1based: list[int]) -> csc_matrix:
    """One column per (node, DOF) pair: a unit ground motion at that DOF only."""
    rows, cols, vals = [], [], []
    col = 0
    for nid in np.asarray(node_ids, dtype=int).reshape(-1):
        for dof in dofs_1based:
            rows.append(gdof(int(nid), int(dof)))
            cols.append(col)
            vals.append(1.0)
            col += 1
    return coo_matrix((vals, (rows, cols)), shape=(int(model["ndof"]), col)).tocsc()


def modal_participation(Phi: np.ndarray, M_full: csc_matrix, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Participation factors Gamma = Phi^T M r and effective masses Gamma^2 / (r^T M r)."""
    Mr = M_full @ r
    Gamma = np.asarray(Phi.T @ Mr, dtype=float).reshape(-1)
    denom = float(r @ Mr)
    if denom <= 0:
        raise ValueError("r^T M r must be positive")
    M_eff = (Gamma**2) / denom
    if float(np.sum(M_eff)) <= 0:
        raise ValueError("effective modal mass is non-positive")
    return Gamma, M_eff


def girder_node_ids(model: dict) -> np.ndarray:
    elems = np.asarray(model["elems"], dtype=float)
    mask = elems[:, 3].astype(int) == G_GIRDER
    return np.unique(np.r_[elems[mask, 1].astype(int), elems[mask, 2].astype(int)]).astype(int)


def girder_station_nodes(model: dict, x_fracs: list[float]) -> np.ndarray:
    """Girder nodes nearest to x = frac * L, ordered girder by girder (j = 0..Ng-1)."""
    nodes = np.asarray(model["nodes"], dtype=float)
    gids = girder_node_ids(model)
    Ng, L_mm, dy_mm = int(model["Ng"]), float(model["L_mm"]), float(model["dy_mm"])
    xg, yg = nodes[gids - 1, 1], nodes[gids - 1, 2]
    band = (yg >= -1e-6) & (yg <= (Ng - 1) * dy_mm + 1e-6)
    gids, xg, yg = gids[band], xg[band], yg[band]
    selected: list[int] = []
    for j in range(Ng):
        jmask = np.abs(yg - j * dy_mm) <= 1e-6
        if not np.any(jmask):
            raise ValueError(f"no girder nodes on girder {j}")
        cand_ids, cand_x = gids[jmask], xg[jmask]
        for frac in x_fracs:
            selected.append(int(cand_ids[int(np.argmin(np.abs(cand_x - float(frac) * L_mm)))]))
    return np.asarray(selected, dtype=int)


def mode_rows(Phi_full: np.ndarray, node_ids: np.ndarray, dof_1based: int) -> np.ndarray:
    """Rows of ``Phi_full`` for one DOF at the given nodes: (n_nodes, n_modes)."""
    Phi = np.asarray(Phi_full, dtype=np.float64)
    nodes = np.asarray(node_ids, dtype=int).reshape(-1)
    out = np.zeros((nodes.size, Phi.shape[1]), dtype=np.float64)
    for i, nid in enumerate(nodes):
        out[i, :] = Phi[gdof(int(nid), dof_1based), :]
    return out
