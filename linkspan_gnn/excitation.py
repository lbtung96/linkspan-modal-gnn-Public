"""Ambient-vibration response of a deck, by modal superposition.

Excitation: independent band-unlimited Gaussian white-noise ground accelerations
at every translational DOF (x, y, z) of every support node - the hinge nodes at
x = 0 and the two ends of the bottom beam.  Because each support DOF has its own
record, the input is not a uniform base shaking: it excites torsion even on a
mirror-symmetric deck.

Response: each retained mode is integrated as a single-degree-of-freedom
oscillator (Newmark average acceleration, 1 % damping) under its modal load
-Gamma a_g; the absolute vertical acceleration at every node is the modal
superposition plus the imposed ground acceleration.

The same random seed is used for every structure, so differences between
structures come from the structures, not from the noise draw.
"""

from __future__ import annotations

import numpy as np

from .fe_model import influence_matrix, support_nodes
from .features import bandpass_filter

SEED = 12345
ZETA = 0.01
SIGMA_AG_MMPS2 = 100.0       # RMS ground acceleration per input, mm/s^2
T_TOTAL_S = 60.0
DT_S = 0.005
FS_HZ = 200.0
BANDPASS_HZ = (1.5, 20.0)


def newmark_sdof(omega: float, zeta: float, p_t: np.ndarray, dt: float,
                 beta: float = 0.25, gamma: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q'' + 2 zeta omega q' + omega^2 q = p(t), from rest; returns (q, q', q'')."""
    nt = int(p_t.size)
    q = np.zeros(nt, dtype=float)
    qd = np.zeros(nt, dtype=float)
    qdd = np.zeros(nt, dtype=float)
    k = float(omega**2)
    c = float(2.0 * zeta * omega)
    if nt == 0:
        return q, qd, qdd
    qdd[0] = p_t[0] - c * qd[0] - k * q[0]
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 + a1 * c
    for i in range(nt - 1):
        p_eff = p_t[i + 1] + (a0 * q[i] + a2 * qd[i] + a3 * qdd[i]) + c * (a1 * q[i] + a4 * qd[i] + a5 * qdd[i])
        q[i + 1] = p_eff / k_eff
        qdd[i + 1] = a0 * (q[i + 1] - q[i]) - a2 * qd[i] - a3 * qdd[i]
        qd[i + 1] = qd[i] + dt * ((1.0 - gamma) * qdd[i] + gamma * qdd[i + 1])
    return q, qd, qdd


def vertical_acceleration(model: dict, Phi: np.ndarray, freq_hz: np.ndarray, M_full,
                          *, bandpass_hz: tuple[float, float] | None = BANDPASS_HZ) -> np.ndarray:
    """Absolute vertical acceleration at every FE node, (n_nodes, n_samples), at ``FS_HZ``.

    ``Phi`` / ``freq_hz`` are the modes superposed (the paper uses all 15 solved
    modes).  The result is band-passed (zero-phase Butterworth) unless
    ``bandpass_hz`` is None.
    """
    Phi = np.asarray(Phi, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float).reshape(-1)
    n_nodes = int(np.asarray(model["nodes"]).shape[0])

    R = influence_matrix(model, support_nodes(model), [1, 2, 3])
    nt = int(round(T_TOTAL_S / DT_S))
    rng = np.random.default_rng(SEED)
    ag = SIGMA_AG_MMPS2 * rng.standard_normal((int(R.shape[1]), nt))

    gamma = np.asarray(Phi.T @ (M_full @ R), dtype=float)        # (n_modes, n_inputs)
    p_modal = -gamma @ ag
    qdd = np.zeros((freq_hz.size, nt), dtype=float)
    for k, omega in enumerate(2.0 * np.pi * freq_hz):
        _, _, qdd[k, :] = newmark_sdof(omega, ZETA, p_modal[k, :], DT_S, beta=0.25, gamma=0.5)

    a_abs = Phi @ qdd + R @ ag                                    # every DOF
    a3 = np.zeros((n_nodes, nt), dtype=float)
    for nid in range(1, n_nodes + 1):
        a3[nid - 1, :] = a_abs[6 * (nid - 1) + 2, :]
    if bandpass_hz is not None:
        a3 = bandpass_filter(a3, FS_HZ, f_low=float(bandpass_hz[0]), f_high=float(bandpass_hz[1]))
    return a3
