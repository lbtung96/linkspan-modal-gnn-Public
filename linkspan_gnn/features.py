"""Signal processing shared by the synthetic and the measured pipelines.

Node features   Welch auto-spectrum of the vertical acceleration at each node,
                row-normalised by its maximum, first 400 bins (0 - 38.96 Hz).
Edge features   complex coherency  P_xy / sqrt(P_xx P_yy)  per directed edge, real
                and imaginary parts interleaved per bin: Re(f0), Im(f0), Re(f1), ...
                (800 values).  Its phase is what tells the network whether two
                nodes move in or out of phase, i.e. the sign of a mode shape.

Welch settings everywhere: Hamming window, nperseg = 2048, noverlap = 1024,
nfft = 2048, no detrending, one-sided density.  The synthetic response is
sampled at 200 Hz, which puts the 400 bins at k * 0.09765625 Hz; measured
signals at other rates are interpolated onto that grid (``to_grid``).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, csd, get_window, sosfiltfilt, welch

WELCH_NPERSEG = 2048
WELCH_NOVERLAP = 1024
WELCH_NFFT = 2048
N_FREQ_BINS = 400
TRAINING_FS_HZ = 200.0
N_PROPAGATION_ITER = 50


def bandpass_filter(signal: np.ndarray, fs: float, f_low: float = 1.5, f_high: float = 20.0,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (effective order 2 * order)."""
    nyq = 0.5 * fs
    if f_low <= 0 or f_high >= nyq:
        raise ValueError(f"passband [{f_low}, {f_high}] Hz is outside (0, {nyq:.1f}) Hz")
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(signal.dtype)
    return np.apply_along_axis(lambda x: sosfiltfilt(sos, x), axis=-1, arr=signal).astype(signal.dtype)


def welch_psd(signals: np.ndarray, fs: float, *, nperseg: int = WELCH_NPERSEG,
              noverlap: int = WELCH_NOVERLAP, nfft: int = WELCH_NFFT) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Welch PSD: returns (freq_hz, psd) with psd (n_rows, nfft // 2 + 1)."""
    window = get_window("hamming", nperseg, fftbins=True)
    f, p0 = welch(signals[0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                  detrend=False, return_onesided=True, scaling="density")
    out = np.zeros((signals.shape[0], p0.size), dtype=float)
    out[0] = p0
    for i in range(1, signals.shape[0]):
        _, out[i] = welch(signals[i], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                          detrend=False, return_onesided=True, scaling="density")
    return f, out


def normalize_rows_by_max_abs(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_max = np.max(np.abs(X), axis=1)
    return X / np.maximum(row_max, eps)[:, None]


def normalize_cols_by_max_abs(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Columns divided by their max |value|; all-zero columns stay zero."""
    A = np.asarray(X, dtype=float)
    scale = np.max(np.abs(A), axis=0)
    out = np.zeros_like(A, dtype=float)
    nz = scale > eps
    out[:, nz] = A[:, nz] / scale[nz]
    return out


def edge_coherency(edge_index: np.ndarray, signals: np.ndarray, fs: float, *, nperseg: int = WELCH_NPERSEG,
                   noverlap: int = WELCH_NOVERLAP, nfft: int = WELCH_NFFT) -> np.ndarray:
    """Complex coherency spectrum of every directed edge: (E, nfft // 2 + 1), complex64."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    sig = np.asarray(signals, dtype=float)
    kw = dict(fs=fs, window="hamming", nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False,
              return_onesided=True, scaling="density")
    out = np.zeros((edge_index.shape[1], nfft // 2 + 1), dtype=np.complex64)
    auto: dict[int, np.ndarray] = {}
    for col in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, col]), int(edge_index[1, col])
        for n in (src, dst):
            if n not in auto:
                auto[n] = np.asarray(csd(sig[n, :], sig[n, :], **kw)[1], dtype=np.complex128)
        pxy = np.asarray(csd(sig[src, :], sig[dst, :], **kw)[1], dtype=np.complex128)
        denom = np.sqrt(np.maximum(np.real(auto[src]), 0.0) * np.maximum(np.real(auto[dst]), 0.0))
        coh = np.zeros_like(pxy, dtype=np.complex128)
        valid = denom > 1e-18
        coh[valid] = pxy[valid] / denom[valid]
        out[col, :] = np.asarray(coh, dtype=np.complex64)
    return out


def interleave_re_im(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """(E, n) real and imaginary parts -> (E, 2n) float32, Re/Im alternating per bin."""
    out = np.zeros((re.shape[0], 2 * re.shape[1]), dtype=np.float32)
    out[:, 0::2] = np.asarray(re).astype(np.float32)
    out[:, 1::2] = np.asarray(im).astype(np.float32)
    return out


def training_frequency_grid() -> np.ndarray:
    """The 400 frequencies [Hz] of the node and edge features, as Welch returns them at 200 Hz."""
    f, _ = welch(np.zeros(WELCH_NPERSEG), fs=TRAINING_FS_HZ, window=get_window("hamming", WELCH_NPERSEG, fftbins=True),
                 nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, nfft=WELCH_NFFT, detrend=False,
                 return_onesided=True, scaling="density")
    return np.asarray(f[:N_FREQ_BINS], dtype=float)


def to_grid(f_native: np.ndarray, rows: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Map spectra onto ``grid``: truncation if the native grid already matches, else linear interpolation."""
    n = grid.size
    if f_native.size >= n and np.allclose(f_native[:n], grid, rtol=0, atol=1e-9):
        return np.ascontiguousarray(rows[:, :n])
    return np.stack([np.interp(grid, f_native, r) for r in rows], axis=0)


def _propagation_matrix(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Symmetrically normalised adjacency D^-1/2 A D^-1/2 as a sparse tensor."""
    w = torch.ones(edge_index.shape[1], dtype=torch.float32)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.scatter_add_(0, col, w)
    d = deg.pow(-0.5)
    d[d == float("inf")] = 0.0
    return torch.sparse_coo_tensor(edge_index, d[row] * w * d[col], (n_nodes, n_nodes))


def feature_propagation(x: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor, n_nodes: int,
                        n_iter: int = N_PROPAGATION_ITER) -> torch.Tensor:
    """Fill unknown rows of ``x`` by diffusion from the known rows (``mask``), which stay clamped."""
    out = torch.zeros_like(x)
    out[mask] = x[mask]
    adj = _propagation_matrix(edge_index, n_nodes)
    for _ in range(n_iter):
        out = torch.sparse.mm(adj, out)
        out[mask] = x[mask]
    return out


def propagate(x: np.ndarray, known_idx: np.ndarray, edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """``feature_propagation`` on numpy arrays (float32 internally)."""
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[known_idx] = True
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return feature_propagation(xt, mask, torch.from_numpy(np.asarray(edge_index, dtype=np.int64)),
                               n_nodes).numpy()
