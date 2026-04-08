"""
Screening Boosted Trees — split scoring (NumPy reference).

The screening transform mirrors "Screening Is Enough" (Nakanishi 2026,
arXiv:2604.01178) but applied to GBDT split selection rather than attention:

  raw_gain = G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G_total^2/(H_total+lam)
  s        = 1 - exp(-raw_gain / tau)         # bounded "similarity" in [0, 1)
  rho      = max(1 - r * (1 - s), 0) ** 2     # Trim-and-Square (absolute screening)

  tau = exp(s_w) + eps           (learned scalar; analogue of the screening window)
  r   = exp(s_r) + 1             (learned scalar; 1/r is the acceptance width)

A node selects argmax_{(feature, bin)} rho. If max(rho) == 0 the node is
**rejected** — no split is emitted, the node becomes a leaf. This is the GBDT
analogue of "no key is relevant" in Multiscreen: weak splits are removed
cleanly without an external min_gain_to_split heuristic, and the threshold is
itself learnable through s_r.

This module contains the pure-NumPy reference. The Triton-accelerated path
lives in `sbt.kernels.screening_split_triton` and is loaded lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScreeningParams:
    """Learned (or fixed) screening scalars and L2 regularizer.

    s_w, s_r are stored in log space exactly as in the Multiscreen paper:
        tau = exp(s_w) + eps
        r   = exp(s_r) + 1
    """

    # Defaults calibrated on sklearn California housing (N=20640, F=8):
    #   s_w=8.0, s_r=0.0 → tau≈2981, r=2.0 → root accept_rate ≈ 15%
    # Interpretation: tau acts as the gain scale reference; gains below
    # tau * ln(r) ≈ 2067 are rejected. Raise s_w to reject more (smaller
    # accept_rate); lower s_r to accept more.
    s_w: float = 8.0      # log-scale gain temperature; tau = exp(s_w) + eps
    s_r: float = 0.0      # log-scale acceptance width; r = exp(s_r) + 1
    lam: float = 1.0      # L2 reg on hessian
    eps: float = 1e-6

    def tau(self) -> float:
        return float(np.exp(self.s_w)) + self.eps

    def r(self) -> float:
        return float(np.exp(self.s_r)) + 1.0


def build_histogram_numpy(
    X_binned: np.ndarray,   # [N, F] int — pre-binned features
    g: np.ndarray,          # [N] float32 — gradients
    h: np.ndarray,          # [N] float32 — hessians
    node_id: np.ndarray,    # [N] int32 — leaf membership
    num_nodes: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-(node, feature, bin) gradient / hessian histograms."""
    N, F = X_binned.shape
    hist_G = np.zeros((num_nodes, F, num_bins), dtype=np.float32)
    hist_H = np.zeros((num_nodes, F, num_bins), dtype=np.float32)
    for f in range(F):
        np.add.at(hist_G, (node_id, f, X_binned[:, f]), g)
        np.add.at(hist_H, (node_id, f, X_binned[:, f]), h)
    return hist_G, hist_H


def _bounded_gain(raw_gain: np.ndarray, tau: float) -> np.ndarray:
    raw_gain = np.maximum(raw_gain, 0.0)
    return 1.0 - np.exp(-raw_gain / tau)


def _trim_square(s: np.ndarray, r: float) -> np.ndarray:
    return np.square(np.maximum(1.0 - r * (1.0 - s), 0.0))


def screening_split_numpy(
    hist_G: np.ndarray,     # [num_nodes, F, num_bins]
    hist_H: np.ndarray,
    params: ScreeningParams,
) -> dict:
    """Reference implementation: cumsum scan + Trim-and-Square per (node, feat),
    then per-node reduction across (feat, bin).

    Returns
    -------
    dict with:
        rho            [num_nodes, F, num_bins]   screened relevance per candidate
        best_feat      [num_nodes]                argmax feature
        best_bin       [num_nodes]                argmax bin
        best_rho       [num_nodes]                relevance of the chosen split
        accepted_mask  [num_nodes]                False => leaf (no split survived)
    """
    num_nodes, F, B = hist_G.shape
    tau = params.tau()
    r = params.r()
    lam = params.lam

    G_total = hist_G.sum(axis=2, keepdims=True)
    H_total = hist_H.sum(axis=2, keepdims=True)
    parent = (G_total ** 2) / (H_total + lam)

    G_L = np.cumsum(hist_G, axis=2)
    H_L = np.cumsum(hist_H, axis=2)
    G_R = G_total - G_L
    H_R = H_total - H_L

    raw_gain = (G_L ** 2) / (H_L + lam) + (G_R ** 2) / (H_R + lam) - parent
    raw_gain[..., -1] = -np.inf  # last bin has no right child

    s = _bounded_gain(raw_gain, tau)
    rho = _trim_square(s, r)
    rho[..., -1] = 0.0

    flat = rho.reshape(num_nodes, F * B)
    best_flat = flat.argmax(axis=1)
    best_feat = (best_flat // B).astype(np.int32)
    best_bin = (best_flat % B).astype(np.int32)
    best_rho = flat.max(axis=1).astype(np.float32)
    accepted = best_rho > 0.0

    return {
        "rho": rho,
        "best_feat": best_feat,
        "best_bin": best_bin,
        "best_rho": best_rho,
        "accepted_mask": accepted,
    }
