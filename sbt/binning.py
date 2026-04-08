"""
Quantile-based feature binner for Screening Boosted Trees.

Splits each feature into at most `num_bins` equal-frequency bins using
percentile cut points. Features with fewer unique values get fewer bins
naturally; all are stored as int32 indices in [0, actual_bins-1].

Thresholds are stored per-feature for use in predict() routing:
    X[:, f] <= bin_edges[f][best_bin + 1]  →  left child
"""

from __future__ import annotations

import numpy as np


class Binner:
    """Fit quantile bin edges on training data; transform to int32 bin indices.

    Parameters
    ----------
    num_bins : int
        Maximum number of bins per feature (default 255, like LightGBM).
    """

    def __init__(self, num_bins: int = 255):
        self.num_bins = num_bins
        self.bin_edges_: list[np.ndarray] = []  # per feature, shape (k+1,) for k bins
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> "Binner":
        X = np.asarray(X, dtype=np.float64)
        N, F = X.shape
        self.n_features_ = F
        self.bin_edges_ = []
        quantiles = np.linspace(0.0, 100.0, self.num_bins + 1)
        for f in range(F):
            edges = np.unique(np.percentile(X[:, f], quantiles))
            self.bin_edges_.append(edges.astype(np.float64))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        N, F = X.shape
        assert F == self.n_features_
        X_binned = np.empty((N, F), dtype=np.int32)
        for f in range(F):
            edges = self.bin_edges_[f]
            # Internal cut points: edges[1:-1] (exclude the min/max endpoints).
            # searchsorted gives bin index in [0, len(cuts)] where len(cuts) = len(edges)-2.
            cuts = edges[1:-1]
            X_binned[:, f] = np.searchsorted(cuts, X[:, f], side="right").astype(np.int32)
        return X_binned

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def num_bins_per_feature(self) -> list[int]:
        """Actual number of bins for each feature (may be < num_bins for low-cardinality)."""
        return [len(e) - 1 for e in self.bin_edges_]

    def max_bins(self) -> int:
        return max(self.num_bins_per_feature())

    def threshold(self, feature: int, bin_idx: int) -> float:
        """Right edge of the given bin — used as the split threshold in predict()."""
        edges = self.bin_edges_[feature]
        right_edge_idx = bin_idx + 1
        if right_edge_idx < len(edges):
            return float(edges[right_edge_idx])
        return float("inf")
