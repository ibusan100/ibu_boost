"""
ScreeningTree — single regression decision tree using screening-based split
selection.

Design (CLAUDE.md § M1):
- Non-oblivious: each node independently chooses its best (feature, bin).
- Split criterion: screening transform applied to MSE gain (see below).
- Stopping: max_depth, min_samples_leaf, or screening rejects all splits.
- Leaf value: mean(y) of samples reaching that node.

Gradient formulation (single tree, not boosted):
    g_i = y_i - mean(y_node)   (residual from node mean)
    h_i = 1
    G_total = 0 always → parent gain term = 0
    raw_gain = G_L²/H_L + G_R²/H_R   (MSE variance reduction)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .binning import Binner
from .diagnostics import NodeDiagnostics, ScreeningDiagnostics
from .screening_split import (
    ScreeningParams,
    build_histogram_numpy,
    screening_split_numpy,
)


# ---------------------------------------------------------------------------
# Internal node representation
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    node_id: int
    depth: int
    sample_idx: np.ndarray   # indices into training X / y

    # Set after evaluation:
    is_leaf: bool = False
    leaf_value: float = 0.0

    # Set for internal nodes:
    split_feature: int = -1
    split_bin: int = -1          # samples with X_binned[:, feat] <= bin go left
    split_threshold: float = float("nan")  # corresponding raw-feature threshold
    left_child: int = -1
    right_child: int = -1


# ---------------------------------------------------------------------------
# ScreeningTree
# ---------------------------------------------------------------------------


class ScreeningTree:
    """Single regression tree with screening-based split selection.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth (root is depth 0).
    min_samples_leaf : int
        Minimum samples required in each child after a split.
    num_bins : int
        Number of quantile bins per feature (passed to Binner).
    params : ScreeningParams or None
        Screening scalars. If None, uses ScreeningParams defaults.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        num_bins: int = 255,
        params: Optional[ScreeningParams] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.params = params if params is not None else ScreeningParams()

        self._binner: Optional[Binner] = None
        self._nodes: list[_Node] = []
        self._diagnostics: Optional[ScreeningDiagnostics] = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScreeningTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        N, F = X.shape
        assert len(y) == N

        # Bin features
        self._binner = Binner(self.num_bins)
        X_binned = self._binner.fit_transform(X).astype(np.int32)
        actual_max_bins = self._binner.max_bins()

        self._nodes = []
        self._diagnostics = ScreeningDiagnostics()

        # BFS
        root = _Node(node_id=0, depth=0, sample_idx=np.arange(N, dtype=np.int32))
        queue: deque[_Node] = deque([root])
        self._nodes.append(root)
        next_id = 1

        while queue:
            node = queue.popleft()
            idx = node.sample_idx
            n = len(idx)
            y_node = y[idx]
            node.leaf_value = float(np.mean(y_node))

            # Stopping conditions that don't need split evaluation
            if node.depth >= self.max_depth or n < 2 * self.min_samples_leaf:
                node.is_leaf = True
                self._diagnostics.nodes.append(NodeDiagnostics(
                    node_id=node.node_id, depth=node.depth,
                    n_samples=n, n_candidates=0, n_accepted=0,
                    accept_rate=0.0, rho_max=0.0, rho_mean=float("nan"),
                    rho_median=float("nan"), split_found=False,
                ))
                continue

            # Compute gradients for this node
            mu = node.leaf_value
            g = (y_node - mu).astype(np.float32)   # g_i = y_i - mu_node
            h = np.ones(n, dtype=np.float32)

            # Build histogram for this node (single node → num_nodes=1)
            nid = np.zeros(n, dtype=np.int32)
            hist_G, hist_H = build_histogram_numpy(
                X_binned[idx], g, h, nid,
                num_nodes=1, num_bins=actual_max_bins,
            )

            # Screening split
            result = screening_split_numpy(hist_G, hist_H, self.params)

            # Diagnostics
            rho_flat = result["rho"].ravel()
            accepted_mask = rho_flat > 0.0
            n_cand = int((actual_max_bins - 1) * F)   # last bin excluded per spec
            n_acc = int(accepted_mask.sum())
            rho_accepted = rho_flat[accepted_mask]
            self._diagnostics.nodes.append(NodeDiagnostics(
                node_id=node.node_id, depth=node.depth,
                n_samples=n, n_candidates=n_cand, n_accepted=n_acc,
                accept_rate=n_acc / n_cand if n_cand > 0 else 0.0,
                rho_max=float(rho_flat.max()),
                rho_mean=float(rho_accepted.mean()) if n_acc > 0 else float("nan"),
                rho_median=float(np.median(rho_accepted)) if n_acc > 0 else float("nan"),
                split_found=bool(result["accepted_mask"][0]),
            ))

            if not result["accepted_mask"][0]:
                node.is_leaf = True
                continue

            # Apply split
            best_feat = int(result["best_feat"][0])
            best_bin  = int(result["best_bin"][0])
            threshold = self._binner.threshold(best_feat, best_bin)

            node.split_feature = best_feat
            node.split_bin = best_bin
            node.split_threshold = threshold

            left_mask  = X_binned[idx, best_feat] <= best_bin
            right_mask = ~left_mask

            # Guard min_samples_leaf
            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                node.is_leaf = True
                continue

            left_idx  = idx[left_mask]
            right_idx = idx[right_mask]

            left_node  = _Node(node_id=next_id,     depth=node.depth + 1, sample_idx=left_idx)
            right_node = _Node(node_id=next_id + 1, depth=node.depth + 1, sample_idx=right_idx)
            next_id += 2

            node.left_child  = left_node.node_id
            node.right_child = right_node.node_id

            self._nodes.append(left_node)
            self._nodes.append(right_node)
            queue.append(left_node)
            queue.append(right_node)

        return self

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._binner is not None, "Call fit() first."
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        preds = np.empty(N, dtype=np.float32)

        # Build a lookup dict from node_id → _Node
        node_map = {n.node_id: n for n in self._nodes}
        root = node_map[0]

        for i in range(N):
            node = root
            while not node.is_leaf:
                if X[i, node.split_feature] <= node.split_threshold:
                    node = node_map[node.left_child]
                else:
                    node = node_map[node.right_child]
            preds[i] = node.leaf_value

        return preds

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    @property
    def diagnostics(self) -> Optional[ScreeningDiagnostics]:
        return self._diagnostics

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self._nodes if n.is_leaf)

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    def depth(self) -> int:
        return max(n.depth for n in self._nodes) if self._nodes else 0
