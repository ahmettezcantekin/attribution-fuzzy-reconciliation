"""Layer 2 — disagreement characterization (plan §1.2, §1.5, manuscript §2.3).

Per-cell conflict = INTERACTION-CORRECTED dispersion among the present sources: pairwise
credit distances weighted by source INDEPENDENCE (1 - redundancy). This enforces the
governing invariant (§1.5): disagreement among independent sources counts as real conflict;
agreement among redundant sources is discounted (it is not genuine consensus).
"""
from __future__ import annotations

import numpy as np


def cell_conflict(X: np.ndarray, P: np.ndarray, redundancy: np.ndarray) -> np.ndarray:
    """Return per-cell interaction-corrected disagreement in [0, 2] (0 = full consensus)."""
    N, S, K = X.shape
    num = np.zeros(N)
    den = np.zeros(N)
    for s in range(S):
        for t in range(s + 1, S):
            both = (P[:, s] & P[:, t]).astype(float)
            w = (1.0 - redundancy[s, t]) * both          # independent pairs weighted more
            l1 = np.abs(X[:, s, :] - X[:, t, :]).sum(axis=1)
            num += w * l1
            den += w
    out = np.zeros(N)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out
