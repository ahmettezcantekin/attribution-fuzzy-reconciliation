"""Layer 3 — Choquet aggregation with L1-closure (plan §1.3, §5.1, manuscript §2.4).

For a 2-additive capacity in Möbius form, the discrete Choquet integral has the closed form
    C(x) = sum_s m_s x_s + sum_{s<t} m_st * min(x_s, x_t),
applied CHANNEL-WISE across sources, followed by an L1-closure to return a valid credit
distribution (Proposition 1).

Proposition 1 (simplex consistency): the per-channel sum equals 1 under an additive capacity
(all m_st = 0), so the closure is the identity and the result reduces to the weighted
arithmetic mean; otherwise the closure corrects only the non-additive mass defect. The
per-cell present-source renormalization cancels in the closure, so missing sources are
handled simply by masking.
"""
from __future__ import annotations

import numpy as np


def reconcile_choquet(X: np.ndarray, P: np.ndarray, m1: np.ndarray, m2: np.ndarray):
    """Reconcile per-cell source credit tensors via 2-additive Choquet + L1-closure.

    Args:
        X: (N, S, K) source credits (0 where a source is absent).
        P: (N, S) bool presence mask.
        m1: (S,) Möbius singletons; m2: (S,S) symmetric Möbius pairs.
    Returns:
        reconciled: (N, K) credit distributions (rows with no reports are NaN).
        valid: (N,) bool mask of rows that produced a distribution.
    """
    N, S, K = X.shape
    Xm = X * P[:, :, None]  # zero absent sources

    # linear term: sum_s m_s x_{s,k}  (absent sources contribute 0)
    linear = np.einsum("s,nsk->nk", m1, Xm)

    # pairwise term: sum_{s<t} m_st * P_s P_t * min(x_s, x_t)
    pair = np.zeros((N, K))
    for s in range(S):
        for t in range(s + 1, S):
            if m2[s, t] == 0:
                continue
            both = (P[:, s] & P[:, t])[:, None]
            mn = np.minimum(Xm[:, s, :], Xm[:, t, :])
            pair += m2[s, t] * both * mn

    C = linear + pair
    C = np.clip(C, 0.0, None)  # guard tiny negatives from non-additivity
    row_sum = C.sum(axis=1)
    valid = row_sum > 1e-12
    reconciled = np.full((N, K), np.nan)
    reconciled[valid] = C[valid] / row_sum[valid, None]   # L1-closure (Eq. 2)
    return reconciled, valid


def reconcile_kadditive(X, P, m1, m2, m3):
    """Up-to-3-additive Choquet + L1-closure.

    C(x) = sum_s m_s x_s + sum_{s<t} m_st min(x_s,x_t) + sum_{s<t<u} m_stu min(x_s,x_t,x_u),
    channel-wise, then L1-closure. Missing sources are masked; the per-cell scaling cancels
    in the closure (Proposition 1 extends to the k-additive case under a monotone capacity).
    """
    N, S, K = X.shape
    Xm = X * P[:, :, None]
    C = np.einsum("s,nsk->nk", m1, Xm)
    for s in range(S):
        for t in range(s + 1, S):
            if m2[s, t] != 0:
                both = (P[:, s] & P[:, t])[:, None]
                C += m2[s, t] * both * np.minimum(Xm[:, s, :], Xm[:, t, :])
    for (s, t, u), val in m3.items():
        if val != 0:
            allp = (P[:, s] & P[:, t] & P[:, u])[:, None]
            mn = np.minimum(np.minimum(Xm[:, s, :], Xm[:, t, :]), Xm[:, u, :])
            C += val * allp * mn
    C = np.clip(C, 0.0, None)
    row_sum = C.sum(axis=1)
    valid = row_sum > 1e-12
    reconciled = np.full((N, K), np.nan)
    reconciled[valid] = C[valid] / row_sum[valid, None]
    return reconciled, valid
