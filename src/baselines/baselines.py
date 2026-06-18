"""Baseline aggregators (plan §5.2, manuscript §2.7).

Tensor signature: agg(X, P, **kw) -> (reconciled (N,K), valid (N,)) on the same simplex
output as the method, so metrics are directly comparable. P3 implements the weak tier +
reliability-weighted mean; the strong tier (Dempster-Shafer, correlation-cluster) is wired
in P4.
"""
from __future__ import annotations

import numpy as np


def _closure(C):
    rs = C.sum(axis=1)
    valid = rs > 1e-12
    out = np.full(C.shape, np.nan)
    out[valid] = C[valid] / rs[valid, None]
    return out, valid


def naive_mean(X, P, **kw):
    """Unweighted mean over present sources."""
    Xm = X * P[:, :, None]
    cnt = P.sum(axis=1)[:, None]
    C = np.divide(Xm.sum(axis=1), np.maximum(cnt, 1))
    return _closure(C)


def reliability_weighted_mean(X, P, reliability=None, **kw):
    """Additive reliability-weighted mean — double-counts redundant sources by design."""
    S = X.shape[1]
    w = np.ones(S) if reliability is None else reliability
    Xm = X * P[:, :, None] * w[None, :, None]
    norm = (P * w[None, :]).sum(axis=1)[:, None]
    C = np.divide(Xm.sum(axis=1), np.maximum(norm, 1e-12))
    return _closure(C)


def median_agg(X, P, **kw):
    """Per-channel median over present sources (then closure)."""
    Xm = np.where(P[:, :, None], X, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(P.any(axis=1)[:, None, None], Xm, 0.0), axis=1)
    C = np.nan_to_num(med, nan=0.0)
    return _closure(C)


# ----------------------------- strong tier (plan §5.2) -----------------------------

def dempster_shafer(X, P, reliability=None, **kw):
    """Dempster–Shafer combination over channel singletons with reliability discounting.

    Each present source's credit is a BPA over singletons; discounted by alpha_s=reliability_s
    (mass 1-alpha_s to the frame Theta). Sources are combined by Dempster's rule; the
    reconciled credit is the pignistic transform BetP. Absent sources combine as vacuous.
    """
    N, S, K = X.shape
    alpha = np.ones(S) if reliability is None else np.clip(reliability, 0, 1)
    A_sing = np.zeros((N, K)); A_theta = np.ones(N)  # vacuous start
    for s in range(S):
        pres = P[:, s]
        B_sing = (X[:, s, :] * alpha[s])
        B_theta = 1.0 - alpha[s] * X[:, s, :].sum(axis=1)
        # absent cells -> vacuous (theta=1, sing=0)
        B_sing = np.where(pres[:, None], B_sing, 0.0)
        B_theta = np.where(pres, B_theta, 1.0)
        unnorm_sing = A_sing * B_sing + A_sing * B_theta[:, None] + A_theta[:, None] * B_sing
        unnorm_theta = A_theta * B_theta
        norm = unnorm_sing.sum(axis=1) + unnorm_theta
        ok = norm > 1e-12
        A_sing = np.where(ok[:, None], unnorm_sing / np.where(ok, norm, 1)[:, None], A_sing)
        A_theta = np.where(ok, unnorm_theta / np.where(ok, norm, 1), A_theta)
    betp = A_sing + A_theta[:, None] / K   # pignistic
    return _closure(betp)


def trimmed_mean(X, P, **kw):
    """Per-channel trimmed mean over present sources (drop min and max when >=3 present)."""
    N, S, K = X.shape
    C = np.zeros((N, K))
    for n in range(N):
        present = np.where(P[n])[0]
        if len(present) == 0:
            continue
        for k in range(K):
            vals = np.sort(X[n, present, k])
            if len(vals) >= 3:
                vals = vals[1:-1]
            C[n, k] = vals.mean()
    return _closure(C)


def single_source_trust(X, P, reliability=None, **kw):
    """Use the single most-reliable present source's credit vector per cell."""
    N, S, K = X.shape
    rel = np.ones(S) if reliability is None else reliability
    C = np.zeros((N, K))
    for n in range(N):
        present = np.where(P[n])[0]
        if len(present) == 0:
            continue
        s = present[np.argmax(rel[present])]
        C[n] = X[n, s, :]
    return _closure(C)


def _mu_all_subsets(m1, m2):
    """Capacity value for every subset bitmask, from 2-additive Möbius coefficients."""
    S = len(m1)
    mu = np.zeros(1 << S)
    for A in range(1 << S):
        members = [i for i in range(S) if A & (1 << i)]
        v = sum(m1[i] for i in members)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                v += m2[members[a], members[b]]
        mu[A] = v
    return np.clip(mu, 0.0, None)


def sugeno(X, P, m1=None, m2=None, **kw):
    """Sugeno integral with the SAME elicited capacity as Choquet, then L1-closure.

    A max–min (ordinal) non-additive aggregation: S_mu(x) = max_i min(x_(i), mu(A_(i))).
    Direct contrast to the Choquet integral on the identical capacity (answers
    'why Choquet, not Sugeno?'). Implemented per cell/channel over present sources.
    """
    N, S, K = X.shape
    mu = _mu_all_subsets(m1, m2)
    C = np.zeros((N, K))
    for n in range(N):
        present = [s for s in range(S) if P[n, s]]
        if not present:
            continue
        for k in range(K):
            vals = sorted(((X[n, s, k], s) for s in present))  # ascending
            best = 0.0
            for i in range(len(vals)):
                amask = 0
                for (_, ss) in vals[i:]:
                    amask |= (1 << ss)
                best = max(best, min(vals[i][0], mu[amask]))
            C[n, k] = best
    return _closure(C)


def correlation_cluster_average(X, P, redundancy=None, threshold=0.8, **kw):
    """Cluster sources by redundancy, average within clusters, then across clusters.

    The cheap way to kill double-counting WITHOUT a capacity. The method must beat this
    (plan §5.2). Clusters via union-find on pairs with redundancy >= threshold.
    """
    N, S, K = X.shape
    parent = list(range(S))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    if redundancy is not None:
        for s in range(S):
            for t in range(s + 1, S):
                if redundancy[s, t] >= threshold:
                    parent[find(s)] = find(t)
    clusters = {}
    for s in range(S):
        clusters.setdefault(find(s), []).append(s)

    cluster_vecs = []
    cluster_present = []
    for members in clusters.values():
        Pm = P[:, members]                       # (N, |members|)
        Xm = X[:, members, :] * Pm[:, :, None]
        cnt = Pm.sum(axis=1)
        vec = np.divide(Xm.sum(axis=1), np.maximum(cnt, 1)[:, None])  # (N,K)
        cluster_vecs.append(vec)
        cluster_present.append(cnt > 0)
    CV = np.stack(cluster_vecs, axis=1)          # (N, n_clusters, K)
    CP = np.stack(cluster_present, axis=1)       # (N, n_clusters)
    cnt = CP.sum(axis=1)
    C = np.divide((CV * CP[:, :, None]).sum(axis=1), np.maximum(cnt, 1)[:, None])
    return _closure(C)
