"""Fuzzy measure / capacity over the source set (plan §1.3, §5.1, manuscript §2.4).

2-ADDITIVE capacity in Möbius form: singletons m_s (per-source reliability) and pairwise
interaction indices m_st. For a 2-additive capacity the Shapley interaction index equals
m_st, so the pair Möbius terms ARE the interaction indices reported in the paper.

Reliability and interactions are estimated from OBSERVABLE source outputs only (elicited
tier, plan §5.1) — no ground truth. Two design choices (P4 revision):
  - SIGNED structural interaction: negative for shared-lens (redundant) pairs, POSITIVE for
    complementary (disjoint-lens) pairs => super-additive, so correct independent agreement
    is not penalized.
  - ADVERSARIAL-ROBUST reliability: down-weight a source that, on its SHARED support, is a
    consistent outlier vs the consensus of the others (closes the Dempster–Shafer gap).
"""
from __future__ import annotations

import numpy as np


def _agreement_factor(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-source mean agreement with the leave-one-out consensus on SHARED support, in [0,1].

    Restricting to shared support means a complementary source (unique channels) is not
    punished; only a source that disagrees on commonly-covered channels (e.g. an adversary)
    scores low.
    """
    N, S, K = X.shape
    cnt = P.sum(axis=1)
    sumX = (X * P[:, :, None]).sum(axis=1)
    ag = np.ones(S)
    for s in range(S):
        others_sum = sumX - X[:, s, :] * P[:, s, None]
        others_cnt = cnt - P[:, s]
        valid = P[:, s] & (others_cnt > 0)
        if valid.sum() == 0:
            continue
        om = np.zeros((N, K))
        om[valid] = others_sum[valid] / others_cnt[valid, None]
        support = om > 1e-9
        ps = X[:, s, :] * support
        qs = om * support
        ps_sum, qs_sum = ps.sum(1), qs.sum(1)
        ok = valid & (ps_sum > 1e-9) & (qs_sum > 1e-9)
        if ok.sum() == 0:
            continue
        p = ps[ok] / ps_sum[ok, None]
        q = qs[ok] / qs_sum[ok, None]
        sim = 1.0 - 0.5 * np.abs(p - q).sum(1)
        ag[s] = float(sim.mean())
    return ag


def _consensus_corr(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Raw per-source correlation with the leave-one-out consensus over (cell,channel) pairs."""
    N, S, K = X.shape
    cnt = P.sum(axis=1)
    sumX = (X * P[:, :, None]).sum(axis=1)
    corr = np.full(S, 1.0)
    for s in range(S):
        others_cnt = cnt - P[:, s]
        valid = P[:, s] & (others_cnt > 0)
        if valid.sum() < 5:
            continue
        om = (sumX[valid] - X[valid, s, :]) / others_cnt[valid, None]
        a, b = X[valid, s, :].ravel(), om.ravel()
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        corr[s] = float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 1.0))
    return corr


def correlation_reliability(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Reliability = how well a source tracks the consensus (individual quality only).

    NOTE: this boosts redundant sources (they correlate with each other), so it is NOT used
    as the primary singleton weight; redundancy is handled by the interaction term. Kept for
    comparison / the adversary gate (see observable_reliability)."""
    corr = _consensus_corr(X, P)
    return np.clip(corr, 0.05, 1.0) / max(np.clip(corr, 0.05, 1.0).max(), 1e-9)


def observable_reliability(X: np.ndarray, P: np.ndarray, robust: bool = False,
                           adversary_gate: bool = False, gate_thresh: float = 0.15) -> np.ndarray:
    """Per-source reliability in (0,1] from observable outputs only.

    Base: channel coverage + reporting rate (good in redundancy regimes). Adversary gate
    (default on): multiply by clip(corr/thresh, floor, 1) so ONLY a near-zero-correlation
    (random/adversarial) source is discounted, while honest — including redundant — sources
    (high correlation) are untouched. Optional robust agreement factor as well.
    """
    S, K = X.shape[1], X.shape[2]
    coverage = ((X > 0).any(axis=0).sum(axis=1)) / K
    report_rate = P.mean(axis=0)
    base = 0.5 * coverage + 0.5 * report_rate
    if robust:
        ag = _agreement_factor(X, P)
        base = base * (0.5 + 0.5 * ag / max(ag.max(), 1e-9))
    if adversary_gate:
        corr = _consensus_corr(X, P)
        base = base * np.clip(corr / gate_thresh, 0.3, 1.0)   # only ~zero-corr sources hit
    return np.clip(base / base.max(), 1e-3, 1.0)


def observable_redundancy(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Coverage-Jaccard redundancy in [0,1] (shared-lens), used by the conflict layer."""
    S = X.shape[1]
    cover = (X > 0).any(axis=0)
    R = np.zeros((S, S))
    for s in range(S):
        for t in range(s + 1, S):
            inter = (cover[s] & cover[t]).sum()
            union = (cover[s] | cover[t]).sum()
            jac = inter / union if union else 0.0
            both = (P[:, s] & P[:, t]).mean()
            either = (P[:, s] | P[:, t]).mean()
            co = both / either if either else 0.0
            R[s, t] = R[t, s] = float(np.clip(jac * (0.5 + 0.5 * co), 0.0, 1.0))
    return R


def structural_interaction(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Signed structural interaction in [-1,1] from coverage patterns.

    I_st = (|symmetric difference| - |intersection|) / |union|
         = -1 for identical coverage (redundant, sub-additive),
         = +1 for disjoint coverage (complementary, super-additive).
    """
    S = X.shape[1]
    cover = (X > 0).any(axis=0)
    I = np.zeros((S, S))
    for s in range(S):
        for t in range(s + 1, S):
            inter = (cover[s] & cover[t]).sum()
            union = (cover[s] | cover[t]).sum()
            symd = union - inter
            I[s, t] = I[t, s] = ((symd - inter) / union) if union else 0.0
    return I


def build_2additive(reliability: np.ndarray, interaction: np.ndarray, beta: float = 0.5):
    """Build a valid (monotone, normalized) 2-additive capacity in Möbius form.

    m_s   ∝ reliability_s
    m_st  = beta * interaction_st * sqrt(r_s r_t)   (signed: <0 redundant, >0 complementary)

    Monotonicity (2-additive sufficient condition): for each s,
        m_s + sum_{t!=s} min(m_st, 0) >= 0.  Shrink pair terms globally until satisfied,
    then normalize so sum_s m_s + sum_{s<t} m_st = 1.
    Returns (m1, m2) with m2 the symmetric interaction-index matrix.
    """
    S = reliability.shape[0]
    m1 = reliability.astype(float).copy()
    m2 = np.zeros((S, S))
    for s in range(S):
        for t in range(s + 1, S):
            m2[s, t] = m2[t, s] = beta * interaction[s, t] * np.sqrt(reliability[s] * reliability[t])

    for _ in range(50):
        if all(m1[s] + sum(min(m2[s, t], 0.0) for t in range(S) if t != s) >= -1e-12 for s in range(S)):
            break
        m2 *= 0.8

    total = m1.sum() + np.triu(m2, 1).sum()
    if total <= 0:
        total, m2 = m1.sum(), np.zeros_like(m2)
    return m1 / total, m2 / total


def shapley_values(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Shapley value of each source for a 2-additive capacity: phi_s = m_s + 0.5 sum_t m_st."""
    return m1 + 0.5 * np.triu(m2, 1).sum(axis=1) + 0.5 * np.triu(m2, 1).sum(axis=0)


# ----------------------------- k-additive (up to 3) -----------------------------

def structural_triple(X: np.ndarray, P: np.ndarray) -> dict:
    """Triple shared-lens redundancy in [0,1] = coverage-Jaccard of three sources."""
    S = X.shape[1]
    cover = (X > 0).any(axis=0)
    T3 = {}
    for s in range(S):
        for t in range(s + 1, S):
            for u in range(t + 1, S):
                inter = (cover[s] & cover[t] & cover[u]).sum()
                union = (cover[s] | cover[t] | cover[u]).sum()
                T3[(s, t, u)] = (inter / union) if union else 0.0
    return T3


def _mu_from_mobius(m1, m2, m3, S):
    """Capacity values for all subsets (bitmasks) from Möbius coefficients."""
    mu = np.zeros(1 << S)
    for A in range(1 << S):
        members = [i for i in range(S) if A & (1 << i)]
        v = sum(m1[i] for i in members)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                v += m2[members[a], members[b]]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                for c in range(b + 1, len(members)):
                    v += m3.get((members[a], members[b], members[c]), 0.0)
        mu[A] = v
    return mu


def _is_monotone(mu, S, tol=1e-9):
    for A in range(1 << S):
        for i in range(S):
            if not (A & (1 << i)) and mu[A | (1 << i)] < mu[A] - tol:
                return False
    return True


def build_kadditive(reliability, R_pair, T3_triple, beta2=0.5, beta3=0.5):
    """Valid (monotone, normalized) up-to-3-additive capacity in Möbius form.

    m_s   ∝ reliability_s
    m_st  = -beta2 * R_pair_st  * sqrt(r_s r_t)         (pairwise sub-additive)
    m_stu = -beta3 * T3_triple  * (r_s r_t r_u)^(1/3)   (triple sub-additive: captures the
            higher-order redundancy of a same-lens bloc that 2-additive only approximates)

    Monotonicity enforced EXACTLY (subset check) by globally shrinking the interaction terms;
    then normalized so mu(full)=1. Returns (m1, m2, m3).
    """
    S = reliability.shape[0]
    m1 = reliability.astype(float).copy()
    m2 = np.zeros((S, S))
    for s in range(S):
        for t in range(s + 1, S):
            m2[s, t] = m2[t, s] = -beta2 * R_pair[s, t] * np.sqrt(reliability[s] * reliability[t])
    m3 = {(s, t, u): -beta3 * T3_triple[(s, t, u)] * (reliability[s] * reliability[t] * reliability[u]) ** (1 / 3)
          for (s, t, u) in T3_triple}

    factor = 1.0
    for _ in range(60):
        mu = _mu_from_mobius(m1, m2 * factor, {k: v * factor for k, v in m3.items()}, S)
        if _is_monotone(mu, S):
            break
        factor *= 0.8
    m2 = m2 * factor
    m3 = {k: v * factor for k, v in m3.items()}

    total = m1.sum() + np.triu(m2, 1).sum() + sum(m3.values())
    if total <= 0:
        return m1 / m1.sum(), np.zeros((S, S)), {k: 0.0 for k in m3}
    return m1 / total, m2 / total, {k: v / total for k, v in m3.items()}
