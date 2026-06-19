"""Uncertainty calibration (plan §5.3, manuscript §4.6, Proposition 8).

A raw Mamdani confidence band is not a calibrated interval. We map it to nominal coverage with a
split-conformal step (the multiplicative / locally-adaptive variant): the per-(cell,channel)
nonconformity score is the ratio |reconciled - truth| / half-width; q_alpha is the conformal
(1-alpha)-quantile of the calibration scores, i.e. their ceil((n+1)(1-alpha))-th order statistic;
the calibrated band is reconciled +- q_alpha * half-width, which preserves the per-cell adaptive
width. This is exactly the object Proposition 8 certifies. Report RAW (q=1) vs CALIBRATED coverage.

    py -m src.eval.calibration
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..methods import reconcile as rec

ROOT = Path(__file__).resolve().parents[2]


def conformal_coverage(reconciled, T, half, valid, levels=(0.80, 0.90), seed=0):
    """Return dict level -> (raw_coverage, calibrated_coverage, q)."""
    rng = np.random.default_rng(seed)
    idx = np.where(valid)[0]
    rng.shuffle(idx)
    cut = len(idx) // 2
    cal, test = idx[:cut], idx[cut:]

    resid = np.abs(reconciled - T)                     # (N,K)
    h = np.maximum(half[:, None], 1e-6)                # per-cell half-width, broadcast to K
    score = resid / h                                  # per-(cell,channel) ratio nonconformity

    cal_scores = np.sort(score[cal].ravel())
    n = cal_scores.size
    out = {}
    raw_test = (resid[test] <= h[test]).mean()         # raw band coverage (q=1)
    for lvl in levels:
        k = int(np.ceil((n + 1) * lvl))                # conformal rank (finite-sample, +1)
        q = float(cal_scores[min(k, n) - 1])           # ceil((n+1)*lvl)-th smallest score
        cov = (resid[test] <= q * h[test]).mean()      # band = reconciled +- q * h
        out[lvl] = (float(raw_test), float(cov), q)
    return out


def main():
    X, P, T, conv_ids, sources, channels = rec.load_tensors(ROOT / "results")
    r = rec.run(X, P, beta=0.5)
    res = conformal_coverage(r["reconciled"], T, r["band_half"], r["valid"])
    print("Uncertainty calibration (default synthetic data):")
    print(f"  mean confidence={r['confidence'][r['valid']].mean():.3f}, "
          f"mean raw half-width={r['band_half'][r['valid']].mean():.3f}")
    for lvl, (raw, cov, q) in res.items():
        print(f"  nominal {lvl:.0%}: raw coverage={raw:.3f}  ->  calibrated={cov:.3f}  (q={q:.2f})")
    return res


if __name__ == "__main__":
    main()
