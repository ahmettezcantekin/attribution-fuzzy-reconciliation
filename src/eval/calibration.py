"""Uncertainty calibration (plan §5.3, manuscript §2.8).

A raw Mamdani confidence band is not a calibrated interval. We map it to nominal empirical
coverage with a split-conformal step: on a calibration split, find the band scale q such
that |reconciled - truth| <= q * half-width at the nominal rate; apply q on the test split.
Report RAW (q=1) vs CALIBRATED coverage so the gap is visible.

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
    score = resid / h                                  # nonconformity

    out = {}
    raw_test = (resid[test] <= h[test]).mean()         # raw band coverage (q=1)
    for lvl in levels:
        q = np.quantile(score[cal].ravel(), lvl)
        cov = (resid[test] <= q * h[test]).mean()
        out[lvl] = (float(raw_test), float(cov), float(q))
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
