"""Layer 4 — Mamdani confidence layer (plan §1.4, §1.5, manuscript §2.5).

A Mamdani fuzzy inference system maps (interaction-corrected disagreement, mean reliability)
-> reconciliation confidence, from which an uncertainty band half-width is derived. Membership
functions use scikit-fuzzy (triangular); inference (min-implication, max-aggregation, centroid
defuzzification) is vectorized over all cells for speed.

NOTE: this band is RAW. The conformal calibration step (eval/calibration.py, plan §5.3) maps
it to nominal empirical coverage; here we only produce the interpretable raw confidence.
"""
from __future__ import annotations

import numpy as np
import skfuzzy as fuzz

_GRID = np.linspace(0.0, 1.0, 101)
# output (confidence) membership functions
_CONF = {
    "low": fuzz.trimf(_GRID, [0.0, 0.0, 0.5]),
    "med": fuzz.trimf(_GRID, [0.25, 0.5, 0.75]),
    "high": fuzz.trimf(_GRID, [0.5, 1.0, 1.0]),
}


def _mf(x, abc):
    return fuzz.trimf(x, abc)


def confidence(disagreement: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    """Vectorized Mamdani inference -> confidence in [0,1] per cell.

    disagreement: (N,) interaction-corrected conflict, normalized to [0,1].
    reliability:  (N,) mean reliability of the reporting sources, in [0,1].
    """
    d = np.clip(disagreement, 0, 1)
    r = np.clip(reliability, 0, 1)

    d_lo, d_md, d_hi = _mf(d, [0, 0, 0.5]), _mf(d, [0.25, 0.5, 0.75]), _mf(d, [0.5, 1, 1])
    r_lo, r_md, r_hi = _mf(r, [0, 0, 0.5]), _mf(r, [0.25, 0.5, 0.75]), _mf(r, [0.5, 1, 1])

    # rule base (min-implication): strength per rule -> consequent confidence term
    rules = [
        (np.minimum(d_lo, r_hi), "high"),   # low disagreement, high reliability -> high conf
        (np.minimum(d_lo, r_md), "high"),
        (np.minimum(d_md, r_hi), "med"),
        (d_md, "med"),
        (np.minimum(d_hi, np.ones_like(r)), "low"),  # high disagreement -> low conf
        (r_lo, "low"),                                # low reliability -> low conf
    ]

    N = d.shape[0]
    agg = np.zeros((N, _GRID.shape[0]))
    for strength, label in rules:
        clipped = np.minimum(strength[:, None], _CONF[label][None, :])
        agg = np.maximum(agg, clipped)

    denom = agg.sum(axis=1)
    conf = np.where(denom > 0, (agg * _GRID[None, :]).sum(axis=1) / np.where(denom > 0, denom, 1), 0.5)
    return conf


def band_halfwidth(conf: np.ndarray, base: float = 0.25) -> np.ndarray:
    """Raw uncertainty band half-width: low confidence -> wide band."""
    return (1.0 - np.clip(conf, 0, 1)) * base
