"""Layer 1 — fuzzy representation (plan §1.1, manuscript §2.2).

Each source's credit becomes a triangular fuzzy number whose spread encodes that source's
LOCAL reliability: higher reliability -> narrower spread. The spread feeds the conflict
measure (Layer 2) and the band width (Layer 4).
"""
from __future__ import annotations

import numpy as np


def triangular_spread(credit: np.ndarray, reliability: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Half-width delta of the triangular fuzzy number for each source-channel credit.

    Args:
        credit: array (..., S, K) of point credits.
        reliability: array (S,) in (0,1]; higher => narrower spread.
        scale: max relative spread.
    Returns:
        delta array broadcastable to `credit`: delta = scale * (1 - reliability) * credit.
    """
    rel = np.clip(reliability, 1e-6, 1.0)
    shape = [1] * credit.ndim
    shape[-2] = rel.shape[0]  # align on the source axis (second to last)
    return scale * (1.0 - rel).reshape(shape) * credit
