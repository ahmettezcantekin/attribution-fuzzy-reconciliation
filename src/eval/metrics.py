"""Recovery + ranking metrics vs ground truth (plan §5.3, manuscript §2.8)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((pred - truth) ** 2).mean()))


def kl_divergence(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-9) -> float:
    """Mean KL(truth || pred) over rows (credit vectors)."""
    p = np.clip(pred, eps, None); p = p / p.sum(axis=1, keepdims=True)
    q = np.clip(truth, eps, None); q = q / q.sum(axis=1, keepdims=True)
    return float((q * np.log(q / p)).sum(axis=1).mean())


def earth_mover(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean 1-D Earth-Mover distance over credit vectors (channels as ordered bins)."""
    pos = np.arange(pred.shape[1])
    pred = np.clip(pred, 0.0, None)
    truth = np.clip(truth, 0.0, None)
    return float(np.mean([wasserstein_distance(pos, pos, pr, tr)
                          for pr, tr in zip(pred, truth) if pr.sum() > 0 and tr.sum() > 0]))


def ranking_spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean Spearman correlation between predicted and true channel rankings."""
    vals = []
    for pr, tr in zip(pred, truth):
        if np.ptp(pr) == 0 or np.ptp(tr) == 0:
            continue
        rho = spearmanr(pr, tr).correlation
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")
