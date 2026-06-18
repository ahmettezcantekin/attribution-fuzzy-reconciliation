"""Source distortion models (plan §4.1).

Each source produces an observed credit vector as a DISTORTED view of the journey through
its own lens: partial observability (visibility set V_s), rule bias, window truncation,
self-preferencing, observation noise, missingness. Two extensions support P4:
  - rule "adversarial": a heavily-biased source that ignores the journey (robustness test);
  - "noise_group": sources sharing a group draw correlated noise => higher-order dependence
    (the redundant-bloc / "dependence" misspecification regime).

Anti-circularity (plan §4.0): redundancy among sources EMERGES from overlapping V_s and
shared-noise groups; none of this is shared with src/methods.
"""
from __future__ import annotations

import numpy as np


def _apply_rule(rule: str, touches, K: int, half_life: float | None):
    credit = np.zeros(K)
    if not touches:
        return credit
    chs = np.array([c for c, _ in touches])
    ts = np.array([t for _, t in touches])
    if rule == "last_touch":
        credit[chs[np.argmin(ts)]] += 1.0
    elif rule == "first_touch":
        credit[chs[np.argmax(ts)]] += 1.0
    elif rule == "time_decay":
        hl = half_life or 1.0
        for c, wi in zip(chs, np.power(0.5, ts / hl)):
            credit[c] += wi
    elif rule == "position_based":
        order = np.argsort(ts)
        if len(touches) == 1:
            credit[chs[0]] += 1.0
        else:
            credit[chs[order[0]]] += 0.4
            credit[chs[order[-1]]] += 0.4
            mid = order[1:-1]
            if len(mid):
                for c in chs[mid]:
                    credit[c] += 0.2 / len(mid)
    elif rule == "data_driven_proxy":
        for c in chs:
            credit[c] += 1.0
        credit = 1.0 - np.exp(-credit)
    else:
        raise ValueError(f"unknown rule: {rule}")
    return credit


def generate_source(cfg_source, journeys, channels, rng, shared_noise=None):
    """Produce one source's observed credit rows: list of (cid, channel, source, credit)."""
    K = len(channels)
    idx = {c: i for i, c in enumerate(channels)}
    name = cfg_source["name"]
    vis = set(cfg_source["visibility"])
    vis_idx = np.array([idx[c] for c in vis if c in idx])
    rule = cfg_source["rule"]
    window = float(cfg_source.get("window_days", 1e9))
    half_life = cfg_source.get("half_life_days")
    sp = cfg_source.get("self_preference")
    noise = cfg_source.get("noise") or {}
    miss = float(cfg_source.get("missingness", 0.0))
    group = cfg_source.get("noise_group")
    own = idx.get((sp or {}).get("channel")) if sp else None

    rows = []
    vis_mask = np.zeros(K, dtype=bool)
    vis_mask[vis_idx] = True
    for cid, journey in journeys.items():
        if rng.random() < miss:
            continue
        if rule == "adversarial":
            credit = np.zeros(K)
            credit[vis_idx] = rng.dirichlet(np.ones(len(vis_idx)))
            if own is not None and vis_mask[own]:
                credit[own] += 2.0  # heavy self-dump, ignores the journey
        else:
            touches = [(c, t) for (c, t) in journey if channels[c] in vis and t <= window]
            credit = _apply_rule(rule, touches, K, half_life)
            if sp and own is not None:
                credit[own] *= float(sp.get("beta", 1.0))

        if credit.sum() <= 0:
            continue

        # correlated (group) noise — creates higher-order dependence across a bloc
        if group is not None and shared_noise is not None and group in shared_noise:
            credit = credit * shared_noise[group][cid]
        # independent observation noise
        if noise:
            sigma = float(noise.get("sigma", 0.0))
            if noise.get("kind") == "multiplicative":
                credit = credit * rng.lognormal(0.0, sigma, size=K)
            elif noise.get("kind") == "additive":
                credit = np.clip(credit + rng.normal(0.0, sigma, size=K), 0, None)
        credit[~vis_mask] = 0.0  # noise must not invent credit on invisible channels

        s = credit.sum()
        if s <= 0:
            continue
        credit = credit / s
        for k in range(K):
            if credit[k] > 0:
                rows.append((cid, channels[k], name, float(credit[k])))
    return rows
