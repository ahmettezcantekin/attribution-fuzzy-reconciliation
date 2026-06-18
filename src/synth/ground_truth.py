"""Ground-truth model (plan §4.1).

For each conversion we generate a touchpoint journey and assign a TRUE credit vector c*
over K channels (sums to 1) via a LATENT-INCREMENTALITY process: each channel has a hidden
true lift, and credit is the channel's saturating (diminishing-returns) incremental
contribution along the realized journey.

THIS IS THE ONLY PLACE GROUND TRUTH LIVES. The reconciliation method (src/methods) must
never import the hidden lifts or the journey. Anti-circularity (plan §4.0): this generating
mechanism (latent lift x saturating counts) is deliberately NOT any rule the method assumes
(last/first/time-decay/position), nor does it expose source redundancy as a parameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ground_truth(cfg: dict, rng: np.random.Generator, channels: list[str]):
    """Return (truth_df, journeys, channel_lift).

    truth_df : DataFrame[conversion_id, channel, true_credit]  (true_credit sums to 1 / cid)
    journeys : dict cid -> list of (channel_idx, days_before_conversion)
    channel_lift : hidden per-channel lift vector (kept out of the method)
    """
    gt = cfg["ground_truth"]
    K = len(channels)
    n = int(gt["num_conversions"])
    lam = float(gt["journey_length"]["lam"])
    conc = float(gt.get("lift_concentration", 0.6))

    # hidden truth: per-channel latent lift + a touch-popularity prior (distinct objects)
    channel_lift = rng.dirichlet(np.full(K, conc))
    touch_prior = rng.dirichlet(np.full(K, 1.0))

    rows, journeys = [], {}
    for cid in range(n):
        L = max(1, int(rng.poisson(lam)))
        touch_ch = rng.choice(K, size=L, p=touch_prior)
        # days before conversion; smaller = more recent (drives recency/window/decay in sources)
        times = np.sort(rng.exponential(scale=3.0, size=L))[::-1]
        journeys[cid] = list(zip(touch_ch.tolist(), times.tolist()))

        # TRUE credit = latent lift x saturating returns on per-channel touch counts
        counts = np.bincount(touch_ch, minlength=K).astype(float)
        contrib = channel_lift * (1.0 - np.exp(-counts))
        if contrib.sum() <= 0:
            contrib = channel_lift.copy()
        cstar = contrib / contrib.sum()
        for k in range(K):
            if cstar[k] > 0:
                rows.append((cid, channels[k], float(cstar[k])))

    truth_df = pd.DataFrame(rows, columns=["conversion_id", "channel", "true_credit"])
    return truth_df, journeys, channel_lift
