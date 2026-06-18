"""Sensitivity analysis (manuscript §7): robustness to the interaction strength beta and
scaling with the number of sources m. Multi-seed; reports the method's MAE margin over the
best strong baseline. Writes results/p4_sensitivity_{beta,m}.csv and results/figures/fig7_sensitivity.{png,pdf}.

    py -m src.experiments.sensitivity
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..synth.generate import generate_tensors
from ..methods import reconcile as rec
from ..baselines import baselines as bl
from ..eval import metrics as M

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260613, 7, 101, 11, 23, 47, 89, 137, 199, 313]
N = 7000
STRONG = ["dempster_shafer", "corr_cluster", "rel_wtd_mean"]


def _load(name):
    c = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
    c["ground_truth"]["num_conversions"] = N
    return c


def _eval(cfg, beta, seed):
    """Per-seed MAE of ours and of each strong baseline on the common mask."""
    cfg = copy.deepcopy(cfg); cfg["seed"] = seed
    X, P, T, ch, src, meta = generate_tensors(cfg)
    r = rec.run(X, P, beta=beta)
    strong = {
        "dempster_shafer": bl.dempster_shafer(X, P, reliability=r["reliability"]),
        "corr_cluster": bl.correlation_cluster_average(X, P, redundancy=r["redundancy"], threshold=0.8),
        "rel_wtd_mean": bl.reliability_weighted_mean(X, P, reliability=r["reliability"]),
    }
    common = r["valid"].copy()
    for _, v in strong.values():
        common &= v
    ours = M.mae(r["reconciled"][common], T[common])
    smae = {k: M.mae(p[common], T[common]) for k, (p, _) in strong.items()}
    return ours, smae, len(src)


def _agg_margin(cfg, beta):
    """Aggregate margin (same definition as run_synth): mean over seeds, then margin vs the
    best-on-average strong baseline."""
    ours, smae, m = [], {k: [] for k in STRONG}, None
    for s in SEEDS:
        o, sd, m = _eval(cfg, beta, s)
        ours.append(o)
        for k in STRONG:
            smae[k].append(sd[k])
    mo = float(np.mean(ours)); best = min(float(np.mean(smae[k])) for k in STRONG)
    return (best - mo) / best * 100, m


def beta_sweep():
    betas = [0.2, 0.35, 0.5, 0.65, 0.8]
    rows = []
    for cond in ["default", "redundant_bloc"]:
        cfg = _load("synthetic_default.yaml" if cond == "default" else "synthetic_redundant.yaml")
        for b in betas:
            marg, _ = _agg_margin(cfg, b)
            rows.append({"condition": cond, "beta": b, "margin_MAE_pct": float(marg)})
            print(f"[beta] {cond} beta={b}: margin={marg:+.1f}%")
    return pd.DataFrame(rows)


def _redundant_cfg_with_bloc(bloc):
    base = _load("synthetic_redundant.yaml")
    trackers = []
    for i in range(bloc):
        trackers.append({"name": f"tracker_{i}", "visibility": ["meta", "google", "organic"],
                         "rule": "last_touch", "window_days": 7, "self_preference": None,
                         "noise": {"kind": "multiplicative", "sigma": 0.10}, "missingness": 0.02,
                         "noise_group": 1})
    indep = [s for s in base["sources"] if s["name"].startswith("indep")]
    base["sources"] = trackers + indep
    return base


def m_sweep():
    rows = []
    for bloc in range(1, 6):           # bloc 1..5 + 2 independent => m = 3..7
        cfg = _redundant_cfg_with_bloc(bloc)
        marg, m = _agg_margin(cfg, 0.5)
        rows.append({"m_sources": m, "bloc_size": bloc, "margin_MAE_pct": float(marg)})
        print(f"[m] bloc={bloc} m={m}: margin={marg:+.1f}%")
    return pd.DataFrame(rows)


def main():
    db = beta_sweep()
    dm = m_sweep()
    db.to_csv(ROOT / "results" / "p4_sensitivity_beta.csv", index=False)
    dm.to_csv(ROOT / "results" / "p4_sensitivity_m.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for cond, g in db.groupby("condition"):
        ax[0].plot(g["beta"], g["margin_MAE_pct"], "o-", label=cond)
    ax[0].axhline(0, color="#999", lw=0.8); ax[0].set_xlabel(r"interaction strength $\beta$")
    ax[0].set_ylabel("MAE margin vs best strong (%)"); ax[0].set_title("Robustness to $\\beta$"); ax[0].legend(fontsize=8)
    ax[1].plot(dm["m_sources"], dm["margin_MAE_pct"], "s-", color="#c0703a")
    ax[1].axhline(0, color="#999", lw=0.8); ax[1].set_xlabel("number of sources $m$ (growing redundant bloc)")
    ax[1].set_ylabel("MAE margin vs best strong (%)"); ax[1].set_title("Scaling with $m$")
    plt.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "fig7_sensitivity.png", dpi=300)
    fig.savefig(ROOT / "results" / "figures" / "fig7_sensitivity.pdf")
    plt.close(fig)
    print("wrote sensitivity CSVs + fig7_sensitivity")


if __name__ == "__main__":
    main()
