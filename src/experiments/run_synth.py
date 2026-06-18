"""P4 experiment battery on synthetic data (plan §5.4, §6).

Sweeps a grid of conditions; for each, generates data in-memory, runs the method (elicited
capacity) and all baselines (weak + strong), computes recovery vs ground truth, and records
win/margin of the method vs the best STRONG baseline. Writes results/p4_grid.csv and prints
a GATE-relevant summary.

    py -m src.experiments.run_synth
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..synth.generate import generate_tensors
from ..methods import reconcile as rec
from ..baselines import baselines as bl
from ..eval import metrics as M

ROOT = Path(__file__).resolve().parents[2]
N_GRID = 7000  # per-condition conversions
SEEDS = [20260613, 7, 101, 11, 23, 47, 89, 137, 199, 313]  # 10-seed: mean + CI, guards against seed-overfit
BETA = 0.5


def _load(name):
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def _scale_noise(cfg, factor):
    for s in cfg["sources"]:
        if s.get("noise"):
            s["noise"]["sigma"] = float(s["noise"]["sigma"]) * factor
    return cfg


def conditions():
    base = _load("synthetic_default.yaml"); base["ground_truth"]["num_conversions"] = N_GRID
    red = _load("synthetic_redundant.yaml"); red["ground_truth"]["num_conversions"] = N_GRID
    out = []

    out.append(("default", copy.deepcopy(base)))
    out.append(("redundant_bloc", copy.deepcopy(red)))

    adv = copy.deepcopy(base); adv.setdefault("experiment", {})["adversarial_source"] = True
    out.append(("adversarial", adv))

    hn = _scale_noise(copy.deepcopy(base), 2.0); out.append(("high_noise", hn))

    # complementary: make visibility sets more disjoint (less redundancy)
    comp = copy.deepcopy(base)
    vis = [["meta", "organic"], ["applovin", "organic"], ["google", "organic"],
           ["meta", "applovin", "other"], ["google", "other"]]
    for s, v in zip(comp["sources"], vis):
        s["visibility"] = v
    out.append(("complementary", comp))

    # heavy self-preferencing bias
    hb = copy.deepcopy(base)
    for s in hb["sources"]:
        if s.get("self_preference"):
            s["self_preference"]["beta"] = 2.2
    out.append(("heavy_bias", hb))

    # misspecification (capacity): broad-visibility source is NOISY, a narrow one is ACCURATE
    mis = copy.deepcopy(base)
    mis["sources"][3]["noise"] = {"kind": "multiplicative", "sigma": 0.5}   # singular (broad) -> noisy
    mis["sources"][1]["rule"] = "data_driven_proxy"                          # applovin (narrow) -> accurate
    mis["sources"][1]["noise"] = {"kind": "multiplicative", "sigma": 0.05}
    mis.setdefault("experiment", {})["misspecification"] = "capacity"
    out.append(("misspec_capacity", mis))

    # more sources (m=7): bigger redundant bloc
    more = copy.deepcopy(red)
    more["sources"] += [
        {"name": "tracker_d", "visibility": ["meta", "google", "organic"], "rule": "last_touch",
         "window_days": 7, "self_preference": None, "noise": {"kind": "multiplicative", "sigma": 0.10},
         "missingness": 0.02, "noise_group": 1},
        {"name": "tracker_e", "visibility": ["meta", "google", "organic"], "rule": "last_touch",
         "window_days": 7, "self_preference": None, "noise": {"kind": "multiplicative", "sigma": 0.10},
         "missingness": 0.02, "noise_group": 1},
    ]
    out.append(("more_sources_m7", more))
    return out


def _score(pred, T, valid):
    p, t = pred[valid], T[valid]
    return M.mae(p, t), M.kl_divergence(p, t)


def run_condition(name, cfg, seed, beta=BETA):
    cfg = copy.deepcopy(cfg); cfg["seed"] = seed
    X, P, T, channels, sources, meta = generate_tensors(cfg)
    r = rec.run(X, P, beta=beta)
    methods = {
        "Choquet(ours)": (r["reconciled"], r["valid"]),
        "naive_mean": bl.naive_mean(X, P),
        "rel_wtd_mean": bl.reliability_weighted_mean(X, P, reliability=r["reliability"]),
        "median": bl.median_agg(X, P),
        "dempster_shafer": bl.dempster_shafer(X, P, reliability=r["reliability"]),
        "corr_cluster": bl.correlation_cluster_average(X, P, redundancy=r["redundancy"], threshold=0.8),
        "sugeno": bl.sugeno(X, P, m1=r["m1"], m2=r["m2"]),
    }
    common = np.ones(X.shape[0], dtype=bool)
    for _, v in methods.values():
        common &= v
    rows = []
    for mname, (pred, _) in methods.items():
        mae, kl = _score(pred, T, common)
        rows.append({"condition": name, "seed": seed, "method": mname, "MAE": mae, "KL": kl,
                     "n": int(common.sum()), "m_sources": len(sources)})
    return rows


def main(beta=BETA, seeds=SEEDS):
    all_rows = []
    for name, cfg in conditions():
        print(f"[grid] {name} (seeds={seeds}, beta={beta}) ...")
        for sd in seeds:
            all_rows += run_condition(name, cfg, sd, beta=beta)
    df = pd.DataFrame(all_rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "p4_grid.csv", index=False)

    # mean over seeds, then method vs best STRONG baseline per condition
    # Sugeno uses OUR elicited capacity, so it is a FRAMEWORK VARIANT, not an external
    # baseline. External strong baselines are DS / correlation-cluster / reliability-wtd mean.
    strong = ["dempster_shafer", "corr_cluster", "rel_wtd_mean"]
    agg = df.groupby(["condition", "method"])[["MAE", "KL"]].mean().reset_index()
    print("\n=== P4 grid summary (mean over seeds; MAE/KL margin vs best strong) ===")
    wins_mae = wins_kl = 0; order = [c for c, _ in conditions()]
    for cond in order:
        gi = agg[agg.condition == cond].set_index("method")
        ours_mae, ours_kl = gi.loc["Choquet(ours)", "MAE"], gi.loc["Choquet(ours)", "KL"]
        best_s_mae = gi.loc[strong, "MAE"].min(); best_s_kl = gi.loc[strong, "KL"].min()
        marg_mae = (best_s_mae - ours_mae) / best_s_mae * 100
        marg_kl = (best_s_kl - ours_kl) / best_s_kl * 100
        wins_mae += marg_mae > 0; wins_kl += marg_kl > 0
        print(f"  {cond:<18} ours MAE={ours_mae:.4f}  bestStrong={best_s_mae:.4f}  "
              f"marginMAE={marg_mae:+.1f}%  marginKL={marg_kl:+.1f}%")
    nC = len(order)
    print(f"\nMethod beats best strong baseline (MAE) in {wins_mae}/{nC}; (KL) in {wins_kl}/{nC}.")

    # framework operator comparison: Choquet vs Sugeno (both on the elicited capacity)
    print("\n=== Operator comparison within the framework (Choquet vs Sugeno) ===")
    for cond in order:
        gi = agg[agg.condition == cond].set_index("method")
        if "sugeno" in gi.index:
            print(f"  {cond:<18} Choquet MAE={gi.loc['Choquet(ours)','MAE']:.4f}  "
                  f"Sugeno MAE={gi.loc['sugeno','MAE']:.4f}  "
                  f"Choquet KL={gi.loc['Choquet(ours)','KL']:.4f}  Sugeno KL={gi.loc['sugeno','KL']:.4f}")
    print(f"Wrote {ROOT / 'results' / 'p4_grid.csv'}")
    return df


if __name__ == "__main__":
    main()
