"""Extra evaluation for the manuscript: (i) a full-baseline comparison on the default
condition across five metrics plus runtime, and (ii) conformal calibration coverage across
all eight conditions. Writes results/p4_default_fullbaseline.csv and
results/p4_calibration_conditions.csv.

    py -m src.experiments.extra_eval
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..synth.generate import generate_tensors
from ..methods import reconcile as rec
from ..baselines import baselines as bl
from ..eval import metrics as M
from ..eval.calibration import conformal_coverage
from .run_synth import conditions

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260613


def full_baseline_default():
    conds = dict(conditions())
    cfg = conds["default"]; cfg = {**cfg, "seed": SEED}
    X, P, T, ch, src, meta = generate_tensors(cfg)

    t0 = time.perf_counter(); r = rec.run(X, P); t_ours = (time.perf_counter() - t0) * 1e3

    def timed(fn):
        t = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t) * 1e3

    methods = {"Choquet (ours)": ((r["reconciled"], r["valid"]), t_ours)}
    methods["Sugeno (ours)"] = timed(lambda: bl.sugeno(X, P, m1=r["m1"], m2=r["m2"]))
    methods["Dempster-Shafer"] = timed(lambda: bl.dempster_shafer(X, P, reliability=r["reliability"]))
    methods["corr-cluster"] = timed(lambda: bl.correlation_cluster_average(X, P, redundancy=r["redundancy"], threshold=0.8))
    methods["reliability-wtd mean"] = timed(lambda: bl.reliability_weighted_mean(X, P, reliability=r["reliability"]))
    methods["trimmed mean"] = timed(lambda: bl.trimmed_mean(X, P))
    methods["single-source trust"] = timed(lambda: bl.single_source_trust(X, P, reliability=r["reliability"]))
    methods["naive mean"] = timed(lambda: bl.naive_mean(X, P))
    methods["median"] = timed(lambda: bl.median_agg(X, P))

    common = r["valid"].copy()
    for (pred, v), _ in methods.values():
        common &= v
    rows = []
    for name, ((pred, v), ms) in methods.items():
        p, t = pred[common], T[common]
        rows.append({"method": name, "MAE": M.mae(p, t), "RMSE": M.rmse(p, t),
                     "KL": M.kl_divergence(p, t), "EMD": M.earth_mover(p, t),
                     "Spearman": M.ranking_spearman(p, t), "runtime_ms": ms})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "p4_default_fullbaseline.csv", index=False)
    print("=== Full baseline on the default condition (seed %d, n=%d) ===" % (SEED, int(common.sum())))
    print(df.round(4).to_string(index=False))
    return df


def calibration_all_conditions():
    rows = []
    for name, cfg0 in conditions():
        cfg = {**cfg0, "seed": SEED}
        X, P, T, ch, src, meta = generate_tensors(cfg)
        r = rec.run(X, P)
        res = conformal_coverage(r["reconciled"], T, r["band_half"], r["valid"], levels=(0.80, 0.90))
        rows.append({"condition": name, "cov80": res[0.80][1], "cov90": res[0.90][1]})
        print(f"[calib] {name}: 80%->{res[0.80][1]:.3f}  90%->{res[0.90][1]:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "p4_calibration_conditions.csv", index=False)
    return df


def main():
    full_baseline_default()
    print()
    calibration_all_conditions()
    print("\nwrote results/p4_default_fullbaseline.csv + results/p4_calibration_conditions.csv")


if __name__ == "__main__":
    main()
