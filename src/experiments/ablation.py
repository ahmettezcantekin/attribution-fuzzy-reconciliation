"""Ablation study (plan §5.4, §6 gate #5, manuscript §7.4).

Removes each contribution and measures the drop, on the SAME configuration as the main grid
(run_synth.py): N=7000 conversions, three seeds (mean), adversary-gated elicited capacity, and
the same all-methods intersection mask. This makes the ablation's "full" column identical to the
"Choquet (ours)" column of the main recovery tables.

  - non-additivity: force the capacity additive (m2 = 0) -> Choquet collapses to the reliability-
    weighted mean. Effect on MAE/KL (point estimate), mean over seeds.
  - Mamdani layer: replace adaptive confidence with a constant -> band efficiency (mean calibrated
    interval width at fixed nominal 90% coverage; lower = sharper/adaptive).

    py -m src.experiments.ablation
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..synth.generate import generate_tensors
from ..methods import reconcile as rec, choquet as cq, mamdani as mam
from ..baselines import baselines as bl
from ..eval import metrics as M
from ..eval.calibration import conformal_coverage

ROOT = Path(__file__).resolve().parents[2]
N = 7000                       # match run_synth.N_GRID
SEEDS = [20260613, 7, 101, 11, 23, 47, 89, 137, 199, 313]     # match run_synth.SEEDS
BETA = 0.5


def _load(name):
    c = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
    c["ground_truth"]["num_conversions"] = N
    return c


def _common_mask(X, P, r):
    """Intersection of all methods' valid masks — identical basis to run_synth.run_condition."""
    masks = [
        r["valid"],
        bl.naive_mean(X, P)[1],
        bl.reliability_weighted_mean(X, P, reliability=r["reliability"])[1],
        bl.median_agg(X, P)[1],
        bl.dempster_shafer(X, P, reliability=r["reliability"])[1],
        bl.correlation_cluster_average(X, P, redundancy=r["redundancy"], threshold=0.8)[1],
        bl.sugeno(X, P, m1=r["m1"], m2=r["m2"])[1],
    ]
    c = np.ones(X.shape[0], dtype=bool)
    for m in masks:
        c &= m
    return c


def point_ablation_seed(cfg, seed):
    cfg = copy.deepcopy(cfg); cfg["seed"] = seed
    X, P, T, ch, src, meta = generate_tensors(cfg)
    r = rec.run(X, P, beta=BETA)                       # full = main "Choquet (ours)" (gate on)
    full = r["reconciled"]
    add, _ = cq.reconcile_choquet(X, P, r["m1"], np.zeros_like(r["m2"]))   # additive: m2 = 0
    c = _common_mask(X, P, r)
    return {
        "full_MAE": M.mae(full[c], T[c]), "full_KL": M.kl_divergence(full[c], T[c]),
        "additive_MAE": M.mae(add[c], T[c]), "additive_KL": M.kl_divergence(add[c], T[c]),
    }


def band_ablation_seed(cfg, seed):
    cfg = copy.deepcopy(cfg); cfg["seed"] = seed
    X, P, T, ch, src, meta = generate_tensors(cfg)
    r = rec.run(X, P, beta=BETA)
    rec_full, valid = r["reconciled"], r["valid"]
    half_full = r["band_half"]
    half_const = mam.band_halfwidth(np.full_like(r["confidence"], 0.5))    # constant confidence

    def cov_and_width(half):
        raw, cov, q = conformal_coverage(rec_full, T, half, valid, levels=(0.90,))[0.90]
        width = float((q * np.maximum(half[valid], 1e-6)).mean()) * 2
        return cov, width

    cov_f, w_f = cov_and_width(half_full)
    cov_c, w_c = cov_and_width(half_const)
    return {"mamdani_cov90": cov_f, "mamdani_width": w_f, "constant_cov90": cov_c, "constant_width": w_c}


def main():
    conds = [("default", _load("synthetic_default.yaml")),
             ("redundant", _load("synthetic_redundant.yaml"))]
    rows = []
    for name, cfg in conds:
        seed_rows = [point_ablation_seed(cfg, sd) for sd in SEEDS]
        avg = {k: float(np.mean([sr[k] for sr in seed_rows])) for k in seed_rows[0]}
        rows.append({"condition": name, **avg})
    df = pd.DataFrame(rows)
    df["MAE_gain_%"] = (df["additive_MAE"] - df["full_MAE"]) / df["additive_MAE"] * 100
    df["KL_gain_%"] = (df["additive_KL"] - df["full_KL"]) / df["additive_KL"] * 100

    braw = [band_ablation_seed(_load("synthetic_default.yaml"), sd) for sd in SEEDS]
    ba = {k: float(np.mean([b[k] for b in braw])) for k in braw[0]}

    print("=== Non-additivity ablation (full 2-additive vs additive m2=0; mean of 3 seeds, N=7000, gate on) ===")
    print(df.round(4).to_string(index=False))
    print("\n=== Mamdani ablation (band efficiency at nominal 90% coverage; mean of 10 seeds) ===")
    print(f"  Mamdani-adaptive : coverage={ba['mamdani_cov90']:.3f}  mean width={ba['mamdani_width']:.3f}")
    print(f"  constant-conf    : coverage={ba['constant_cov90']:.3f}  mean width={ba['constant_width']:.3f}")
    verdict = "efficiency" if ba["mamdani_width"] < ba["constant_width"] else "interpretability (not efficiency)"
    print(f"  => the Mamdani layer earns its place on: {verdict}")

    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "p4_ablation.csv", index=False)
    print(f"\nWrote {ROOT / 'results' / 'p4_ablation.csv'}")


if __name__ == "__main__":
    main()
