"""End-to-end reconciliation pipeline (plan §1, §5.1, manuscript §2).

    load synth -> tensors -> elicited capacity -> Choquet (+L1-closure)
                -> conflict -> Mamdani confidence/band -> output + diagnostics

Run on the synthetic data and report recovery vs baselines (P3 exit criterion):
    py -m src.methods.reconcile
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import capacity as cap
from . import choquet as cq
from . import conflict as cf
from . import mamdani as mam
from ..baselines import baselines as bl
from ..eval import metrics as M


def load_tensors(results_dir: Path):
    """Build (X, P, T, conv_ids, sources, channels) from the synthetic CSVs."""
    from ..synth.generate import build_tensors
    src = pd.read_csv(results_dir / "synth_sources.csv")
    truth = pd.read_csv(results_dir / "synth_truth.csv")
    meta = json.loads((results_dir / "synth_run_meta.json").read_text(encoding="utf-8"))
    channels, sources = meta["channels"], meta["sources"]
    X, P, T, conv_ids = build_tensors(src, truth, channels, sources)
    return X, P, T, conv_ids, sources, channels


def run(X, P, beta: float = 0.5, beta3: float = 0.5, robust: bool = False,
        reliability_mode: str = "coverage", kadditive: bool = False, adversary_gate: bool = True):
    """Run the full pipeline. Returns dict with reconciled, band, conflict, confidence, capacity.

    reliability_mode: "coverage" (coverage+report-rate + adversary gate — the locked default)
    or "corr" (pure consensus-correlation; boosts redundant sources, kept for comparison).
    kadditive=True uses an up-to-3-additive capacity (captures higher-order redundancy);
    beta/beta3 weight the pairwise/triple interaction strength."""
    if reliability_mode == "corr":
        reliability = cap.correlation_reliability(X, P)
    else:
        reliability = cap.observable_reliability(X, P, robust=robust, adversary_gate=adversary_gate)
    redundancy = cap.observable_redundancy(X, P)          # coverage-Jaccard (shared lens)

    if kadditive:
        T3 = cap.structural_triple(X, P)
        m1, m2, m3 = cap.build_kadditive(reliability, redundancy, T3, beta2=beta, beta3=beta3)
        reconciled, valid = cq.reconcile_kadditive(X, P, m1, m2, m3)
    else:
        m1, m2 = cap.build_2additive(reliability, -redundancy, beta=beta)
        m3 = {}
        reconciled, valid = cq.reconcile_choquet(X, P, m1, m2)

    raw_conflict = cf.cell_conflict(X, P, redundancy)          # in [0,2]
    disagreement = np.clip(raw_conflict / 2.0, 0, 1)
    # mean reliability of reporting sources per cell
    rel_cell = (P * reliability[None, :]).sum(axis=1) / np.maximum(P.sum(axis=1), 1)
    conf = mam.confidence(disagreement, rel_cell)
    half = mam.band_halfwidth(conf)

    return {
        "reconciled": reconciled, "valid": valid, "conflict": raw_conflict,
        "confidence": conf, "band_half": half, "reliability": reliability,
        "redundancy": redundancy, "m1": m1, "m2": m2, "m3": m3,
        "shapley": cap.shapley_values(m1, m2),
    }


def _score(name, pred, truth, valid):
    p, t = pred[valid], truth[valid]
    return {"method": name, "MAE": M.mae(p, t), "RMSE": M.rmse(p, t),
            "KL": M.kl_divergence(p, t), "Spearman": M.ranking_spearman(p, t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()
    rd = Path(args.results)

    X, P, T, conv_ids, sources, channels = load_tensors(rd)
    print(f"Loaded {X.shape[0]} cells, {len(sources)} sources, {len(channels)} channels.")

    out = run(X, P, beta=args.beta)
    valid = out["valid"]

    print("\nElicited reliability (observable):")
    for s, r in zip(sources, out["reliability"]):
        print(f"  {s:<22} r={r:.3f}  shapley={out['shapley'][list(sources).index(s)]:.3f}")
    print("Interaction indices (Möbius pairs, negative => redundant/sub-additive):")
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            if abs(out["m2"][i, j]) > 1e-4:
                print(f"  I({sources[i]},{sources[j]}) = {out['m2'][i, j]:+.3f}")

    # baselines
    nm, nmv = bl.naive_mean(X, P)
    rw, rwv = bl.reliability_weighted_mean(X, P, reliability=out["reliability"])
    md, mdv = bl.median_agg(X, P)

    common = valid & nmv & rwv & mdv
    rows = [
        _score("Choquet (ours)", out["reconciled"], T, common),
        _score("naive mean", nm, T, common),
        _score("reliability-wtd mean", rw, T, common),
        _score("median", md, T, common),
    ]
    df = pd.DataFrame(rows).set_index("method")
    print(f"\nRecovery vs ground truth (n={int(common.sum())} cells):")
    print(df.round(4).to_string())

    # mean confidence/band sanity
    print(f"\nMean confidence={out['confidence'][valid].mean():.3f}, "
          f"mean band half-width={out['band_half'][valid].mean():.3f}")

    rd.mkdir(exist_ok=True)
    df.to_csv(rd / "p3_recovery.csv")
    print(f"\nWrote {rd / 'p3_recovery.csv'}")


if __name__ == "__main__":
    main()
