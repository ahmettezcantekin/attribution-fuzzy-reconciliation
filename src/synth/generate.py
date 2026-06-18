"""Top-level synthetic generation entry point (plan §4.3).

ground_truth -> per-source distortion -> long table + truth + run metadata. Supports:
  - experiment.adversarial_source: inject a heavily-biased adversarial source (robustness);
  - per-source noise_group: correlated noise across a bloc (higher-order dependence);
and exposes in-memory tensor builders for the P4 grid harness.

Run:
    py -m src.synth.generate --config config/synthetic_default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .ground_truth import generate_ground_truth
from .sources import generate_source


def _channels(cfg: dict) -> list[str]:
    chans: set[str] = set()
    for s in cfg["sources"]:
        chans.update(s["visibility"])
    return sorted(chans)


def _maybe_adversarial(cfg: dict, channels: list[str]) -> list[dict]:
    sources = [dict(s) for s in cfg["sources"]]
    if cfg.get("experiment", {}).get("adversarial_source"):
        sources.append({
            "name": "adversarial", "visibility": list(channels), "rule": "adversarial",
            "window_days": 9999, "self_preference": {"channel": channels[0], "beta": 1.0},
            "noise": {"kind": "multiplicative", "sigma": 0.4}, "missingness": 0.0,
        })
    return sources


def generate(cfg: dict):
    """Return (sources_long_df, truth_df, run_meta)."""
    rng = np.random.default_rng(int(cfg["seed"]))
    channels = _channels(cfg)
    truth_df, journeys, channel_lift = generate_ground_truth(cfg, rng, channels)
    sources_cfg = _maybe_adversarial(cfg, channels)
    K, N = len(channels), len(journeys)

    # precompute correlated noise per group (higher-order dependence)
    groups = {s["noise_group"] for s in sources_cfg if s.get("noise_group") is not None}
    sigma_g = float(cfg.get("experiment", {}).get("group_noise_sigma", 0.3))
    shared_noise = {g: rng.lognormal(0.0, sigma_g, size=(N, K)) for g in groups}

    rows = []
    for s in sources_cfg:
        rows += generate_source(s, journeys, channels, rng, shared_noise=shared_noise)
    sources_long = pd.DataFrame(rows, columns=["conversion_id", "channel", "source", "credit"])

    run_meta = {
        "seed": int(cfg["seed"]), "channels": channels,
        "channel_lift_hidden": [float(x) for x in channel_lift],
        "num_conversions": N, "sources": [s["name"] for s in sources_cfg],
        "config_snapshot": cfg,
    }
    return sources_long, truth_df, run_meta


def build_tensors(sources_long, truth_df, channels, sources):
    """Long dfs -> (X (N,S,K), P (N,S) bool, T (N,K), conv_ids)."""
    ch_idx = {c: i for i, c in enumerate(channels)}
    s_idx = {s: i for i, s in enumerate(sources)}
    conv_ids = np.array(sorted(set(sources_long["conversion_id"]) | set(truth_df["conversion_id"])))
    cid_row = {c: i for i, c in enumerate(conv_ids)}
    N, S, K = len(conv_ids), len(sources), len(channels)
    X = np.zeros((N, S, K)); P = np.zeros((N, S), dtype=bool); T = np.zeros((N, K))
    if len(sources_long):
        r = sources_long["conversion_id"].map(cid_row).to_numpy()
        X[r, sources_long["source"].map(s_idx).to_numpy(), sources_long["channel"].map(ch_idx).to_numpy()] = sources_long["credit"].to_numpy()
        P[r, sources_long["source"].map(s_idx).to_numpy()] = True
    tr = truth_df["conversion_id"].map(cid_row).to_numpy()
    T[tr, truth_df["channel"].map(ch_idx).to_numpy()] = truth_df["true_credit"].to_numpy()
    return X, P, T, conv_ids


def generate_tensors(cfg: dict):
    """In-memory generation -> (X, P, T, channels, sources, run_meta) for the grid harness."""
    sl, td, meta = generate(cfg)
    X, P, T, _ = build_tensors(sl, td, meta["channels"], meta["sources"])
    return X, P, T, meta["channels"], meta["sources"], meta


def divergence_report(sources_long, truth_df, channels):
    def agg(df, val):
        n = df["conversion_id"].nunique()
        v = df.groupby("channel")[val].sum().reindex(channels).fillna(0.0) / n
        return v / v.sum()
    tab = {"TRUTH": agg(truth_df, "true_credit")}
    for name, g in sources_long.groupby("source"):
        tab[name] = agg(g, "credit")
    mat = pd.DataFrame(tab).T[channels]
    names = [n for n in mat.index if n != "TRUTH"]
    pair, npair = 0.0, 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair += np.abs(mat.loc[names[i]] - mat.loc[names[j]]).sum(); npair += 1
    to_truth = {s: float(np.abs(mat.loc[s] - mat.loc["TRUTH"]).sum()) for s in names}
    return mat, pair / max(1, npair), to_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/synthetic_default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sl, td, meta = generate(cfg)
    out = cfg["output"]
    for p in (out["long_table"], out["ground_truth"], out["run_metadata"]):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    sl.to_csv(out["long_table"], index=False)
    td.to_csv(out["ground_truth"], index=False)
    Path(out["run_metadata"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mat, pairwise, to_truth = divergence_report(sl, td, meta["channels"])
    print("\nAggregate credit share (rows=source, cols=channel):")
    print(mat.round(3).to_string())
    print(f"\nMean pairwise L1 between sources: {pairwise:.3f}")
    print(f"Wrote {out['long_table']} ({len(sl)} rows), {out['ground_truth']}, {out['run_metadata']}")


if __name__ == "__main__":
    main()
