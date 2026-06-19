"""Generate the publication figures (manuscript §7). Robust layouts (no overlapping
elements). Seven figures -> results/figures/{png,pdf}.

    py -m src.experiments.figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

from ..synth.generate import generate_tensors
from ..methods import reconcile as rec

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300})
BLUE, ORANGE = "#3b6ea5", "#c0703a"


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}")


def _load(name, n=8000, seed=20260613):
    c = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
    c["ground_truth"]["num_conversions"] = n
    c["seed"] = seed
    return c


# ---------------------------------------------------------------- fig 1
def fig_framework():
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [
        ("Sources\n$\\hat c_s$", "#dfe7f5"),
        ("L1 Fuzzify\n(reliability)", "#cfe3d4"),
        ("L2 Conflict\n(corrected)", "#cfe3d4"),
        ("L3 Choquet\n$+\\,L_1$-closure", "#f3dfcf"),
        ("L4 Mamdani\n(confidence)", "#f3cfd9"),
        ("Reconciled\ncredit + band", "#dfe7f5"),
    ]
    n = len(boxes); w, gap = 0.137, 0.028
    x = (1 - (n * w + (n - 1) * gap)) / 2
    for i, (label, color) in enumerate(boxes):
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.32), w, 0.42, boxstyle="round,pad=0.008",
                                             fc=color, ec="#333", lw=1.1))
        ax.text(x + w / 2, 0.53, label, ha="center", va="center", fontsize=8.5)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.53), xytext=(x + w + 0.004, 0.53),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
        x += w + gap
    ax.set_title("Four-layer fuzzy reconciliation framework", fontsize=11, pad=8)
    _save(fig, "fig1_framework")


# ---------------------------------------------------------------- fig 2
def fig_divergence():
    X, P, T, channels, sources, meta = generate_tensors(_load("synthetic_default.yaml"))
    rows = {"TRUTH": T.sum(axis=0) / T.sum()}
    for s, name in enumerate(sources):
        v = X[:, s, :].sum(axis=0); rows[name] = v / v.sum()
    mat = pd.DataFrame(rows, index=channels).T
    fig, ax = plt.subplots(figsize=(9.5, 4.0), constrained_layout=True)
    mat.plot(kind="bar", ax=ax, width=0.8, legend=False)
    ax.set_ylabel("mean credit share"); ax.set_xlabel("")
    ax.set_title("Sources diverge from ground truth and from each other")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="channel", fontsize=8, loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=True)
    _save(fig, "fig2_divergence")


# ---------------------------------------------------------------- fig 3
def fig_recovery():
    df = pd.read_csv(ROOT / "results" / "p4_grid.csv")
    strong = ["dempster_shafer", "corr_cluster", "rel_wtd_mean"]
    order = ["default", "redundant_bloc", "high_noise", "heavy_bias", "misspec_capacity",
             "adversarial", "complementary", "more_sources_m7"]
    mae_m, kl_m, mae_e, kl_e = [], [], [], []
    for c in order:
        sub = df[df.condition == c]
        means = sub.groupby("method")[["MAE", "KL"]].mean()
        om, ok = means.loc["Choquet(ours)", "MAE"], means.loc["Choquet(ours)", "KL"]
        bms, bks = means.loc[strong, "MAE"].idxmin(), means.loc[strong, "KL"].idxmin()
        bm, bk = means.loc[bms, "MAE"], means.loc[bks, "KL"]
        mae_m.append((bm - om) / bm * 100); kl_m.append((bk - ok) / bk * 100)
        pm = sub.pivot_table(index="seed", columns="method", values="MAE")
        pk = sub.pivot_table(index="seed", columns="method", values="KL")
        dM = (pm[bms] - pm["Choquet(ours)"]) / pm[bms] * 100
        dK = (pk[bks] - pk["Choquet(ours)"]) / pk[bks] * 100
        mae_e.append(1.96 * dM.std(ddof=1) / np.sqrt(len(dM)))
        kl_e.append(1.96 * dK.std(ddof=1) / np.sqrt(len(dK)))
    xp = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.0), constrained_layout=True)
    ax.bar(xp - 0.2, mae_m, 0.4, yerr=mae_e, capsize=3, label="MAE margin", color=BLUE)
    ax.bar(xp + 0.2, kl_m, 0.4, yerr=kl_e, capsize=3, label="KL margin", color=ORANGE)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(xp); ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("improvement vs best strong baseline (%)")
    ax.set_title("Recovery margin over the best strong baseline (mean of 10 seeds; 95% CI)")
    ax.legend()
    _save(fig, "fig3_recovery")


# ---------------------------------------------------------------- fig 4
def fig_calibration():
    from ..eval.calibration import conformal_coverage
    X, P, T, channels, sources, meta = generate_tensors(_load("synthetic_default.yaml", n=7000))
    r = rec.run(X, P)
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    res = conformal_coverage(r["reconciled"], T, r["band_half"], r["valid"], levels=tuple(levels))
    cal = [res[l][1] for l in levels]; raw = res[levels[0]][0]
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.plot([0.45, 1], [0.45, 1], "--", color="#999", label="ideal")
    ax.plot(levels, cal, "o-", color=BLUE, label="conformal-calibrated")
    ax.axhline(raw, color=ORANGE, ls=":", label=f"raw band ($={raw:.2f}$)")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Uncertainty calibration"); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.45, 1); ax.set_ylim(0.45, 1)
    _save(fig, "fig4_calibration")


# ---------------------------------------------------------------- fig 5
def fig_ablation():
    df = pd.read_csv(ROOT / "results" / "p4_ablation.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    xp = np.arange(len(df))
    ax.bar(xp - 0.2, df["MAE_gain_%"], 0.4, label="MAE gain", color=BLUE)
    ax.bar(xp + 0.2, df["KL_gain_%"], 0.4, label="KL gain", color=ORANGE)
    ax.set_xticks(xp); ax.set_xticklabels(df["condition"])
    ax.set_ylabel("gain from non-additivity (%)")
    ax.set_title("Ablation: non-additivity (2-additive vs additive)")
    ax.legend()
    _save(fig, "fig5_ablation")


# ---------------------------------------------------------------- fig 6
def fig_interaction():
    X, P, T, channels, sources, meta = generate_tensors(_load("synthetic_redundant.yaml"))
    r = rec.run(X, P)
    m2, shap = r["m2"], r["shapley"]
    short = [s.replace("indep_", "ind_").replace("tracker_", "trk_") for s in sources]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 0.85]})
    vmax = np.abs(m2).max()
    im = axes[0].imshow(m2, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].set_xticks(range(len(short))); axes[0].set_yticks(range(len(short)))
    axes[0].set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    axes[0].set_yticklabels(short, fontsize=8)
    axes[0].set_title("Interaction indices $m_{st}$\n(negative = redundant)", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].barh(range(len(short)), shap, color=BLUE)
    axes[1].set_yticks(range(len(short))); axes[1].set_yticklabels(short, fontsize=8)
    axes[1].invert_yaxis(); axes[1].set_xlabel("Shapley value")
    axes[1].set_title("Source importance (Shapley)", fontsize=10)
    _save(fig, "fig6_interaction")


# ---------------------------------------------------------------- fig 7
def fig_sensitivity():
    db = pd.read_csv(ROOT / "results" / "p4_sensitivity_beta.csv")
    dm = pd.read_csv(ROOT / "results" / "p4_sensitivity_m.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for cond, g in db.groupby("condition"):
        ax[0].plot(g["beta"], g["margin_MAE_pct"], "o-", label=cond)
    ax[0].axhline(0, color="#999", lw=0.8)
    ax[0].set_xlabel(r"interaction strength $\beta$")
    ax[0].set_ylabel("MAE margin vs best strong (%)")
    ax[0].set_title(r"Robustness to $\beta$"); ax[0].legend(fontsize=8)
    ax[1].plot(dm["m_sources"], dm["margin_MAE_pct"], "s-", color=ORANGE)
    ax[1].axhline(0, color="#999", lw=0.8)
    ax[1].set_xlabel("number of sources $m$ (growing redundant bloc)")
    ax[1].set_ylabel("MAE margin vs best strong (%)")
    ax[1].set_title("Scaling with $m$")
    _save(fig, "fig7_sensitivity")


def main():
    print("Generating figures ->", FIG)
    fig_framework(); fig_divergence(); fig_recovery(); fig_calibration()
    fig_ablation(); fig_interaction(); fig_sensitivity()
    print("done.")


if __name__ == "__main__":
    main()
