"""Generate two sensitivity figures from data/lcb_calibration_v2/<gen>/sensitivity.json:

  fig_cver_sweep.png       — utility vs c_ver curves, one panel per generator
  fig_theta_sensitivity.png — Bayesian Δ at clean / ±10% / ±20% perturbations

Usage:
  python3 scripts/fig_sensitivity.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}

POLICIES = [
    ("bayesian_greedy", "bayes_greedy",  "#1d4ed8", "-"),
    ("bayesian_DP",     "bayes_DP",      "#0ea5e9", "-"),
    ("threshold_L2",    "threshold(L2)", "#22c55e", "-"),
    ("threshold_L0",    "threshold(L0)", "#94a3b8", "--"),
    ("threshold_L3",    "threshold(L3)", "#a855f7", "--"),
    ("always_verify",   "always_verify", "black",   ":"),
    ("best_of_3",       "best_of_3",     "#f59e0b", "--"),
    ("fixed_pipeline",  "fixed_pipeline","#ef4444", "--"),
]


def fig_cver_sweep(args):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, gen in enumerate(GENERATORS):
        path = args.data_root / "lcb_calibration_v2" / gen / "sensitivity.json"
        if not path.exists():
            axes[i].text(0.5, 0.5, f"{gen}\n(no data)", ha="center", va="center",
                         transform=axes[i].transAxes, fontsize=12)
            axes[i].set_axis_off()
            continue
        d = json.loads(path.read_text())
        d2 = d.get("D2_c_ver_sweep", {})
        # Extract c_ver values (sorted numerically) and policy curves
        cvers = sorted([int(k.split("_")[-1]) for k in d2.keys()])
        ax = axes[i]
        for pol_key, pol_label, color, style in POLICIES:
            ys = []
            for cv in cvers:
                v = d2[f"c_ver_{cv}"].get(pol_key, {}).get("mean_utility")
                ys.append(v if v is not None else np.nan)
            ax.plot(cvers, ys, marker="o", markersize=4, linewidth=1.5,
                    linestyle=style, color=color, label=pol_label)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{GEN_DISPLAY.get(gen, gen)}  (LCB-hard, n={d.get('n_instances')}, prior={d.get('prior',0):.2f})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel(r"$c_\mathrm{ver}$  (verifier cost; $R{=}100$)", fontsize=9)
        ax.set_ylabel("mean utility", fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")

    # Empty 6th cell → legend
    legend_ax = axes[5]
    legend_ax.set_axis_off()
    handles = []
    labels = []
    for pol_key, pol_label, color, style in POLICIES:
        handles.append(plt.Line2D([0], [0], color=color, linestyle=style,
                                  marker="o", markersize=4, linewidth=1.8))
        labels.append(pol_label)
    legend_ax.legend(handles, labels, loc="center", fontsize=10,
                     frameon=True, ncol=1, title="Policy")

    fig.suptitle(r"$c_\mathrm{ver}$ sweep on LCB-hard --- regime shifts as verifier cost grows",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = args.out_dir / "fig_cver_sweep.png"
    out_pdf = args.out_dir / "fig_cver_sweep.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def fig_theta_sensitivity(args):
    """Bar chart: bayesian_greedy Δ vs always_verify under each perturbation,
    one bar group per generator, grouped bars per condition."""
    conditions = [("clean", "clean"),
                  ("minus_20", r"$-20\%$"),
                  ("minus_10", r"$-10\%$"),
                  ("plus_10",  r"$+10\%$"),
                  ("plus_20",  r"$+20\%$"),
                  ("alt_20",   r"alt $\pm 20\%$")]
    cond_keys = [c[0] for c in conditions]
    cond_labels = [c[1] for c in conditions]

    n_gens = len(GENERATORS)
    n_cond = len(conditions)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(n_gens, dtype=float)
    bar_w = 0.13

    cmap = plt.get_cmap("RdYlGn")

    for j, (ck, cl) in enumerate(conditions):
        offset = (j - (n_cond - 1) / 2) * bar_w
        deltas, los, his = [], [], []
        for gen in GENERATORS:
            path = args.data_root / "lcb_calibration_v2" / gen / "sensitivity.json"
            if not path.exists():
                deltas.append(np.nan); los.append(0); his.append(0)
                continue
            d = json.loads(path.read_text())
            d1 = d.get("D1_theta_sensitivity", {})
            entry = d1.get(ck, {})
            bg = entry.get("bayesian_greedy", {})
            delta = bg.get("diff_vs_baseline")
            lo = bg.get("ci95_lo", delta)
            hi = bg.get("ci95_hi", delta)
            if delta is None:
                deltas.append(np.nan); los.append(0); his.append(0)
            else:
                deltas.append(delta)
                los.append(delta - lo if lo is not None else 0)
                his.append(hi - delta if hi is not None else 0)
        deltas = np.array(deltas)
        # Use a sequential color per condition
        color = cmap(0.15 + 0.7 * j / max(1, n_cond - 1))
        ax.bar(x + offset, deltas, bar_w,
               yerr=[los, his], color=color, edgecolor="black",
               linewidth=0.5, capsize=2.0, label=cl)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([GEN_DISPLAY.get(g, g) for g in GENERATORS],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(r"bayesian_greedy $\Delta$ utility vs always_verify", fontsize=10)
    ax.set_title(r"D1 $\theta$-sensitivity: each $P(z|Y)$ entry perturbed ±10\%, ±20\%; controllers refit; result re-evaluated",
                 fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=9, ncol=2, framealpha=0.95,
              title="perturbation")

    fig.tight_layout()
    out_png = args.out_dir / "fig_theta_sensitivity.png"
    out_pdf = args.out_dir / "fig_theta_sensitivity.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir",   required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_cver_sweep(args)
    fig_theta_sensitivity(args)


if __name__ == "__main__":
    main()
