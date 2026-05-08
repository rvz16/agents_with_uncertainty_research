"""Per-cell winner matrix: benchmarks (rows) × generators (cols), colored by
the policy that wins each cell. Visually shows the regime structure.

Output: data/paper_figs/fig_winner_matrix.{png,pdf}

Usage:
  python3 scripts/fig_winner_matrix.py --paper-table data/PAPER_TABLE.json \
    --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


GEN_ORDER = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}
BENCH_ORDER = ["lcb_hard", "lcb_medium", "lcb_easy",
               "mbpp", "humaneval",
               "swe_lite", "swe_verified"]
BENCH_DISPLAY = {
    "lcb_hard":          "LCB-hard",
    "lcb_medium":        "LCB-medium",
    "lcb_easy":          "LCB-easy",
    "mbpp":              "MBPP+",
    "humaneval":         "HumanEval+",
    "swe_lite":     "SWE-Lite",
    "swe_verified": "SWE-Verified",
}

# Color per policy (winner) — cell background
POLICY_COLOR = {
    "bayesian_greedy":  "#86efac",  # light green
    "bayesian_DP":      "#bbf7d0",  # very light green
    "threshold_L2":     "#fed7aa",  # light orange
    "threshold_L0":     "#bae6fd",  # light blue
    "threshold_L3":     "#bae6fd",  # light blue
    "best_of_3":        "#fde68a",
    "fixed_pipeline":   "#fecaca",
    "always_verify":    "#e5e7eb",  # light grey
    "selfrefine_last":  "#fef3c7",
    "reflexion_first_pass":"#fef3c7",
}
POLICY_SHORT = {
    "bayesian_greedy": "bayes_g",
    "bayesian_DP":     "bayes_DP",
    "threshold_L0":    "thr(L0)",
    "threshold_L2":    "thr(L2)",
    "threshold_L3":    "thr(L3)",
    "best_of_3":       "best3",
    "fixed_pipeline":  "fixed_pipe",
    "always_verify":   "always_v",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-table", required=True, type=Path)
    parser.add_argument("--out-dir",     required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = json.loads(args.paper_table.read_text())

    n_b = len(BENCH_ORDER)
    n_g = len(GEN_ORDER)
    fig, ax = plt.subplots(figsize=(13.0, 5.5))

    # Draw each cell as a coloured rectangle with the winner policy name
    for bi, bench in enumerate(BENCH_ORDER):
        for gi, gen in enumerate(GEN_ORDER):
            rev = table.get(bench, {}).get(gen, {}).get("haiku45_default")
            if not rev:
                ax.add_patch(Rectangle((gi, n_b - 1 - bi), 1, 1,
                                       facecolor="white", edgecolor="black",
                                       linewidth=0.5, hatch="///"))
                continue
            policies = rev["policies"]
            # rank by diff_vs_always_verify
            ranked = sorted(
                [(k, v.get("diff_vs_always_verify"))
                 for k, v in policies.items()
                 if v.get("diff_vs_always_verify") is not None],
                key=lambda kv: kv[1], reverse=True
            )
            if not ranked:
                continue
            winner_name, winner_util = ranked[0]
            color = POLICY_COLOR.get(winner_name, "#ffffff")
            ax.add_patch(Rectangle((gi, n_b - 1 - bi), 1, 1,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.6))
            short = POLICY_SHORT.get(winner_name, winner_name)
            ax.text(gi + 0.5, n_b - 1 - bi + 0.62, short,
                    ha="center", va="center", fontsize=8.5,
                    fontfamily="monospace", fontweight="bold")
            # Winner utility below the policy name
            util_str = f"$\\Delta\\!=\\!{'+' if winner_util >= 0 else ''}{winner_util:.1f}$"
            ax.text(gi + 0.5, n_b - 1 - bi + 0.30, util_str,
                    ha="center", va="center", fontsize=7.5, color="#374151")

    # Axes labels
    ax.set_xlim(0, n_g)
    ax.set_ylim(0, n_b)
    ax.set_xticks([i + 0.5 for i in range(n_g)])
    ax.set_xticklabels([GEN_DISPLAY[g] for g in GEN_ORDER],
                       rotation=20, ha="right", fontsize=10)
    ax.set_yticks([n_b - 1 - i + 0.5 for i in range(n_b)])
    ax.set_yticklabels([BENCH_DISPLAY[b] for b in BENCH_ORDER], fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend (policy → color)
    legend_keys = ["bayesian_greedy", "bayesian_DP", "threshold_L2",
                   "threshold_L0", "threshold_L3", "always_verify"]
    handles = [Rectangle((0, 0), 1, 1, facecolor=POLICY_COLOR[k],
                         edgecolor="black", linewidth=0.5)
               for k in legend_keys]
    labels = [POLICY_SHORT.get(k, k) for k in legend_keys]
    ax.legend(handles, labels, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True,
              title="Per-cell winner")

    fig.tight_layout()
    out_png = args.out_dir / "fig_winner_matrix.png"
    out_pdf = args.out_dir / "fig_winner_matrix.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
