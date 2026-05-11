"""Single-panel winner matrix using the SAME code path as fig_winner_grid_R_cver.py
(uses run_policies() from lcb_sensitivity, computes from raw critic_results +
likelihood_tables). Replaces fig_winner_matrix.py to guarantee consistency
with the 4x4 grid's (R=100, c_ver=1) panel.

Output: data/paper_figs/fig_winner_matrix.{png,pdf}

Usage:
  python3 scripts/fig_winner_matrix_consistent.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_baseline_vs_controller import CostModel  # noqa: E402
from lcb_sensitivity import run_policies  # noqa: E402
from lcb_compare import load_lcb_trajectories  # noqa: E402


GEN_ORDER = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}
BENCH_ORDER = ["lcb_hard", "lcb_medium", "lcb_easy",
               "mbpp", "humaneval", "swebench_lite", "swebench_verified"]
BENCH_DISPLAY = {
    "lcb_hard":          "LCB-hard",
    "lcb_medium":        "LCB-medium",
    "lcb_easy":          "LCB-easy",
    "mbpp":              "MBPP+",
    "humaneval":         "HumanEval+",
    "swebench_lite":     "SWE-Lite",
    "swebench_verified": "SWE-Verified",
}
BENCH_DIR = {
    "lcb_hard":          "lcb_calibration_v2",
    "lcb_medium":        "lcb_calibration_medium",
    "lcb_easy":          "lcb_calibration_easy",
    "mbpp":              "mbpp_calibration",
    "humaneval":         "humaneval_calibration",
    "swebench_lite":     "swebench_lite",
    "swebench_verified": "swebench_verified",
}
QWEN32B_SWE_DIR = {
    "swebench_lite":     "swebench_lite_qwen32b",
    "swebench_verified": "swebench_verified_qwen32b",
}

POLICY_COLOR = {
    "bayesian_greedy":  "#86efac",
    "bayesian_DP":      "#bbf7d0",
    "threshold_L2":     "#fed7aa",
    "threshold_L0":     "#bae6fd",
    "threshold_L3":     "#bae6fd",
    "best_of_3":        "#fde68a",
    "fixed_pipeline":   "#fecaca",
    "always_verify":    "#e5e7eb",
}
POLICY_SHORT = {
    "bayesian_greedy": "bayes_g",
    "bayesian_DP":     "bayes_DP",
    "threshold_L0":    "thr(L0)",
    "threshold_L2":    "thr(L2)",
    "threshold_L3":    "thr(L3)",
    "best_of_3":       "best3",
    "fixed_pipeline":  "fixed",
    "always_verify":   "always_v",
}


def cell_data_path(data_root: Path, bench: str, gen: str) -> Path:
    if gen == "qwen25_32b" and bench in QWEN32B_SWE_DIR:
        return data_root / QWEN32B_SWE_DIR[bench] / gen
    return data_root / BENCH_DIR[bench] / gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--R", type=float, default=100.0)
    p.add_argument("--c-ver", type=float, default=30.0)
    p.add_argument("--c-L0", type=float, default=1.0)
    p.add_argument("--c-L2", type=float, default=2.0)
    p.add_argument("--c-L3", type=float, default=5.0)
    p.add_argument("--c-gen", type=float, default=5.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cost = CostModel(c_gen=args.c_gen, c_L0=args.c_L0, c_L2=args.c_L2,
                     c_L3=args.c_L3, c_ver=args.c_ver, reward=args.R)

    # Compute winners
    winners: dict[tuple[str, str], tuple[str, float, float]] = {}
    for b in BENCH_ORDER:
        for g in GEN_ORDER:
            cdp = cell_data_path(args.data_root, b, g)
            cr, lt = cdp / "critic_results.jsonl", cdp / "likelihood_tables.json"
            if not (cr.exists() and lt.exists()):
                continue
            likes = json.loads(lt.read_text())
            prior = likes.get("prior_Y1", 0.5)
            traj = load_lcb_trajectories(cr)
            try:
                res = run_policies(traj, likes, prior, cost, n_boot=1)
            except Exception as e:
                print(f"  skip {b}/{g}: {e}")
                continue
            winner = max(res.keys(), key=lambda k: res[k]["mean_utility"])
            # Δ vs always_verify
            av_u = res.get("always_verify", {}).get("mean_utility", 0.0)
            w_u = res[winner]["mean_utility"]
            winners[(b, g)] = (winner, w_u - av_u, w_u)

    # Plot
    n_b = len(BENCH_ORDER)
    n_g = len(GEN_ORDER)
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    for bi, b in enumerate(BENCH_ORDER):
        for gi, g in enumerate(GEN_ORDER):
            y = n_b - 1 - bi
            x = gi
            if (b, g) not in winners:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="white",
                                       edgecolor="black", linewidth=0.6, hatch="///"))
                continue
            w, delta, _ = winners[(b, g)]
            color = POLICY_COLOR.get(w, "#fff")
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color,
                                   edgecolor="black", linewidth=0.6))
            ax.text(x + 0.5, y + 0.62, POLICY_SHORT.get(w, w),
                    ha="center", va="center", fontsize=10, fontweight="bold")
            ax.text(x + 0.5, y + 0.32, f"$\\Delta$= {delta:+.1f}",
                    ha="center", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, n_g)
    ax.set_ylim(0, n_b)
    ax.set_xticks([i + 0.5 for i in range(n_g)])
    ax.set_xticklabels([GEN_DISPLAY[g] for g in GEN_ORDER],
                       rotation=30, ha="right", fontsize=10)
    ax.set_yticks([n_b - 1 - i + 0.5 for i in range(n_b)])
    ax.set_yticklabels([BENCH_DISPLAY[b] for b in BENCH_ORDER], fontsize=10)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # Legend
    legend_keys = ["bayesian_greedy", "bayesian_DP", "threshold_L2",
                   "threshold_L0", "threshold_L3", "always_verify"]
    handles = [Rectangle((0, 0), 1, 1, facecolor=POLICY_COLOR[k],
                         edgecolor="black", linewidth=0.6) for k in legend_keys]
    labels = [POLICY_SHORT[k] for k in legend_keys]
    ax.legend(handles, labels, loc="center left",
              bbox_to_anchor=(1.02, 0.5), title="Per-cell winner",
              fontsize=9, frameon=True)

    fig.tight_layout()
    out_png = args.out_dir / "fig_winner_matrix.png"
    out_pdf = args.out_dir / "fig_winner_matrix.pdf"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")
    print(f"  cost: R={args.R}, c_ver={args.c_ver}, c_L0={args.c_L0}, c_L2={args.c_L2}, c_L3={args.c_L3}, c_gen={args.c_gen}")
    print(f"  cells populated: {len(winners)}/{n_b * n_g}")


if __name__ == "__main__":
    main()
