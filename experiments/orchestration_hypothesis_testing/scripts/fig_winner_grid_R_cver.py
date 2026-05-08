"""4x4 grid of winner-matrix heatmaps: vary R (rows) and c_ver (columns).

Each panel is a 7x5 cell heatmap showing the winning policy per
(benchmark, generator) at that specific (R, c_ver) combo. Visualizes how the
regime structure shifts as the cost vector changes.

Output: data/paper_figs/fig_winner_grid_R_cver.{png,pdf}

Usage:
  python3 scripts/fig_winner_grid_R_cver.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_baseline_vs_controller import CostModel  # noqa: E402
from lcb_sensitivity import run_policies  # noqa: E402
from lcb_compare import load_lcb_trajectories  # noqa: E402


# 4 R values × 4 c_ver values = 16 panels
R_VALUES   = [25, 50, 100, 200]
CVER_VALUES = [1, 10, 30, 60]

GEN_ORDER = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
BENCH_ORDER = ["lcb_hard", "lcb_medium", "lcb_easy",
               "mbpp", "humaneval", "swebench_lite", "swebench_verified"]
BENCH_DIR = {
    "lcb_hard":          "lcb_calibration_v2",
    "lcb_medium":        "lcb_calibration_medium",
    "lcb_easy":          "lcb_calibration_easy",
    "mbpp":              "mbpp_calibration",
    "humaneval":         "humaneval_calibration",
    "swebench_lite":     "swebench_lite",
    "swebench_verified": "swebench_verified",
}
# Qwen32B SWE cells live in a separate dir
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


def cell_data_path(data_root: Path, bench: str, gen: str) -> Path:
    if gen == "qwen25_32b" and bench in QWEN32B_SWE_DIR:
        return data_root / QWEN32B_SWE_DIR[bench] / gen
    return data_root / BENCH_DIR[bench] / gen


def load_cell(data_root: Path, bench: str, gen: str):
    p = cell_data_path(data_root, bench, gen)
    cr = p / "critic_results.jsonl"
    lt = p / "likelihood_tables.json"
    if not (cr.exists() and lt.exists()):
        return None
    likes = json.loads(lt.read_text())
    prior = likes.get("prior_Y1", 0.5)
    traj = load_lcb_trajectories(cr)
    return traj, likes, prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir",   required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: load all 35 cells once ---
    print("Loading 35 cells …")
    cells: dict[tuple[str, str], tuple] = {}
    for b in BENCH_ORDER:
        for g in GEN_ORDER:
            data = load_cell(args.data_root, b, g)
            if data is not None:
                cells[(b, g)] = data
    print(f"  loaded {len(cells)} cells")

    # --- Step 2: for each (R, c_ver), find winner per cell ---
    print(f"Simulating {len(cells)} cells × {len(R_VALUES)*len(CVER_VALUES)} cost combos …")
    winners: dict[tuple[int, int], dict[tuple[str, str], str]] = {}
    for R in R_VALUES:
        for cv in CVER_VALUES:
            # Default per-critic costs (c_L0=1, c_L2=2, c_L3=5). Slide 51
            # uses the same defaults at (R=100, c_ver=30), so the matching
            # panel of this grid reproduces it cell-by-cell.
            cost = CostModel(c_gen=5, c_L0=1, c_L2=2, c_L3=5, c_ver=cv, reward=R)
            w_map: dict[tuple[str, str], str] = {}
            for (b, g), (traj, likes, prior) in cells.items():
                try:
                    res = run_policies(traj, likes, prior, cost, n_boot=1)
                except Exception as e:
                    print(f"    skip {b}/{g} R={R} cv={cv}: {e}")
                    continue
                winner = max(res.keys(), key=lambda k: res[k]["mean_utility"])
                w_map[(b, g)] = winner
            winners[(R, cv)] = w_map
            print(f"  R={R:>3} cv={cv:>3}: done ({len(w_map)}/{len(cells)} cells)")

    # --- Step 3: render 4x4 grid ---
    n_b = len(BENCH_ORDER)
    n_g = len(GEN_ORDER)
    nR = len(R_VALUES)
    nC = len(CVER_VALUES)
    fig, axes = plt.subplots(nR, nC, figsize=(14, 11.5))

    # Render with R growing bottom-to-top (top row = highest R).
    R_render = list(reversed(R_VALUES))
    for ri, R in enumerate(R_render):
        for ci, cv in enumerate(CVER_VALUES):
            ax = axes[ri, ci]
            w_map = winners.get((R, cv), {})
            for bi, b in enumerate(BENCH_ORDER):
                for gi, g in enumerate(GEN_ORDER):
                    win = w_map.get((b, g))
                    if win is None:
                        ax.add_patch(Rectangle((gi, n_b - 1 - bi), 1, 1,
                                               facecolor="white", edgecolor="black",
                                               linewidth=0.4, hatch="///"))
                    else:
                        ax.add_patch(Rectangle((gi, n_b - 1 - bi), 1, 1,
                                               facecolor=POLICY_COLOR.get(win, "#fff"),
                                               edgecolor="black", linewidth=0.4))
            ax.set_xlim(0, n_g)
            ax.set_ylim(0, n_b)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            # Inner title per panel: cost ratio
            ax.set_title(f"$R={R},\\ c_\\mathrm{{ver}}={cv}$  (ratio $c/R={cv/R:.2g}$)",
                         fontsize=9)

    # Outer axis labels via fig.text (R on the left, c_ver on the bottom)
    fig.text(0.5, 0.005, r"$c_\mathrm{ver}$ (verifier cost) → grows left to right",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.text(0.005, 0.5, r"$R$ (reward) → grows bottom to top",
             ha="left", va="center", rotation=90, fontsize=11, fontweight="bold")

    # Legend
    legend_keys = ["bayesian_greedy", "bayesian_DP", "threshold_L2",
                   "threshold_L0", "threshold_L3", "always_verify"]
    handles = [Rectangle((0, 0), 1, 1, facecolor=POLICY_COLOR[k],
                         edgecolor="black", linewidth=0.5)
               for k in legend_keys]
    labels = {"bayesian_greedy":"bayes_g", "bayesian_DP":"bayes_DP",
              "threshold_L2":"thr(L2)","threshold_L0":"thr(L0)",
              "threshold_L3":"thr(L3)","always_verify":"always_v"}
    fig.legend(handles, [labels[k] for k in legend_keys],
               loc="lower center", ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.04), title="Per-cell winner")

    fig.suptitle("Winner per (benchmark, generator) under varying $R$ and $c_\\mathrm{ver}$ "
                 " --- 7 benches × 5 gens per panel",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.97])

    out_png = args.out_dir / "fig_winner_grid_R_cver.png"
    out_pdf = args.out_dir / "fig_winner_grid_R_cver.pdf"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
