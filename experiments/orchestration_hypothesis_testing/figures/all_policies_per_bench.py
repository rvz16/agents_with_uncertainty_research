"""Per-benchmark figure: 5 generator panels, each showing ALL policies as bars.

For each (benchmark, generator) cell, plot all available policies side-by-side
on the same axis: bayesian_greedy, bayesian_DP, threshold_L0, threshold_L2,
threshold_L3, best_of_3, fixed_pipeline, plus Self-Refine and Reflexion where
iter trajectories exist.

Output: data/paper_figs/fig_all_policies_<benchslug>.{png,pdf}

Usage:
  python3 scripts/fig_all_policies_per_bench.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# (display label, calibration subdir, iter subdir or None)
BENCHES = [
    ("LCB-hard",     "lcb_calibration_v2",     "lcb_calibration_v2_iter"),
    ("LCB-medium",   "lcb_calibration_medium", "lcb_calibration_medium_iter"),
    ("LCB-easy",     "lcb_calibration_easy",   "lcb_calibration_easy_iter"),
    ("SWE-Lite",     "swebench_lite",          "swebench_lite/source"),
    ("SWE-Verified", "swebench_verified",      "swebench_verified_iter"),
    ("MBPP+",        "mbpp_calibration",       None),
    ("HumanEval+",   "humaneval_calibration",  None),
]

GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}

# (key, display, color) — order = bar order
POLICIES = [
    ("bayesian_greedy",     "bayesian_greedy",      "#1d4ed8"),
    ("bayesian_DP",         "bayesian_DP (kernel)", "#0ea5e9"),
    ("threshold_L0",        "threshold(L0)",        "#94a3b8"),
    ("threshold_L2",        "threshold(L2)",        "#a855f7"),
    ("threshold_L3",        "threshold(L3)",        "#ec4899"),
    ("best_of_3",           "best-of-3",            "#fbbf24"),
    ("fixed_pipeline",      "fixed pipeline",       "#ef4444"),
    ("selfrefine_last",     "Self-Refine",          "#f97316"),
    ("reflexion_first_pass","Reflexion",            "#10b981"),
]


def safe_load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def get_pol(d: dict | None, name: str) -> dict | None:
    if not d:
        return None
    pols = d.get("policies", d)
    return pols.get(name)


def get_dp_with_kernel(calib_gen_dir: Path) -> dict | None:
    """Prefer measured-kernel DP, fall back to default policy_comparison.json."""
    for variant in ["policy_comparison_kernel_measured.json",
                    "policy_comparison_kernel_iterative.json"]:
        d = safe_load(calib_gen_dir / variant)
        p = get_pol(d, "bayesian_DP")
        if p and p.get("diff_vs_always_verify") is not None:
            return p
    return get_pol(safe_load(calib_gen_dir / "policy_comparison.json"), "bayesian_DP")


def get_policies_for_cell(calib_gen_dir: Path, iter_gen_dir: Path | None) -> dict:
    """Return {policy_key: (delta, lo, hi)} for all available policies."""
    out = {}
    pc = safe_load(calib_gen_dir / "policy_comparison.json")
    for key, _, _ in POLICIES:
        if key == "bayesian_DP":
            p = get_dp_with_kernel(calib_gen_dir)
        elif key in ("selfrefine_last", "reflexion_first_pass"):
            if iter_gen_dir is None:
                continue
            d = safe_load(iter_gen_dir / "policy_comparison_iter_replay_baselines.json")
            p = get_pol(d, key)
        else:
            p = get_pol(pc, key)
        if not p or p.get("diff_vs_always_verify") is None:
            continue
        out[key] = (p["diff_vs_always_verify"], p.get("ci95_lo"), p.get("ci95_hi"))
    return out


def render_bench(bench_label: str, calib_subdir: str, iter_subdir: str | None,
                 data_root: Path, out_dir: Path) -> None:
    cells = []  # (gen, {key: (d, lo, hi)})
    for gen in GENERATORS:
        calib_dir = data_root / calib_subdir / gen
        if not calib_dir.exists():
            continue
        iter_dir = (data_root / iter_subdir / gen) if iter_subdir else None
        if iter_dir and not iter_dir.exists():
            iter_dir = None
        pols = get_policies_for_cell(calib_dir, iter_dir)
        if pols:
            cells.append((gen, pols))
    if not cells:
        print(f"  no data for {bench_label}; skipping")
        return

    n_gens = len(cells)
    fig, axes = plt.subplots(1, n_gens, figsize=(3.2 * n_gens, 4.0), sharey=True)
    if n_gens == 1:
        axes = [axes]

    # Determine y-range across all panels
    all_vals = []
    for _, pols in cells:
        for key, (d, lo, hi) in pols.items():
            if d is not None:
                all_vals.append(d)
            if lo is not None:
                all_vals.append(lo)
            if hi is not None:
                all_vals.append(hi)
    if all_vals:
        y_min = min(all_vals) - 2
        y_max = max(all_vals) + 4
    else:
        y_min, y_max = -5, 5

    for ax, (gen, pols) in zip(axes, cells):
        keys_present = [k for k, _, _ in POLICIES if k in pols]
        x = np.arange(len(keys_present), dtype=float)
        deltas = []
        lows = []
        highs = []
        colors = []
        labels = []
        for k, label, color in POLICIES:
            if k not in pols:
                continue
            d, lo, hi = pols[k]
            deltas.append(d)
            lows.append(d - (lo if lo is not None else d))
            highs.append((hi if hi is not None else d) - d)
            colors.append(color)
            labels.append(label)

        ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.4,
               yerr=[lows, highs], capsize=2.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7.5)
        ax.set_title(GEN_DISPLAY.get(gen, gen), fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(y_min, y_max)
        # Highlight winning bar with a thicker edge
        if deltas:
            best_idx = int(np.argmax(deltas))
            ax.get_children()[best_idx].set_edgecolor("black")
            ax.get_children()[best_idx].set_linewidth(2.0)

    axes[0].set_ylabel(r"$\Delta$ utility vs always_verify", fontsize=10)
    fig.suptitle(f"{bench_label}: all policies $\\Delta$ vs \\texttt{{always\\_verify}} per generator "
                 "(thick-edged bar = winner per cell)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    slug = bench_label.lower().replace("-", "_").replace("+", "plus").replace(" ", "")
    png = out_dir / f"fig_all_policies_{slug}.png"
    pdf = out_dir / f"fig_all_policies_{slug}.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png.name}  ({n_gens} gens, {sum(len(p) for _,p in cells)} bars)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for bench, calib_sub, iter_sub in BENCHES:
        render_bench(bench, calib_sub, iter_sub, args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
