"""Per-benchmark figures for MBPP+ and HumanEval+ — calibration-only.

These two benchmarks have no iter trajectories (regime C, saturated), so we
cannot show Self-Refine / Reflexion replays. Instead we show Bayesian vs the
stateless per-patch baselines (always_verify, threshold_L*, best_of_3,
fixed_pipeline) on the calibration corpus.

Usage:
  python3 scripts/fig_calib_only_mbpp_humaneval.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CELLS = [
    ("MBPP+",      "mbpp_calibration",      "mbpp"),
    ("HumanEval+", "humaneval_calibration", "humaneval"),
]
GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}

# Policies to show (left to right within each generator group)
POLICIES = [
    ("bayesian_best", "Bayesian (best of greedy/DP)", "#1d4ed8"),
    ("threshold_L0",  "threshold(L0)",                "#94a3b8"),
    ("threshold_L2",  "threshold(L2)",                "#0ea5e9"),
    ("threshold_L3",  "threshold(L3)",                "#a855f7"),
    ("best_of_3",     "best-of-3",                    "#f59e0b"),
    ("fixed_pipeline","fixed pipeline",               "#ef4444"),
]


def safe_load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def get_policies_for_gen(calib_dir: Path, gen: str) -> dict | None:
    pc = safe_load(calib_dir / gen / "policy_comparison.json")
    if pc is None:
        return None
    pols = pc.get("policies", pc)
    out = dict(pols)
    # Synthesize "bayesian_best" = max of greedy / DP
    bg = pols.get("bayesian_greedy")
    bd = pols.get("bayesian_DP")
    cands = [c for c in (bg, bd) if c and c.get("diff_vs_always_verify") is not None]
    if cands:
        best = max(cands, key=lambda c: c["diff_vs_always_verify"])
        out["bayesian_best"] = best
    return out


def render_one(cell_label: str, calib_dir: Path, slug: str, out_dir: Path) -> None:
    rows: list[tuple[str, dict]] = []
    for gen in GENERATORS:
        pols = get_policies_for_gen(calib_dir, gen)
        if pols is None:
            continue
        rows.append((gen, pols))
    if not rows:
        print(f"  no data for {cell_label}; skipping")
        return

    n_gens = len(rows)
    n_pol = len(POLICIES)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bar_w = 0.13
    x = np.arange(n_gens, dtype=float)

    for j, (pol_key, pol_label, color) in enumerate(POLICIES):
        offset = (j - (n_pol - 1) / 2) * bar_w
        diffs, los, his = [], [], []
        for gen, pols in rows:
            p = pols.get(pol_key, {})
            diff = p.get("diff_vs_always_verify")
            lo = p.get("ci95_lo", diff)
            hi = p.get("ci95_hi", diff)
            if diff is None:
                diff = lo = hi = 0.0
            diffs.append(diff)
            los.append(diff - lo)
            his.append(hi - diff)
        ax.bar(x + offset, diffs, bar_w,
               yerr=[los, his], color=color, edgecolor="black",
               linewidth=0.4, capsize=2.0, label=pol_label)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([GEN_DISPLAY.get(g, g) for g, _ in rows],
                       rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(r"$\Delta$ utility vs always_verify", fontsize=10)
    ax.set_title(f"{cell_label} (regime C: saturated, no iter trajectories)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=7.5, framealpha=0.92, ncol=2)

    fig.tight_layout()
    out_png = out_dir / f"fig_framework_vs_baselines_{slug}.png"
    out_pdf = out_dir / f"fig_framework_vs_baselines_{slug}.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cell_label, calib_subdir, slug in CELLS:
        render_one(cell_label, args.data_root / calib_subdir, slug, args.out_dir)


if __name__ == "__main__":
    main()
