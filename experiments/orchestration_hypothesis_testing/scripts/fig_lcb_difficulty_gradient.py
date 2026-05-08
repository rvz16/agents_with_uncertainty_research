"""Render the LCB difficulty-gradient figure for the paper.

Shows that iterative refinement value scales with problem tractability
within LCB. Three difficulty levels (hard, medium, easy) and four
generators per level. For each cell, reads:
  - P_fix(iter, with critic feedback) from <iter-dir>/<gen>/transition_kernel.json
  - P_fix(IID baseline, no feedback) from <calibration-dir>/<gen>/transition_kernel_iid_baseline.json

Then plots:
  Panel A (top):    bars of (P_fix iter, P_fix baseline) per (difficulty x generator)
  Panel B (bottom): bars of delta = P_fix(iter) - P_fix(baseline), red=hurts/green=helps

Outputs: data/paper_figs/fig_lcb_difficulty_gradient.{png, pdf}

Usage:
  python3 scripts/fig_lcb_difficulty_gradient.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIFFICULTIES = [
    # (label, calib_dir, iter_dir, qwen32b_realbaselines_dir)
    ("LCB-hard",   "lcb_calibration_v2",     "lcb_calibration_v2_iter",
     "lcb_calibration_hard_realbaselines"),
    ("LCB-medium", "lcb_calibration_medium", "lcb_calibration_medium_iter",
     "lcb_calibration_medium_realbaselines"),
    ("LCB-easy",   "lcb_calibration_easy",   "lcb_calibration_easy_iter",
     "lcb_calibration_easy_realbaselines"),
]

GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}
GEN_COLORS = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}


def load_kernel(path: Path, key: str = "P_fix_given_broken") -> float | None:
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if key in d:
        return d[key]
    if "kernel_all" in d and key in d["kernel_all"]:
        return d["kernel_all"][key]
    return None


def load_iter_kernel_qwen32b(realbaselines_dir: Path) -> float | None:
    return load_iter_kernel_realbaselines(realbaselines_dir, "qwen25_32b")


def load_iter_kernel_realbaselines(realbaselines_dir: Path, gen: str) -> float | None:
    """Per-method realbaselines layout: <rb>/<gen>/{selfrefine,reflexion}/transition_kernel.json.
    Prefer selfrefine; fall back to reflexion."""
    for method in ("selfrefine", "reflexion"):
        p = realbaselines_dir / gen / method / "transition_kernel.json"
        v = load_kernel(p)
        if v is not None:
            return v
    return None




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for diff_label, calib_dir, iter_dir, rb_dir in DIFFICULTIES:
        for gen in GENERATORS:
            calib_path = args.data_root / calib_dir
            # Iter kernel — qwen32b lives in the *_realbaselines tree under
            # selfrefine/, closed-API gens live in the standard *_iter/<gen>/.
            # Use per-method realbaselines (preferring selfrefine, fallback reflexion)
            # for ALL generators — closed-API and qwen32b alike.
            p_fix_iter = load_iter_kernel_realbaselines(args.data_root / rb_dir, gen)

            # IID-baseline kernel — measured from the (instance, patch_id) pairs
            # in the calibration corpus via scripts/lcb_baseline_kernel.py.
            base_kernel = calib_path / gen / "transition_kernel_iid_baseline.json"
            p_fix_base = load_kernel(base_kernel)

            rows.append({
                "difficulty": diff_label,
                "generator": gen,
                "p_fix_iter": p_fix_iter,
                "p_fix_baseline": p_fix_base,
                "delta": (p_fix_iter - p_fix_base) if (p_fix_iter is not None and p_fix_base is not None) else None,
            })

    # --- Figure: 2 panels stacked, shared X-axis grouping ---
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 7.0),
                                     gridspec_kw={"height_ratios": [1.1, 1.0]})

    # X-positions: 4 gens per difficulty, with gap between difficulty groups
    n_gen = len(GENERATORS)
    n_diff = len(DIFFICULTIES)
    bar_w = 0.36
    group_w = n_gen * 1.0
    gap = 0.6
    x_centers = []
    x_labels = []
    diff_xrange = []
    for i, (diff_label, _, _, _) in enumerate(DIFFICULTIES):
        start = i * (group_w + gap)
        for j, gen in enumerate(GENERATORS):
            x_centers.append(start + j)
            x_labels.append(GEN_DISPLAY[gen])
        diff_xrange.append((start - 0.4, start + n_gen - 1 + 0.4))

    x_centers = np.array(x_centers)

    # === Panel A: P_fix iter vs baseline (paired bars) ===
    p_iter = np.array([(r["p_fix_iter"] or 0) for r in rows])
    p_base = np.array([(r["p_fix_baseline"] or 0) for r in rows])
    bars_iter = ax_a.bar(x_centers - bar_w/2, p_iter, bar_w,
                         color="#2c5d9f", edgecolor="black", linewidth=0.5,
                         label=r"$P_{fix}$ (iter, with critic feedback)")
    bars_base = ax_a.bar(x_centers + bar_w/2, p_base, bar_w,
                         color="white", edgecolor="black", linewidth=1.0, hatch="///",
                         label=r"$P_{fix}$ (IID baseline, no feedback)")
    ax_a.set_ylabel(r"$P(Y_{t+1}=1 \mid Y_t=0)$", fontsize=11)
    ax_a.set_xticks(x_centers)
    ax_a.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8.5)
    ax_a.set_ylim(0, max(p_iter.max(), p_base.max()) * 1.18 + 0.04)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
                frameon=True, fontsize=9)
    ax_a.grid(axis="y", linestyle=":", alpha=0.4)

    # Difficulty group labels at top edge of panel A
    y_top = ax_a.get_ylim()[1]
    for (lo, hi), (diff_label, _, _, _) in zip(diff_xrange, DIFFICULTIES):
        ax_a.text((lo + hi) / 2, y_top * 0.97,
                  diff_label, ha="center", va="top", fontsize=11,
                  fontweight="bold", bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor="#f0f0f0", edgecolor="gray"))

    # === Panel B: delta P_fix = iter - baseline (signed bars) ===
    deltas = np.array([(r["delta"] if r["delta"] is not None else 0) for r in rows])
    colors = ["#2ca02c" if d > 0.005 else "#d62728" if d < -0.005 else "#cccccc"
              for d in deltas]
    bars_d = ax_b.bar(x_centers, deltas, 0.85, color=colors,
                      edgecolor="black", linewidth=0.5)
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.set_ylabel(r"$\Delta P_{fix}$ = iter $-$ baseline", fontsize=11)
    ax_b.set_xticks(x_centers)
    ax_b.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8.5)
    pad = 0.02
    # Reserve extra headroom on the y-axis so the difficulty group labels can
    # sit above the bars without colliding.
    ax_b.set_ylim(deltas.min() - pad, max(deltas.max(), 0.0) + pad + 0.10)
    ax_b.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate each bar with its delta value
    for x, d in zip(x_centers, deltas):
        offset = 0.005 if d >= 0 else -0.005
        va = "bottom" if d >= 0 else "top"
        ax_b.text(x, d + offset, f"{d:+.2f}", ha="center", va=va, fontsize=8)

    # Difficulty group labels at top edge of panel B (matches panel A)
    y_top_b = ax_b.get_ylim()[1]
    for (lo, hi), (diff_label, _, _, _) in zip(diff_xrange, DIFFICULTIES):
        ax_b.text((lo + hi) / 2, y_top_b * 0.95,
                  diff_label, ha="center", va="top", fontsize=11,
                  fontweight="bold", bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor="#f0f0f0", edgecolor="gray"))

    # Difficulty group labels at bottom
    for (lo, hi), (diff_label, _, _, _) in zip(diff_xrange, DIFFICULTIES):
        ax_b.axvspan(lo - 0.05, hi + 0.05, ymin=0, ymax=1, alpha=0.0)

    fig.suptitle(
        "Iterative refinement value scales with problem tractability (LCB)\n"
        "Capable generators (gpt-5-mini, sonnet-4.5) benefit dramatically as problems become tractable",
        fontsize=12.5, y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = args.out_dir / "fig_lcb_difficulty_gradient.png"
    out_pdf = args.out_dir / "fig_lcb_difficulty_gradient.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    # Also dump the data table for reference
    out_csv = args.out_dir / "fig_lcb_difficulty_gradient.csv"
    with open(out_csv, "w") as f:
        f.write("difficulty,generator,p_fix_iter,p_fix_baseline,delta\n")
        for r in rows:
            iter_s = f"{r['p_fix_iter']:.4f}" if r['p_fix_iter'] is not None else ""
            base_s = f"{r['p_fix_baseline']:.4f}" if r['p_fix_baseline'] is not None else ""
            d_s    = f"{r['delta']:+.4f}" if r['delta'] is not None else ""
            f.write(f"{r['difficulty']},{r['generator']},{iter_s},{base_s},{d_s}\n")

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_csv}")
    print()
    print("=== Data summary ===")
    print(f"{'difficulty':12} {'generator':14} {'P_fix iter':>10}  {'P_fix base':>10}  {'Δ':>8}")
    for r in rows:
        iter_s = f"{r['p_fix_iter']:.3f}" if r['p_fix_iter'] is not None else "n/a"
        base_s = f"{r['p_fix_baseline']:.3f}" if r['p_fix_baseline'] is not None else "n/a"
        d_s    = f"{r['delta']:+.3f}" if r['delta'] is not None else "n/a"
        print(f"{r['difficulty']:12} {GEN_DISPLAY[r['generator']]:14} {iter_s:>10}  {base_s:>10}  {d_s:>8}")


if __name__ == "__main__":
    main()
