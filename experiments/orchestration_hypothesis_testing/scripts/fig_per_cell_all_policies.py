"""Per-(benchmark, model) figures showing ALL policies side by side.

For each cell we plot Δ utility vs always_verify for every policy we have:
  Bayesian methods:  bayesian_greedy, bayesian_DP (measured kernel if available)
  Threshold policies: threshold_L0, threshold_L2, threshold_L3
  Other stateless:   best_of_3, fixed_pipeline
  Iter replay (when available): selfrefine_last, reflexion_first_pass

Output: data/paper_figs/fig_all_policies_<bench>_<gen>.{png,pdf}

Usage:
  python3 scripts/fig_per_cell_all_policies.py --data-root data --out-dir data/paper_figs
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
    ("LCB-hard",      "lcb_calibration_v2",     "lcb_calibration_v2_iter"),
    ("LCB-medium",    "lcb_calibration_medium", "lcb_calibration_medium_iter"),
    ("LCB-easy",      "lcb_calibration_easy",   "lcb_calibration_easy_iter"),
    ("MBPP+",         "mbpp_calibration",       None),
    ("HumanEval+",    "humaneval_calibration",  None),
    ("SWE-Lite",      "swebench_lite",          "swebench_lite/source"),
    ("SWE-Verified",  "swebench_verified",      "swebench_verified_iter"),
]
GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
    "qwen25_32b":  "qwen2.5-32b (local)",
}

POLICIES = [
    ("bayesian_greedy",     "bayes_greedy",            "#1d4ed8"),
    ("bayesian_DP",         "bayes_DP",                "#0ea5e9"),
    ("threshold_L0",        "threshold(L0)",           "#94a3b8"),
    ("threshold_L2",        "threshold(L2)",           "#22c55e"),
    ("threshold_L3",        "threshold(L3)",           "#a855f7"),
    ("best_of_3",           "best_of_3",               "#f59e0b"),
    ("fixed_pipeline",      "fixed_pipeline",          "#ef4444"),
    ("selfrefine_last",     "Self-Refine [Madaan]",    "#fb923c"),
    ("reflexion_first_pass","Reflexion [Shinn]",       "#10b981"),
]


def safe_load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def get_pol(d: dict | None, name: str) -> dict | None:
    if not d: return None
    if "policies" in d and isinstance(d["policies"], dict):
        return d["policies"].get(name)
    return d.get(name)


def gather_cell(calib_dir: Path, iter_dir: Path | None, gen: str) -> dict | None:
    """Return {policy: (delta, lo, hi)} for every policy we have on this cell."""
    out: dict[str, tuple[float, float, float]] = {}

    # Calibration / per-patch policies
    pc = safe_load(calib_dir / gen / "policy_comparison.json")
    if pc is None:
        return None
    pols = pc.get("policies", pc)
    for pol_key in ["bayesian_greedy", "threshold_L0", "threshold_L2",
                    "threshold_L3", "best_of_3", "fixed_pipeline"]:
        p = pols.get(pol_key) if pol_key in pols else None
        if p and p.get("diff_vs_always_verify") is not None:
            out[pol_key] = (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])

    # bayesian_DP under measured kernel (if available), else default
    for vf in ["policy_comparison_kernel_measured.json",
               "policy_comparison_kernel_iterative.json"]:
        d = safe_load(calib_dir / gen / vf)
        p = get_pol(d, "bayesian_DP")
        if p and p.get("diff_vs_always_verify") is not None:
            out["bayesian_DP"] = (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])
            break
    if "bayesian_DP" not in out:
        p = pols.get("bayesian_DP")
        if p and p.get("diff_vs_always_verify") is not None:
            out["bayesian_DP"] = (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])

    # Iter replay baselines (if iter_dir exists)
    if iter_dir is not None:
        rb = safe_load(iter_dir / gen / "policy_comparison_iter_replay_baselines.json")
        if rb is not None:
            for pol_key in ["selfrefine_last", "reflexion_first_pass"]:
                p = get_pol(rb, pol_key)
                if p and p.get("diff_vs_always_verify") is not None:
                    out[pol_key] = (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])

    return out


def render_cell(cell_label: str, gen: str, data: dict, out_dir: Path) -> Path:
    """Render one (bench, gen) figure with all available policies."""
    keys = [k for k, _, _ in POLICIES if k in data]
    if not keys:
        return None
    deltas = [data[k][0] for k in keys]
    lo     = [data[k][0] - data[k][1] for k in keys]
    hi     = [data[k][2] - data[k][0] for k in keys]
    yerr   = np.array([lo, hi])
    labels = [next(lbl for k, lbl, _ in POLICIES if k == kk) for kk in keys]
    colors = [next(c   for k, _, c   in POLICIES if k == kk) for kk in keys]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(keys), dtype=float)
    bars = ax.bar(x, deltas, yerr=yerr, color=colors, capsize=3,
                  edgecolor="black", linewidth=0.55)
    # Annotate winning bar with star
    winner_idx = int(np.argmax(deltas))
    for i, (b, v) in enumerate(zip(bars, deltas)):
        offset = max(hi[i], 0.3) + 0.4
        text = f"{'+' if v >= 0 else ''}{v:.1f}"
        if i == winner_idx:
            text = f"$\\bigstar$ " + text
        ax.text(b.get_x() + b.get_width() / 2, v + offset, text,
                ha="center", fontsize=8.0,
                fontweight="bold" if i == winner_idx else "normal")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ utility vs always_verify", fontsize=9)
    ax.set_title(f"{cell_label} / {GEN_DISPLAY.get(gen, gen)}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    bench_slug = cell_label.lower().replace("-", "_").replace("+", "plus")
    out_png = out_dir / f"fig_all_policies_{bench_slug}_{gen}.png"
    out_pdf = out_dir / f"fig_all_policies_{bench_slug}_{gen}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir",   required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for cell_label, calib_subdir, iter_subdir in CELLS:
        calib_dir = args.data_root / calib_subdir
        iter_dir  = args.data_root / iter_subdir if iter_subdir else None
        for gen in GENERATORS:
            data = gather_cell(calib_dir, iter_dir, gen)
            if data is None:
                continue
            out = render_cell(cell_label, gen, data, args.out_dir)
            if out is not None:
                n_written += 1
                print(f"  wrote {out.name} ({len(data)} policies)")
    print(f"\n{n_written} figures written.")


if __name__ == "__main__":
    main()
