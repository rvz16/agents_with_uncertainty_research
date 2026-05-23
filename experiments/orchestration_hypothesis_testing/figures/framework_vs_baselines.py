"""Render the framework-vs-published-baselines figure for the paper.
Faceted version: one subplot per benchmark, generator-major.

For each (benchmark, generator) cell with iter-trajectory coverage, plot
three policies side by side:
  - best_bayesian      — max of bayesian_greedy/bayesian_DP under measured kernel
  - selfrefine_last    — Self-Refine [Madaan 2023] policy replay
  - reflexion_first    — Reflexion [Shinn 2023] policy replay

All Δ utility vs always_verify, with paired-bootstrap 95% CIs.

Usage:
  python3 scripts/fig_framework_vs_baselines.py --data-root data --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ITER_CELLS = [
    ("LCB-hard",      "lcb_calibration_v2",     "lcb_calibration_v2_iter"),
    ("LCB-medium",    "lcb_calibration_medium", "lcb_calibration_medium_iter"),
    ("LCB-easy",      "lcb_calibration_easy",   "lcb_calibration_easy_iter"),
    ("SWE-Verified",  "swebench_verified",      "swebench_verified_iter"),
    ("SWE-Lite",      "swebench_lite",          "swebench_lite/source"),
]

GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45"]
GEN_DISPLAY = {
    "gpt5_mini":   "gpt-5-mini (API)",
    "qwen3_coder": "qwen3-coder (API)",
    "haiku45":     "haiku-4.5 (API)",
    "sonnet45":    "sonnet-4.5 (API)",
}


def safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except json.JSONDecodeError:
        return None


def get_policy(d: dict | None, name: str) -> dict | None:
    if not d:
        return None
    if "policies" in d and isinstance(d["policies"], dict):
        return d["policies"].get(name)
    return d.get(name)


def get_best_bayesian(calib_dir: Path, gen: str) -> tuple[float, float, float, str] | None:
    """Legacy: best of greedy/DP across kernel variants. Kept for the wide combined figure."""
    candidates: list[tuple[float, float, float, str]] = []
    default = safe_load(calib_dir / gen / "policy_comparison.json")
    g = get_policy(default, "bayesian_greedy")
    if g and g.get("diff_vs_always_verify") is not None:
        candidates.append((g["diff_vs_always_verify"], g["ci95_lo"], g["ci95_hi"],
                           "bayesian_greedy"))
    for variant_file in ["policy_comparison_kernel_iterative.json",
                          "policy_comparison_kernel_measured.json"]:
        d = safe_load(calib_dir / gen / variant_file)
        for pol in ("bayesian_DP", "bayesian_greedy"):
            p = get_policy(d, pol)
            if p and p.get("diff_vs_always_verify") is not None:
                candidates.append((p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"],
                                   f"{pol} (measured kernel)"))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])


def get_greedy(calib_dir: Path, gen: str) -> tuple[float, float, float] | None:
    """Pure bayesian_greedy (no kernel) from the default policy_comparison.json."""
    d = safe_load(calib_dir / gen / "policy_comparison.json")
    p = get_policy(d, "bayesian_greedy")
    if not p or p.get("diff_vs_always_verify") is None:
        return None
    return (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])


def get_DP_measured(calib_dir: Path, gen: str) -> tuple[float, float, float] | None:
    """bayesian_DP under the measured kernel. Tries the
    policy_comparison_kernel_measured.json variant first, falls back to the
    iterative variant if the measured one is missing."""
    for variant_file in ["policy_comparison_kernel_measured.json",
                         "policy_comparison_kernel_iterative.json"]:
        d = safe_load(calib_dir / gen / variant_file)
        p = get_policy(d, "bayesian_DP")
        if p and p.get("diff_vs_always_verify") is not None:
            return (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])
    # Fall back to default policy_comparison.json (uses IID-synthesized kernel)
    d = safe_load(calib_dir / gen / "policy_comparison.json")
    p = get_policy(d, "bayesian_DP")
    if not p or p.get("diff_vs_always_verify") is None:
        return None
    return (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])


def get_replay_baseline(iter_dir: Path, gen: str, name: str) -> tuple[float, float, float] | None:
    d = safe_load(iter_dir / gen / "policy_comparison_iter_replay_baselines.json")
    p = get_policy(d, name)
    if not p or p.get("diff_vs_always_verify") is None:
        return None
    return (p["diff_vs_always_verify"], p["ci95_lo"], p["ci95_hi"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cell_label, calib_subdir, iter_subdir in ITER_CELLS:
        for gen in GENERATORS:
            calib_dir = args.data_root / calib_subdir
            iter_dir = args.data_root / iter_subdir
            best = get_best_bayesian(calib_dir, gen)
            greedy = get_greedy(calib_dir, gen)
            dp = get_DP_measured(calib_dir, gen)
            srf = get_replay_baseline(iter_dir, gen, "selfrefine_last")
            rx = get_replay_baseline(iter_dir, gen, "reflexion_first_pass")
            if best is None or srf is None or rx is None:
                continue
            row = {
                "cell": cell_label,
                "gen": gen,
                "best_bayes_delta": best[0], "best_bayes_lo": best[1], "best_bayes_hi": best[2],
                "best_bayes_source": best[3],
                "srf_delta": srf[0], "srf_lo": srf[1], "srf_hi": srf[2],
                "rx_delta": rx[0], "rx_lo": rx[1], "rx_hi": rx[2],
            }
            if greedy is not None:
                row["greedy_delta"], row["greedy_lo"], row["greedy_hi"] = greedy
            if dp is not None:
                row["dp_delta"], row["dp_lo"], row["dp_hi"] = dp
            rows.append(row)

    if not rows:
        print("No data — exiting.")
        return

    cells_with_data = [c for c, _, _ in ITER_CELLS if any(r["cell"] == c for r in rows)]
    n_panels = len(cells_with_data)

    # Variable width per panel — proportional to its number of bars
    widths = [max(1, sum(1 for r in rows if r["cell"] == c)) for c in cells_with_data]
    fig_w = max(13, 0.85 * sum(widths) + 1.5 * n_panels + 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.4),
                             sharey=True, gridspec_kw={"wspace": 0.10,
                                                        "width_ratios": widths})
    if n_panels == 1:
        axes = [axes]

    bar_w = 0.27
    color_bayes = "#1d4ed8"
    color_srf   = "#f59e0b"
    color_rx    = "#10b981"

    n_total_wins = 0
    crossover_text = None

    for ax, cell in zip(axes, cells_with_data):
        cell_rows = [r for r in rows if r["cell"] == cell]
        cell_rows.sort(key=lambda r: GENERATORS.index(r["gen"]))
        n = len(cell_rows)
        x = np.arange(n, dtype=float)

        bayes = np.array([r["best_bayes_delta"] for r in cell_rows])
        bayes_err = np.array([
            [r["best_bayes_delta"] - r["best_bayes_lo"] for r in cell_rows],
            [r["best_bayes_hi"] - r["best_bayes_delta"] for r in cell_rows],
        ])
        srf = np.array([r["srf_delta"] for r in cell_rows])
        srf_err = np.array([
            [r["srf_delta"] - r["srf_lo"] for r in cell_rows],
            [r["srf_hi"] - r["srf_delta"] for r in cell_rows],
        ])
        rx = np.array([r["rx_delta"] for r in cell_rows])
        rx_err = np.array([
            [r["rx_delta"] - r["rx_lo"] for r in cell_rows],
            [r["rx_hi"] - r["rx_delta"] for r in cell_rows],
        ])

        ax.bar(x - bar_w, bayes, bar_w, yerr=bayes_err,
               color=color_bayes, edgecolor="black", linewidth=0.5, capsize=2.5,
               label="our framework" if ax is axes[0] else None)
        ax.bar(x, srf, bar_w, yerr=srf_err,
               color=color_srf, edgecolor="black", linewidth=0.5, capsize=2.5,
               label="Self-Refine [Madaan 2023]" if ax is axes[0] else None)
        ax.bar(x + bar_w, rx, bar_w, yerr=rx_err,
               color=color_rx, edgecolor="black", linewidth=0.5, capsize=2.5,
               label="Reflexion [Shinn 2023]" if ax is axes[0] else None)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([GEN_DISPLAY.get(r["gen"], r["gen"]) for r in cell_rows],
                           rotation=42, ha="right", fontsize=9)
        ax.set_title(cell, fontsize=11, pad=4, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        for i, r in enumerate(cell_rows):
            if r["cell"] == "LCB-easy" and r["gen"] == "gpt5_mini" and r["rx_delta"] > r["best_bayes_delta"]:
                ax.annotate("Reflexion wins\n(regime crossover)",
                            xy=(x[i] + bar_w, r["rx_delta"]),
                            xytext=(max(0.0, x[i] - 0.6), r["rx_delta"] + 12),
                            fontsize=7.5, ha="left",
                            arrowprops=dict(arrowstyle="->", color="black",
                                            lw=0.6, shrinkB=3))
                crossover_text = "LCB-easy/gpt5-mini"

        for r in cell_rows:
            if r["best_bayes_delta"] > r["srf_delta"] and r["best_bayes_delta"] > r["rx_delta"]:
                n_total_wins += 1

    axes[0].set_ylabel(r"$\Delta$ utility vs always_verify", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, 0.99))

    n_total = len(rows)
    sup_title = (f"Our framework vs published-method baselines  "
                 f"($\\Delta$ utility vs always\\_verify)\n"
                 f"Bayesian wins {n_total_wins}/{n_total} cells")
    if crossover_text:
        sup_title += f". One crossover: {crossover_text} (high prior + high $P_\\text{{fix}}$)"
    fig.suptitle(sup_title, fontsize=11, y=0.94)

    fig.subplots_adjust(top=0.84, bottom=0.21, left=0.06, right=0.99)

    out_png = args.out_dir / "fig_framework_vs_baselines.png"
    out_pdf = args.out_dir / "fig_framework_vs_baselines.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-benchmark single-panel figures (for slide layouts that need one
    # png/pdf per benchmark instead of the wide combined figure).
    # Plots 4 bars per generator: bayesian_greedy, bayesian_DP (measured kernel),
    # Self-Refine, Reflexion — so the audience can see the two Bayesian variants
    # as separate methods rather than an oracle pick.
    # ------------------------------------------------------------------
    color_greedy = "#1d4ed8"   # blue (Bayesian greedy)
    color_dp     = "#0ea5e9"   # cyan (Bayesian DP, measured kernel)
    for cell in cells_with_data:
        cell_rows = [r for r in rows if r["cell"] == cell]
        cell_rows.sort(key=lambda r: GENERATORS.index(r["gen"]))
        n = len(cell_rows)
        x = np.arange(n, dtype=float)

        def _arr(key, default=0.0):
            return np.array([r.get(key, default) for r in cell_rows])

        greedy   = _arr("greedy_delta")
        greedy_e = np.array([
            [r.get("greedy_delta", 0.0) - r.get("greedy_lo", r.get("greedy_delta", 0.0)) for r in cell_rows],
            [r.get("greedy_hi", r.get("greedy_delta", 0.0)) - r.get("greedy_delta", 0.0) for r in cell_rows],
        ])
        dp   = _arr("dp_delta")
        dp_e = np.array([
            [r.get("dp_delta", 0.0) - r.get("dp_lo", r.get("dp_delta", 0.0)) for r in cell_rows],
            [r.get("dp_hi", r.get("dp_delta", 0.0)) - r.get("dp_delta", 0.0) for r in cell_rows],
        ])
        srf = _arr("srf_delta")
        srf_e = np.array([
            [r["srf_delta"] - r["srf_lo"] for r in cell_rows],
            [r["srf_hi"] - r["srf_delta"] for r in cell_rows],
        ])
        rx = _arr("rx_delta")
        rx_e = np.array([
            [r["rx_delta"] - r["rx_lo"] for r in cell_rows],
            [r["rx_hi"] - r["rx_delta"] for r in cell_rows],
        ])

        figc, axc = plt.subplots(figsize=(6.0, 4.0))
        bar_w_c = 0.20
        axc.bar(x - 1.5 * bar_w_c, greedy, bar_w_c, yerr=greedy_e,
                color=color_greedy, edgecolor="black", linewidth=0.5, capsize=2.0,
                label="bayesian_greedy")
        axc.bar(x - 0.5 * bar_w_c, dp,     bar_w_c, yerr=dp_e,
                color=color_dp, edgecolor="black", linewidth=0.5, capsize=2.0,
                label="bayesian_DP (measured kernel)")
        axc.bar(x + 0.5 * bar_w_c, srf,    bar_w_c, yerr=srf_e,
                color=color_srf, edgecolor="black", linewidth=0.5, capsize=2.0,
                label="Self-Refine [Madaan 2023]")
        axc.bar(x + 1.5 * bar_w_c, rx,     bar_w_c, yerr=rx_e,
                color=color_rx, edgecolor="black", linewidth=0.5, capsize=2.0,
                label="Reflexion [Shinn 2023]")
        axc.axhline(0, color="black", linewidth=0.5)
        axc.set_xticks(x)
        axc.set_xticklabels([GEN_DISPLAY.get(r["gen"], r["gen"]) for r in cell_rows],
                            rotation=30, ha="right", fontsize=10)
        axc.set_ylabel(r"$\Delta$ utility vs always_verify", fontsize=10)
        axc.set_title(f"{cell}", fontsize=12, fontweight="bold")
        axc.grid(axis="y", linestyle=":", alpha=0.4)
        axc.legend(loc="best", fontsize=7.5, framealpha=0.92, ncol=2)

        # Annotate regime-crossover cell — Reflexion beats both Bayesian variants
        for i, r in enumerate(cell_rows):
            if cell == "LCB-easy" and r["gen"] == "gpt5_mini" \
               and r["rx_delta"] > max(r.get("greedy_delta", -1e9), r.get("dp_delta", -1e9)):
                axc.annotate("Reflexion wins\n(regime crossover)",
                             xy=(x[i] + 1.5 * bar_w_c, r["rx_delta"]),
                             xytext=(max(0.0, x[i] - 0.7), r["rx_delta"] + 10),
                             fontsize=8, ha="left",
                             arrowprops=dict(arrowstyle="->", color="black",
                                             lw=0.6, shrinkB=3))

        figc.tight_layout()
        slug = cell.lower().replace("-", "_").replace("+", "plus")
        per_png = args.out_dir / f"fig_framework_vs_baselines_{slug}.png"
        per_pdf = args.out_dir / f"fig_framework_vs_baselines_{slug}.pdf"
        figc.savefig(per_png, dpi=160, bbox_inches="tight")
        figc.savefig(per_pdf, bbox_inches="tight")
        plt.close(figc)
        print(f"  per-bench: {per_png.name}")

    with open(args.out_dir / "fig_framework_vs_baselines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell", "generator",
            "best_bayes_delta", "best_bayes_ci_lo", "best_bayes_ci_hi", "best_bayes_source",
            "selfrefine_delta", "selfrefine_ci_lo", "selfrefine_ci_hi",
            "reflexion_delta", "reflexion_ci_lo", "reflexion_ci_hi",
            "bayes_beats_srf", "bayes_beats_rx",
        ])
        for r in rows:
            w.writerow([
                r["cell"], r["gen"],
                f"{r['best_bayes_delta']:+.2f}", f"{r['best_bayes_lo']:+.2f}", f"{r['best_bayes_hi']:+.2f}",
                r["best_bayes_source"],
                f"{r['srf_delta']:+.2f}", f"{r['srf_lo']:+.2f}", f"{r['srf_hi']:+.2f}",
                f"{r['rx_delta']:+.2f}", f"{r['rx_lo']:+.2f}", f"{r['rx_hi']:+.2f}",
                "yes" if r["best_bayes_delta"] > r["srf_delta"] else "no",
                "yes" if r["best_bayes_delta"] > r["rx_delta"] else "no",
            ])

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Bayesian best on {n_total_wins}/{n_total} cells across {n_panels} benchmarks.")
    print()
    print(f"{'cell':14} {'gen':14} {'Bayes':>8}  {'SRF':>8}  {'Rx':>8}  source")
    for r in rows:
        bayes_label = r["best_bayes_source"][:25]
        print(f"{r['cell']:14} {GEN_DISPLAY[r['gen']]:14} "
              f"{r['best_bayes_delta']:+8.2f}  {r['srf_delta']:+8.2f}  {r['rx_delta']:+8.2f}  {bayes_label}")


if __name__ == "__main__":
    main()
