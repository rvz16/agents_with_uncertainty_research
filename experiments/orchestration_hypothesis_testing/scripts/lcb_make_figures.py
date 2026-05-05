"""Generate paper-ready figures from PAPER_TABLE.json.

Produces:
  fig1_headline.pdf       — bar chart of bayesian_greedy Δ vs always_verify
                              for each (generator, difficulty) with 95% CIs
  fig2_l3_heatmap.pdf     — (generator × reviewer) L3 gap heatmap on hard
  fig3_invariance.pdf     — bayesian_greedy invariance vs threshold_L3 swing
                              under reviewer choice (medium qwen3+haiku)

Usage:
  python lcb_make_figures.py --paper-table data/PAPER_TABLE.json --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GEN_ORDER = ["sonnet45", "gpt5_mini", "qwen3_coder", "haiku45"]   # rough capability order
DIFF_ORDER = ["hard", "medium"]
REV_ORDER = ["haiku45", "gpt4omini", "sonnet45"]


def fig_headline(table: dict, out_path: Path) -> None:
    """Bar chart: bayesian_greedy Δ for each (gen, difficulty) with 95% CIs."""
    bars = []  # (label, diff, lo, hi)
    for diff in DIFF_ORDER:
        if diff not in table:
            continue
        for gen in GEN_ORDER:
            if gen not in table[diff]:
                continue
            rev = table[diff][gen].get("haiku45_default")
            if not rev:
                continue
            p = rev["policies"].get("bayesian_greedy")
            if not p or p["diff_vs_always_verify"] is None:
                continue
            bars.append((f"{gen}\n{diff}", p["diff_vs_always_verify"],
                          p["ci95_lo"], p["ci95_hi"]))
    if not bars:
        print(f"  fig1: no data, skipping")
        return

    labels = [b[0] for b in bars]
    diffs = [b[1] for b in bars]
    lo = [b[1] - b[2] for b in bars]
    hi = [b[3] - b[1] for b in bars]
    yerr = np.array([lo, hi])

    fig, ax = plt.subplots(figsize=(max(7, len(bars) * 1.2), 4.5))
    colors = ["#3b82f6" if "hard" in l else "#f59e0b" for l in labels]
    bars_obj = ax.bar(range(len(bars)), diffs, yerr=yerr, color=colors,
                       capsize=5, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Δ utility vs always_verify", fontsize=11)
    ax.set_title("Bayesian Greedy: pre-registered effect, replicated across generators × difficulties",
                  fontsize=11, pad=10)
    for i, (d, l_lo, l_hi) in enumerate(zip(diffs, lo, hi)):
        ax.text(i, d + l_hi + 0.4, f"+{d:.1f}", ha="center", fontsize=9, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#3b82f6"),
                plt.Rectangle((0, 0), 1, 1, color="#f59e0b")]
    ax.legend(handles, ["hard", "medium"], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_l3_heatmap(table: dict, out_path: Path) -> None:
    """Heatmap of L3 gap (gen × reviewer)."""
    # Use HARD difficulty for this fig (most complete cube)
    if "hard" not in table:
        return
    gens = [g for g in GEN_ORDER if g in table["hard"]]
    revs = REV_ORDER
    grid = np.full((len(gens), len(revs)), np.nan)
    for i, gen in enumerate(gens):
        for j, rev in enumerate(revs):
            row = table["hard"][gen].get(rev)
            if row is None:
                continue
            grid[i, j] = row.get("L3_gap_used") or 0
    if np.isnan(grid).all():
        return

    fig, ax = plt.subplots(figsize=(6, 0.8 + len(gens)*0.6))
    vmax = max(0.5, np.nanmax(np.abs(grid)))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(revs)))
    ax.set_xticklabels(revs)
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels(gens)
    ax.set_xlabel("L3 reviewer model")
    ax.set_ylabel("Generator")
    ax.set_title("L3 informativeness gap (LCB-hard)\nself-review highlighted with bold border", fontsize=10)
    for i in range(len(gens)):
        for j in range(len(revs)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    color="white" if abs(v) > vmax*0.55 else "black", fontsize=10)
            # Highlight self-review cells
            if gens[i] == revs[j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                            edgecolor="black", linewidth=2.5))
    fig.colorbar(im, ax=ax, label="P(L3 pass | Y=1) − P(L3 pass | Y=0)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_invariance(table: dict, out_path: Path) -> None:
    """Show bayesian_greedy is invariant under reviewer choice while
    threshold_L3 swings. Use the cells where we have all 3 reviewers."""
    # Find cells with complete 3-reviewer data
    cells = []
    for diff in DIFF_ORDER:
        if diff not in table:
            continue
        for gen in GEN_ORDER:
            if gen not in table[diff]:
                continue
            revs = table[diff][gen]
            if not all(r in revs for r in REV_ORDER):
                continue
            cells.append((diff, gen, revs))
    if not cells:
        return

    fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 4), sharey=True)
    if len(cells) == 1:
        axes = [axes]
    for ax, (diff, gen, revs) in zip(axes, cells):
        bg = [revs[r]["policies"].get("bayesian_greedy", {}).get("diff_vs_always_verify") or 0
              for r in REV_ORDER]
        tl3 = [revs[r]["policies"].get("threshold_L3", {}).get("diff_vs_always_verify") or 0
               for r in REV_ORDER]
        x = np.arange(len(REV_ORDER))
        w = 0.35
        ax.bar(x - w/2, bg, w, label="bayesian_greedy", color="#1d4ed8", edgecolor="black", linewidth=0.7)
        ax.bar(x + w/2, tl3, w, label="threshold_L3", color="#dc2626", edgecolor="black", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(REV_ORDER, fontsize=9)
        ax.set_title(f"{gen} / {diff}", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Δ utility vs always_verify")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Bayesian Greedy is invariant to L3 reviewer choice; threshold_L3 swings.",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-table", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = json.loads(args.paper_table.read_text())
    print(f"  loaded table: difficulties={list(table.keys())}")
    fig_headline(table, args.out_dir / "fig1_headline.pdf")
    fig_l3_heatmap(table, args.out_dir / "fig2_l3_heatmap.pdf")
    fig_invariance(table, args.out_dir / "fig3_invariance.pdf")
    # Also produce png variants for easy embedding
    fig_headline(table, args.out_dir / "fig1_headline.png")
    fig_l3_heatmap(table, args.out_dir / "fig2_l3_heatmap.png")
    fig_invariance(table, args.out_dir / "fig3_invariance.png")


if __name__ == "__main__":
    main()
