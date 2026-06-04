"""Generate summary figures for the EXPERIMENTS_SUMMARY writeup.

Two PNGs:
  fig_methods_comparison.png — bar chart Ū per method per cell.
  fig_thompson_posterior.png — why Thompson works on gpt5_mini, ties on haiku45.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent

# ---- 1. Methods comparison bar chart ----

cells = ["LCB-hard\n/ haiku45", "CC\n/ haiku45", "HumanEvalFix\n/ haiku45",
          "CC\n/ gpt5_mini", "CC\n/ sonnet45"]
methods = ["Baseline\n(always_verify)", "Offline BDP", "Conditional", "Thompson BDP"]
data = np.array([
    # baseline, offline, conditional, thompson
    [-38.50, -11.45, -11.70, -11.70 ],  # LCB-hard / haiku45
    [+6.25,  +13.60, +13.60, +13.60 ],  # CC / haiku45
    [+78.50, +77.85, +70.80, +77.00 ],  # HumanEvalFix / haiku45
    [+0.50,  +13.60, +13.60, +22.45 ],  # CC / gpt5_mini
    [+26.50, +13.60, +13.60, +16.20 ],  # CC / sonnet45
])

# Bonus: Thompson combinations on CC / gpt5_mini
bonus_methods = ["Offline", "Thompson", "Thompson_conditional"]
bonus_data = [+13.60, +22.45, -0.40]

fig, ax = plt.subplots(figsize=(11, 5.5))
n_cells, n_methods = data.shape
x = np.arange(n_cells)
width = 0.20
colors = ["#999999", "#4477AA", "#66CCEE", "#EE6677"]

for i, m in enumerate(methods):
    vals = data[:, i].copy()
    mask = ~np.isnan(vals)
    bars = ax.bar(x[mask] + (i - 1.5) * width, vals[mask], width,
                    label=m, color=colors[i], edgecolor="black", linewidth=0.5)
    for j, bar in zip(np.where(mask)[0], bars):
        v = vals[j]
        offset = 1.5 if v >= 0 else -3.0
        ax.text(bar.get_x() + bar.get_width()/2, v + offset, f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(cells, fontsize=10)
ax.set_ylabel(r"$\bar U$ (utility per instance)", fontsize=11)
ax.set_title("Live results: 4 cells × 4 methods (n=20 each)", fontsize=12)
ax.legend(loc="lower right", fontsize=9, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.annotate("Thompson wins\n(+8.85 vs offline)", xy=(3 + 1.5*width, 22.45),
             xytext=(2.0, 50), fontsize=9, color="#CC2244",
             arrowprops=dict(arrowstyle="->", color="#CC2244", lw=1.3))

plt.tight_layout()
out1 = OUT_DIR / "fig_methods_comparison.png"
plt.savefig(out1, dpi=140, bbox_inches="tight")
print(f"saved {out1}")
plt.close()

# ---- 2. Why Thompson works on one cell but not the other ----

fig, ax = plt.subplots(figsize=(10, 5))

from scipy.stats import beta as beta_dist

# CC / gpt5_mini: 20 fixes / 186 broken
a1, b1 = 20 + 1, 166 + 1
# CC / haiku45:   11 fixes / 363 broken
a2, b2 = 11 + 1, 352 + 1

xs = np.linspace(0.001, 0.30, 800)
y_gpt5 = beta_dist.pdf(xs, a1, b1)
y_haiku = beta_dist.pdf(xs, a2, b2)

ax.fill_between(xs, 0, y_gpt5, alpha=0.30, color="#EE6677",
                  label=f"CC / gpt5_mini posterior  Beta({a1}, {b1})")
ax.plot(xs, y_gpt5, color="#EE6677", linewidth=2)
ax.fill_between(xs, 0, y_haiku, alpha=0.30, color="#4477AA",
                  label=f"CC / haiku45 posterior  Beta({a2}, {b2})")
ax.plot(xs, y_haiku, color="#4477AA", linewidth=2)

# Bail threshold (illustrative — actually depends on cost vector)
bail_thr = 0.13
ax.axvline(bail_thr, color="black", linestyle="--", linewidth=1.5,
            label="Approx. bail threshold")
ax.text(bail_thr + 0.005, ax.get_ylim()[1] * 0.85,
         "DP bails\n← here", fontsize=9)

ax.set_xlabel(r"Sampled $p_\mathrm{fix}$", fontsize=11)
ax.set_ylabel("Posterior density", fontsize=11)
ax.set_title("Why Thompson wins on gpt5_mini and ties on haiku45",
              fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 0.25)
plt.tight_layout()
out2 = OUT_DIR / "fig_thompson_posterior.png"
plt.savefig(out2, dpi=140, bbox_inches="tight")
print(f"saved {out2}")
plt.close()

# ---- 3. Thompson combinations on CC/gpt5_mini ----

fig, ax = plt.subplots(figsize=(7, 4.5))
bcolors = ["#4477AA", "#EE6677", "#AA3344"]
bars = ax.bar(bonus_methods, bonus_data, color=bcolors,
              edgecolor="black", linewidth=0.6)
for bar, v in zip(bars, bonus_data):
    offset = 1.0 if v >= 0 else -2.5
    ax.text(bar.get_x() + bar.get_width()/2, v + offset, f"{v:+.2f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=10,
            fontweight="bold")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel(r"$\bar U$ (utility per instance)", fontsize=11)
ax.set_title("Thompson variants on CC / gpt5_mini (n=20)\n"
              "Vanilla wins; combining with conditional loses",
              fontsize=11)
ax.annotate("paired Δ vs Offline:\n+8.85", xy=(1, 22.45),
             xytext=(1.5, 18), fontsize=9, color="#CC2244",
             arrowprops=dict(arrowstyle="->", color="#CC2244", lw=1.2))
ax.annotate("paired Δ vs Offline:\n−14.00 (CI excludes 0)",
             xy=(2, -0.4), xytext=(0.6, -10), fontsize=9, color="#AA3344",
             arrowprops=dict(arrowstyle="->", color="#AA3344", lw=1.2))
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-20, 30)
plt.tight_layout()
out3 = OUT_DIR / "fig_thompson_variants.png"
plt.savefig(out3, dpi=140, bbox_inches="tight")
print(f"saved {out3}")
plt.close()
