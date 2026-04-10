#!/usr/bin/env python3
"""Plot the L2 noise ablation results."""
import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib not installed")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default=str(Path(__file__).parent / "l2_noise_ablation.json"),
)
parser.add_argument(
    "--out",
    default=str(Path(__file__).parent / "l2_noise_ablation.png"),
)
args = parser.parse_args()

with open(args.input) as f:
    data = json.load(f)

p_flip_values = data["p_flip_values"]
results = data["results"]

policies = ["Bayesian", "Threshold(L2)", "Threshold(L1)", "Threshold(L3)", "Fixed"]
colors = {
    "Bayesian": "tab:blue",
    "Threshold(L2)": "tab:orange",
    "Threshold(L1)": "tab:green",
    "Threshold(L3)": "tab:red",
    "Fixed": "tab:gray",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Utility plot
for policy in policies:
    utilities = []
    errors = []
    for p in p_flip_values:
        trials = results[str(p)]
        utils = [t[policy]["avg_utility"] for t in trials]
        utilities.append(np.mean(utils))
        errors.append(np.std(utils) / np.sqrt(len(utils)))
    ax1.errorbar(
        p_flip_values, utilities, yerr=errors,
        marker="o", label=policy, color=colors[policy], linewidth=2,
    )

ax1.axhline(0, color="black", linestyle="--", alpha=0.3)
ax1.set_xlabel("L2 noise rate (p_flip)")
ax1.set_ylabel("Expected utility per episode")
ax1.set_title("Utility vs L2 noise")
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Cost plot
for policy in policies:
    costs = []
    for p in p_flip_values:
        trials = results[str(p)]
        cs = [t[policy]["avg_cost"] for t in trials]
        costs.append(np.mean(cs))
    ax2.plot(
        p_flip_values, costs,
        marker="s", label=policy, color=colors[policy], linewidth=2,
    )

ax2.set_xlabel("L2 noise rate (p_flip)")
ax2.set_ylabel("Average cost per episode")
ax2.set_title("Cost vs L2 noise")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

# Highlight crossover region
for ax in (ax1, ax2):
    ax.axvspan(0.05, 0.15, alpha=0.1, color="yellow", label=None)

fig.suptitle(
    "L2 noise ablation: Bayesian controller is robust; Threshold(L2) degrades",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved: {args.out}")
