#!/usr/bin/env python3
"""Plot belief trajectories for representative episodes.

Reads calibration data and the Bayesian controller, simulates a few episodes,
and produces a line plot of belief over time with actions annotated. This
visualizes the controller's decision process for the paper.

Usage:
    python plot_belief_trajectories.py --out belief_trajectories.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; install with: pip install matplotlib")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import (
    Action,
    BayesianController,
    CostModel,
)

DEFAULT_CALIBRATION_DATA = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "raw_results_v2.jsonl"
)
DEFAULT_LIKELIHOOD_TABLES = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "likelihood_tables.json"
)


def load_episodes(path: Path) -> dict[str, list[dict]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_instance[r["instance_id"]].append(r)
    for patches in by_instance.values():
        patches.sort(key=lambda r: r.get("step", r.get("patch_id", 0)))
    return dict(by_instance)


def simulate_one(
    controller: BayesianController,
    patches: list[dict],
    prior: float,
    max_steps: int = 12,
) -> tuple[list[float], list[str]]:
    """Return (beliefs, action_labels) along the trajectory.

    Enforces the no-repeat-critic-on-same-patch rule (deterministic critics).
    """
    b = prior
    beliefs = [b]
    labels = ["init"]
    patch_idx = 0
    used_critics: set[str] = set()

    level_map = {
        Action.CRITIC_L0: "L0_syntax",
        Action.CRITIC_L1: "L1_lint",
        Action.CRITIC_L2: "L2_fast_test",
        Action.CRITIC_L3: "L3_llm_review",
    }

    for step in range(max_steps):
        action = controller.select_action(b, step)
        if action is None:
            labels.append("give_up")
            beliefs.append(b)
            break

        # Fall back if chosen critic is already used on this patch
        if action in level_map and level_map[action] in used_critics:
            q_ver = b * controller.costs.reward - controller.costs.c_ver
            action = Action.VERIFY if q_ver >= 0 else Action.GENERATE

        if action == Action.VERIFY:
            labels.append("verify")
            beliefs.append(b)
            break

        if action == Action.GENERATE:
            patch_idx = min(patch_idx + 1, len(patches) - 1)
            used_critics.clear()
            b = controller.update_belief_after_generation(b)
            labels.append("gen")
            beliefs.append(b)
            continue

        if action in level_map:
            level = level_map[action]
            used_critics.add(level)
            current = patches[min(patch_idx, len(patches) - 1)]
            passed = current["critic_results"].get(level, {}).get("passed", False)
            b = controller.update_belief(b, level, passed)
            short = level.split("_")[0]
            labels.append(f"{short}{'+' if passed else '-'}")
            beliefs.append(b)

    return beliefs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot belief trajectories.")
    parser.add_argument("--calibration-data", default=str(DEFAULT_CALIBRATION_DATA))
    parser.add_argument("--likelihood-tables", default=str(DEFAULT_LIKELIHOOD_TABLES))
    parser.add_argument("--out", default="belief_trajectories.png")
    parser.add_argument("--n-examples", type=int, default=6)
    args = parser.parse_args()

    episodes = load_episodes(Path(args.calibration_data))
    all_patches = [p for ps in episodes.values() for p in ps]
    n_correct = sum(1 for p in all_patches if p["ground_truth"] == 1)
    prior = max(n_correct / len(all_patches), 0.05)

    costs = CostModel()
    controller = BayesianController.from_likelihood_tables(
        args.likelihood_tables, costs=costs, horizon=12,
    )

    # Pick a mix: some Y=1 (successful) and some Y=0 (failed)
    y1_instances = [iid for iid, ps in episodes.items() if any(p["ground_truth"] == 1 for p in ps)]
    y0_instances = [iid for iid, ps in episodes.items() if all(p["ground_truth"] == 0 for p in ps)]

    n_y1 = min(args.n_examples // 2, len(y1_instances))
    n_y0 = args.n_examples - n_y1
    selected = y1_instances[:n_y1] + y0_instances[:n_y0]

    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, iid in zip(axes, selected):
        patches = episodes[iid]
        beliefs, labels = simulate_one(controller, patches, prior)

        steps = list(range(len(beliefs)))
        any_y1 = any(p["ground_truth"] == 1 for p in patches)
        color = "tab:green" if any_y1 else "tab:red"

        ax.plot(steps, beliefs, marker="o", color=color, linewidth=1.5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (steps[i], beliefs[i]),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=7, ha="center")

        # Threshold lines
        verify_threshold = costs.c_ver / costs.reward
        ax.axhline(verify_threshold, linestyle="--", color="gray", alpha=0.4, linewidth=0.8)
        ax.axhline(prior, linestyle=":", color="gray", alpha=0.4, linewidth=0.8)

        ax.set_ylim(-0.05, 1.05)
        y_str = "".join(str(p["ground_truth"]) for p in patches)
        ax.set_ylabel(f"b_t\n({iid.split('__')[1]})\nY={y_str}", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle("Bayesian Controller: belief trajectories", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
