#!/usr/bin/env python3
"""L2 noise ablation: sweep p_flip and see how it affects each policy.

Key question: when L2 stops being near-perfect, how does Bayesian vs
threshold performance compare?

Procedure:
  1. Load sympy calibration data (231 patches, 4 critics).
  2. For each p_flip in {0, 0.05, 0.10, 0.20, 0.30, 0.50}:
     a. Flip L2 outcomes independently with probability p_flip.
     b. Recompute likelihood tables from the noised data.
     c. Re-solve the Bayesian controller with the new tables.
     d. Run the simulation with the noised data.
     e. Record pass@1, cost, utility for each policy.
  3. Plot: utility vs p_flip for Bayesian, Threshold(L2), Threshold(L1), Fixed.

This directly tests the characterization: Bayesian should win as L2 degrades.

Usage:
    python l2_noise_ablation.py --seed 42 --out ablation_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import (
    Action,
    BayesianController,
    CostModel,
    CriticLikelihood,
    TransitionKernel,
)
from evaluation.run_simulation import (
    EpisodeResult,
    run_bayesian_policy,
    run_fixed_pipeline,
    run_threshold_policy,
    print_results,
)

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "raw_results_v2.jsonl"
)
DEFAULT_TABLES = (
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


def noise_l2(
    episodes: dict[str, list[dict]],
    p_flip: float,
    rng: random.Random,
) -> dict[str, list[dict]]:
    """Return a deep copy of episodes with L2 outcomes flipped independently."""
    noised = {}
    for iid, patches in episodes.items():
        new_patches = []
        for p in patches:
            p_copy = deepcopy(p)
            l2 = p_copy["critic_results"].get("L2_fast_test")
            if l2 is not None and rng.random() < p_flip:
                l2["passed"] = not l2["passed"]
                l2["detail"] = f"[NOISE] flipped: {l2.get('detail', '')[:80]}"
            new_patches.append(p_copy)
        noised[iid] = new_patches
    return noised


def compute_likelihoods_from_episodes(
    episodes: dict[str, list[dict]],
    smoothing: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Recompute P(pass|Y) likelihoods from (possibly noised) episode data."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    critics = ["L0_syntax", "L1_lint", "L2_fast_test", "L3_llm_review"]

    for patches in episodes.values():
        for p in patches:
            y = p["ground_truth"]
            for level in critics:
                c = p["critic_results"].get(level)
                if c is None:
                    continue
                passed = c.get("passed", False)
                if passed and y == 1:
                    counts[level]["tp"] += 1
                elif passed and y == 0:
                    counts[level]["fp"] += 1
                elif not passed and y == 1:
                    counts[level]["fn"] += 1
                else:
                    counts[level]["tn"] += 1

    likelihoods = {}
    for level in critics:
        c = counts[level]
        n_correct = c["tp"] + c["fn"]
        n_incorrect = c["fp"] + c["tn"]
        p_pass_correct = (c["tp"] + smoothing) / (n_correct + 2 * smoothing)
        p_pass_incorrect = (c["fp"] + smoothing) / (n_incorrect + 2 * smoothing)
        likelihoods[level] = {
            "p_pass_given_correct": round(p_pass_correct, 4),
            "p_pass_given_incorrect": round(p_pass_incorrect, 4),
        }
    return likelihoods


def build_controller_from_tables(
    likelihoods: dict,
    transition: dict,
    costs: CostModel,
) -> BayesianController:
    """Build a controller directly from in-memory likelihood tables."""
    critics = {}
    for level, lk in likelihoods.items():
        critics[level] = CriticLikelihood(
            p_pass_given_correct=lk["p_pass_given_correct"],
            p_pass_given_incorrect=lk["p_pass_given_incorrect"],
        )
    tk = TransitionKernel(
        p_fix=transition["p_fix_given_broken"],
        p_break=transition["p_break_given_correct"],
    )
    return BayesianController(
        critic_likelihoods=critics,
        transition=tk,
        costs=costs,
        horizon=10,
        grid_size=1000,
    )


def run_all_policies(
    controller: BayesianController,
    episodes: dict[str, list[dict]],
    costs: CostModel,
    prior: float,
) -> dict[str, dict]:
    """Run Bayesian + baselines, return summary dict."""
    b_res, f_res, t1, t2, t3 = [], [], [], [], []
    for iid, patches in episodes.items():
        b_res.append(run_bayesian_policy(controller, patches, costs, prior))
        f_res.append(run_fixed_pipeline(patches, costs))
        t1.append(run_threshold_policy(patches, costs, "L1_lint"))
        t2.append(run_threshold_policy(patches, costs, "L2_fast_test"))
        t3.append(run_threshold_policy(patches, costs, "L3_llm_review"))

    def _summary(name: str, results: list[EpisodeResult]) -> dict:
        n = len(results)
        resolved = sum(1 for r in results if r.resolved)
        costs_list = [r.total_cost for r in results]
        utils = [(costs.reward if r.resolved else 0) - r.total_cost for r in results]
        return {
            "policy": name,
            "pass_rate": resolved / n,
            "avg_cost": sum(costs_list) / n,
            "avg_utility": float(np.mean(utils)),
            "std_utility": float(np.std(utils) / np.sqrt(n)) if n > 1 else 0.0,
            "verify_per_ep": sum(r.n_verify_calls for r in results) / n,
        }

    return {
        "Bayesian": _summary("Bayesian", b_res),
        "Fixed": _summary("Fixed", f_res),
        "Threshold(L1)": _summary("Threshold(L1)", t1),
        "Threshold(L2)": _summary("Threshold(L2)", t2),
        "Threshold(L3)": _summary("Threshold(L3)", t3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 noise ablation.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--tables", default=str(DEFAULT_TABLES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--p-flip-values",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    )
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--out", default="ablation_results.json")
    args = parser.parse_args()

    episodes = load_episodes(Path(args.data))
    all_patches = [p for ps in episodes.values() for p in ps]
    base_rate = sum(1 for p in all_patches if p["ground_truth"] == 1) / len(all_patches)
    prior = max(base_rate, 0.05)
    print(f"Loaded {len(episodes)} instances, {len(all_patches)} patches, base rate {base_rate:.3f}")

    # Load fixed transition kernel (same across all p_flip values)
    with open(args.tables) as f:
        base_tables = json.load(f)
    transition = base_tables["generator_transition"]
    costs = CostModel()

    results_by_pflip: dict[float, list[dict]] = {}

    for p_flip in args.p_flip_values:
        trial_summaries = []
        for trial in range(args.n_trials):
            rng = random.Random(args.seed + trial)
            noised = noise_l2(episodes, p_flip, rng)
            likelihoods = compute_likelihoods_from_episodes(noised)
            controller = build_controller_from_tables(likelihoods, transition, costs)
            summary = run_all_policies(controller, noised, costs, prior)
            trial_summaries.append(summary)
        results_by_pflip[p_flip] = trial_summaries

        # Print average across trials
        print(f"\np_flip = {p_flip:.2f}")
        print(f"{'Policy':<20} {'Pass@1':>8} {'Cost':>8} {'Utility':>10} {'±SE':>8}")
        print("-" * 60)
        for policy in ["Bayesian", "Threshold(L2)", "Threshold(L1)", "Threshold(L3)", "Fixed"]:
            pass_rates = [t[policy]["pass_rate"] for t in trial_summaries]
            costs_list = [t[policy]["avg_cost"] for t in trial_summaries]
            utils = [t[policy]["avg_utility"] for t in trial_summaries]
            print(f"{policy:<20} {100*np.mean(pass_rates):>7.1f}% "
                  f"{np.mean(costs_list):>8.1f} {np.mean(utils):>10.1f} "
                  f"{np.std(utils)/np.sqrt(len(utils)):>8.2f}")

    # Save
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump({
            "p_flip_values": args.p_flip_values,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "prior": prior,
            "results": {str(k): v for k, v in results_by_pflip.items()},
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
