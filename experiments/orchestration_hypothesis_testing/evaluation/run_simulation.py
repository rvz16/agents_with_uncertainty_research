#!/usr/bin/env python3
"""Simulate and compare orchestration policies on calibration data.

Instead of re-running the LLM, this uses the calibration data as a simulator:
- We know which patches pass/fail each critic level and the ground truth.
- We can replay different orchestration policies on the same data.
- This gives a fair comparison between Bayesian controller and baselines.

The simulation treats the calibration data as episodes:
    - Each instance is one episode.
    - The agent starts with prior b_0 = base_rate (from calibration data).
    - At each step, the policy selects an action.
    - Critic outcomes are drawn from the actual calibration data.
    - Generation "draws" the next patch from the instance's patch sequence.
    - Verification checks ground truth Y.

Usage:
    python run_simulation.py
    python run_simulation.py --likelihood-tables path/to/likelihood_tables.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import (
    Action,
    BayesianController,
    CostModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CALIBRATION_DATA = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "raw_results_v3.jsonl"
)
DEFAULT_LIKELIHOOD_TABLES = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "likelihood_tables.json"
)


@dataclass(frozen=True)
class EpisodeResult:
    """Result of running one policy on one instance."""
    instance_id: str
    resolved: bool           # Did the submitted patch pass?
    total_cost: float        # Sum of all action costs
    n_gen_calls: int         # Number of generation calls
    n_critic_calls: int      # Number of critic calls (all levels)
    n_verify_calls: int      # Number of verifier calls
    trajectory: list[str]    # Sequence of actions taken
    final_belief: float      # Belief at time of verification/termination


def load_episodes(data_path: Path) -> dict[str, list[dict]]:
    """Load calibration data and group by instance."""
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    episodes: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = r.get("instance_id") or r.get("question_id") or "?"
        episodes[key].append(r)

    # Sort patches within each instance by step or patch_id
    for patches in episodes.values():
        patches.sort(key=lambda r: r.get("step", r.get("patch_id", 0)))

    return dict(episodes)


def run_bayesian_policy(
    controller: BayesianController,
    patches: list[dict],
    costs: CostModel,
    prior: float,
    max_steps: int = 10,
) -> EpisodeResult:
    """Run the Bayesian controller on one instance.

    The controller selects actions based on belief state.
    Critic outcomes come from the calibration data.

    IMPORTANT: critics are deterministic on a given patch (running lint twice
    gives the same answer). We enforce this by tracking which critics have
    already been used on the current patch_idx. If the controller picks a
    critic it has already used on this patch, we force it to either generate
    (to get a new patch) or verify (to terminate). This prevents the fallacy
    of treating repeated critic calls as independent samples.
    """
    b = prior
    total_cost = 0.0
    n_gen = 0
    n_critic = 0
    n_verify = 0
    trajectory = []
    patch_idx = 0
    used_critics: set[str] = set()  # Critics already run on current patch

    def _level_from_action(a: Action) -> str | None:
        return {
            Action.CRITIC_L0: "L0_syntax",
            Action.CRITIC_L1: "L1_lint",
            Action.CRITIC_L2: "L2_fast_test",
            Action.CRITIC_L3: "L3_llm_review",
            Action.CRITIC_L4: "L4_mypy",
        }.get(a)

    def _cost_for_action(a: Action) -> float:
        return {
            Action.CRITIC_L0: costs.c_crit_l0,
            Action.CRITIC_L1: costs.c_crit_l1,
            Action.CRITIC_L2: costs.c_crit_l2,
            Action.CRITIC_L3: costs.c_crit_l3,
            Action.CRITIC_L4: costs.c_crit_l4,
        }.get(a, 0.0)

    for step in range(max_steps):
        action = controller.select_action(b, step)

        if action is None:
            trajectory.append("give_up")
            break

        # If the chosen critic has already been used on this patch, the
        # observation would be deterministic. Fall back to the next-best
        # available action.
        if action in (Action.CRITIC_L0, Action.CRITIC_L1,
                      Action.CRITIC_L2, Action.CRITIC_L3, Action.CRITIC_L4):
            level = _level_from_action(action)
            if level in used_critics:
                # Pick the best remaining action: verify or generate
                q_ver = b * costs.reward - costs.c_ver
                q_gen = -costs.c_gen + (
                    b * costs.reward * (1 - controller.transition.p_break)
                    + (1 - b) * costs.reward * controller.transition.p_fix
                    - costs.c_ver
                )
                action = Action.VERIFY if q_ver >= q_gen else Action.GENERATE
                trajectory.append(f"repeat_{level}->{'ver' if action == Action.VERIFY else 'gen'}")

        if action == Action.VERIFY:
            total_cost += costs.c_ver
            n_verify += 1
            trajectory.append("verify")
            current_patch = patches[min(patch_idx, len(patches) - 1)]
            resolved = current_patch["ground_truth"] == 1
            return EpisodeResult(
                instance_id=(patches[0].get("instance_id") or patches[0].get("question_id") or "?"),
                resolved=resolved,
                total_cost=total_cost,
                n_gen_calls=n_gen,
                n_critic_calls=n_critic,
                n_verify_calls=n_verify,
                trajectory=trajectory,
                final_belief=b,
            )

        elif action == Action.GENERATE:
            total_cost += costs.c_gen
            n_gen += 1
            trajectory.append("generate")
            patch_idx = min(patch_idx + 1, len(patches) - 1)
            used_critics.clear()  # new patch, fresh critic budget
            b = controller.update_belief_after_generation(b)

        elif action in (Action.CRITIC_L0, Action.CRITIC_L1,
                        Action.CRITIC_L2, Action.CRITIC_L3, Action.CRITIC_L4):
            level = _level_from_action(action)
            cost = _cost_for_action(action)
            total_cost += cost
            n_critic += 1
            used_critics.add(level)
            trajectory.append(f"critic_{level}")
            current_patch = patches[min(patch_idx, len(patches) - 1)]
            passed = current_patch["critic_results"][level]["passed"]
            b = controller.update_belief(b, level, passed)

    return EpisodeResult(
        instance_id=(patches[0].get("instance_id") or patches[0].get("question_id") or "?"),
        resolved=False,
        total_cost=total_cost,
        n_gen_calls=n_gen,
        n_critic_calls=n_critic,
        n_verify_calls=n_verify,
        trajectory=trajectory,
        final_belief=b,
    )


def run_fixed_pipeline(
    patches: list[dict],
    costs: CostModel,
    max_attempts: int = 3,
) -> EpisodeResult:
    """Fixed pipeline: for each patch, lint then verify if lint passes."""
    total_cost = 0.0
    n_gen = 0
    n_critic = 0
    n_verify = 0
    trajectory = []

    for idx, patch in enumerate(patches[:max_attempts]):
        if idx > 0:
            total_cost += costs.c_gen
            n_gen += 1
            trajectory.append("generate")

        # Run lint
        total_cost += costs.c_crit_l1
        n_critic += 1
        lint_pass = patch["critic_results"]["L1_lint"]["passed"]
        trajectory.append("critic_L1_lint")

        if lint_pass:
            # Verify
            total_cost += costs.c_ver
            n_verify += 1
            trajectory.append("verify")
            if patch["ground_truth"] == 1:
                return EpisodeResult(
                    instance_id=(patches[0].get("instance_id") or patches[0].get("question_id") or "?"),
                    resolved=True,
                    total_cost=total_cost,
                    n_gen_calls=n_gen,
                    n_critic_calls=n_critic,
                    n_verify_calls=n_verify,
                    trajectory=trajectory,
                    final_belief=0.0,
                )

    # Last resort: verify the last patch regardless
    last_patch = patches[min(max_attempts - 1, len(patches) - 1)]
    total_cost += costs.c_ver
    n_verify += 1
    trajectory.append("verify_final")

    return EpisodeResult(
        instance_id=(patches[0].get("instance_id") or patches[0].get("question_id") or "?"),
        resolved=last_patch["ground_truth"] == 1,
        total_cost=total_cost,
        n_gen_calls=n_gen,
        n_critic_calls=n_critic,
        n_verify_calls=n_verify,
        trajectory=trajectory,
        final_belief=0.0,
    )


def run_threshold_policy(
    patches: list[dict],
    costs: CostModel,
    critic_level: str = "L1_lint",
    max_attempts: int = 3,
) -> EpisodeResult:
    """Threshold policy: run critic, verify if pass, regenerate if fail."""
    total_cost = 0.0
    n_gen = 0
    n_critic = 0
    n_verify = 0
    trajectory = []
    critic_cost = {"L0_syntax": costs.c_crit_l0, "L1_lint": costs.c_crit_l1, "L2_fast_test": costs.c_crit_l2}

    for idx, patch in enumerate(patches[:max_attempts]):
        if idx > 0:
            total_cost += costs.c_gen
            n_gen += 1
            trajectory.append("generate")

        # Run critic
        total_cost += critic_cost.get(critic_level, costs.c_crit_l1)
        n_critic += 1
        passed = patch["critic_results"][critic_level]["passed"]
        trajectory.append(f"critic_{critic_level}")

        if passed:
            total_cost += costs.c_ver
            n_verify += 1
            trajectory.append("verify")
            if patch["ground_truth"] == 1:
                return EpisodeResult(
                    instance_id=(patches[0].get("instance_id") or patches[0].get("question_id") or "?"),
                    resolved=True,
                    total_cost=total_cost,
                    n_gen_calls=n_gen,
                    n_critic_calls=n_critic,
                    n_verify_calls=n_verify,
                    trajectory=trajectory,
                    final_belief=0.0,
                )

    return EpisodeResult(
        instance_id=(patches[0].get("instance_id") or patches[0].get("question_id") or "?"),
        resolved=False,
        total_cost=total_cost,
        n_gen_calls=n_gen,
        n_critic_calls=n_critic,
        n_verify_calls=n_verify,
        trajectory=trajectory,
        final_belief=0.0,
    )


def print_results(
    name: str,
    results: list[EpisodeResult],
    costs: CostModel,
) -> dict[str, float]:
    """Print summary statistics for a policy."""
    n = len(results)
    resolved = sum(1 for r in results if r.resolved)
    total_cost = sum(r.total_cost for r in results)
    avg_cost = total_cost / n if n else 0
    total_verify = sum(r.n_verify_calls for r in results)
    total_critic = sum(r.n_critic_calls for r in results)
    total_gen = sum(r.n_gen_calls for r in results)
    pass_rate = resolved / n if n else 0

    # Expected utility per episode
    utilities = []
    for r in results:
        u = (costs.reward if r.resolved else 0) - r.total_cost
        utilities.append(u)
    avg_utility = np.mean(utilities)
    std_utility = np.std(utilities) / np.sqrt(n) if n > 1 else 0

    print(f"\n{'='*60}")
    print(f"Policy: {name}")
    print(f"{'='*60}")
    print(f"  Episodes:        {n}")
    print(f"  Resolved:        {resolved} ({100*pass_rate:.1f}%)")
    print(f"  Avg cost:        {avg_cost:.1f}")
    print(f"  Total verify:    {total_verify} ({total_verify/n:.1f}/ep)")
    print(f"  Total critic:    {total_critic} ({total_critic/n:.1f}/ep)")
    print(f"  Total gen:       {total_gen} ({total_gen/n:.1f}/ep)")
    print(f"  Avg utility:     {avg_utility:.1f} +/- {std_utility:.1f}")
    print(f"  Cost-adj pass:   {pass_rate/avg_cost*100:.2f}%/unit" if avg_cost > 0 else "  Cost-adj pass:   N/A")

    return {
        "policy": name,
        "n_episodes": n,
        "pass_rate": pass_rate,
        "avg_cost": avg_cost,
        "avg_utility": avg_utility,
        "std_utility": std_utility,
        "verify_per_ep": total_verify / n if n else 0,
        "critic_per_ep": total_critic / n if n else 0,
        "gen_per_ep": total_gen / n if n else 0,
    }


def _rebuild_controller_without_levels(
    tables_path: str,
    exclude_levels: list[str],
    costs: CostModel,
    horizon: int,
) -> BayesianController:
    """Load likelihood tables, drop specified critic levels, rebuild controller.

    Used to test the partial-information regime (e.g., drop L2 so no critic is
    near-perfect).
    """
    with open(tables_path) as f:
        data = json.load(f)

    filtered = {
        level: lk
        for level, lk in data["critic_likelihoods"].items()
        if level not in exclude_levels
    }
    # Write to a temp file and reload
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(
            {
                "critic_likelihoods": filtered,
                "generator_transition": data["generator_transition"],
            },
            f,
        )
        tmp_path = f.name

    ctrl = BayesianController.from_likelihood_tables(
        tmp_path, costs=costs, horizon=horizon
    )
    Path(tmp_path).unlink(missing_ok=True)
    return ctrl


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate orchestration policies.")
    parser.add_argument("--calibration-data", default=str(DEFAULT_CALIBRATION_DATA))
    parser.add_argument("--likelihood-tables", default=str(DEFAULT_LIKELIHOOD_TABLES))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument(
        "--exclude-l2",
        action="store_true",
        help="Run in partial-info regime: Bayesian controller has no L2 access.",
    )
    args = parser.parse_args()

    # Load data
    episodes = load_episodes(Path(args.calibration_data))
    log.info("Loaded %d instances with calibration data", len(episodes))

    # Compute base rate (prior)
    all_patches = [p for patches in episodes.values() for p in patches]
    n_correct = sum(1 for p in all_patches if p["ground_truth"] == 1)
    base_rate = n_correct / len(all_patches) if all_patches else 0.2
    log.info("Base rate (prior): %.3f (%d/%d correct)", base_rate, n_correct, len(all_patches))

    # If base rate is 0, use a small prior to avoid degenerate behavior
    prior = max(base_rate, 0.05)

    costs = CostModel()

    # Load Bayesian controller
    if args.exclude_l2:
        log.info("PARTIAL-INFO MODE: Bayesian controller without L2 fast test")
        controller = _rebuild_controller_without_levels(
            args.likelihood_tables,
            exclude_levels=["L2_fast_test"],
            costs=costs,
            horizon=args.horizon,
        )
    else:
        controller = BayesianController.from_likelihood_tables(
            args.likelihood_tables,
            costs=costs,
            horizon=args.horizon,
        )
    controller.print_policy()

    # Run all policies
    bayesian_results = []
    fixed_results = []
    threshold_l1_results = []
    threshold_l2_results = []
    threshold_l3_results = []
    threshold_l4_results = []

    has_l3 = any(
        "L3_llm_review" in p["critic_results"]
        for patches in episodes.values()
        for p in patches
    )
    has_l4 = any(
        "L4_mypy" in p["critic_results"]
        for patches in episodes.values()
        for p in patches
    )

    for instance_id, patches in episodes.items():
        bayesian_results.append(
            run_bayesian_policy(controller, patches, costs, prior)
        )
        fixed_results.append(
            run_fixed_pipeline(patches, costs)
        )
        threshold_l1_results.append(
            run_threshold_policy(patches, costs, "L1_lint")
        )
        threshold_l2_results.append(
            run_threshold_policy(patches, costs, "L2_fast_test")
        )
        if has_l3:
            threshold_l3_results.append(
                run_threshold_policy(patches, costs, "L3_llm_review")
            )
        if has_l4:
            threshold_l4_results.append(
                run_threshold_policy(patches, costs, "L4_mypy")
            )

    # Print comparison
    print("\n" + "#" * 70)
    print("ORCHESTRATION POLICY COMPARISON")
    if args.exclude_l2:
        print("(PARTIAL-INFO REGIME: Bayesian controller uses L0/L1/L3/L4 only)")
    print("#" * 70)

    all_summaries = []
    bayesian_name = "Bayesian (no L2)" if args.exclude_l2 else "Bayesian Controller"
    all_summaries.append(print_results(bayesian_name, bayesian_results, costs))
    all_summaries.append(print_results("Fixed Pipeline (lint+verify)", fixed_results, costs))
    all_summaries.append(print_results("Threshold (L1 lint)", threshold_l1_results, costs))
    all_summaries.append(print_results("Threshold (L2 fast test)", threshold_l2_results, costs))
    if has_l3:
        all_summaries.append(print_results("Threshold (L3 LLM review)", threshold_l3_results, costs))
    if has_l4:
        all_summaries.append(print_results("Threshold (L4 mypy)", threshold_l4_results, costs))

    # Summary comparison table
    print("\n" + "=" * 80)
    print(f"{'Policy':<30} {'Pass@1':>8} {'AvgCost':>8} {'AvgUtil':>10} {'Verify/ep':>10}")
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['policy']:<30} {100*s['pass_rate']:>7.1f}% {s['avg_cost']:>8.1f} "
              f"{s['avg_utility']:>10.1f} {s['verify_per_ep']:>10.1f}")
    print("=" * 80)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "summaries": all_summaries,
                "prior": prior,
                "n_instances": len(episodes),
                "n_patches": len(all_patches),
                "base_rate": base_rate,
            }, f, indent=2)
        log.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
