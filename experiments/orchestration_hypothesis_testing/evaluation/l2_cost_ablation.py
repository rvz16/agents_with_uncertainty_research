#!/usr/bin/env python3
"""L2 cost ablation: when does L2 become too expensive to be worth it?

On sympy Lite, L2 is near-oracle (gap=0.976) so Threshold(L2) wins at
c_crit_l2=5.0. But in real repos, running test files takes 30-60s, not
1s. This ablation sweeps c_crit_l2 and finds the cost at which
Bayesian starts to beat Threshold(L2).

The Bayesian controller automatically adapts: as L2 cost rises, the
controller learns to skip L2 and use cheaper critics (L0, L1, L3).
Threshold(L2) has no such adaptation — it always runs L2.

Usage:
    python l2_cost_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import BayesianController, CostModel
from evaluation.run_simulation import (
    load_episodes,
    run_bayesian_policy,
    run_fixed_pipeline,
    run_threshold_policy,
)

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "raw_results_v2.jsonl"
)
DEFAULT_TABLES = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "likelihood_tables.json"
)


def run_with_cost(
    episodes: dict,
    tables_path: str,
    costs: CostModel,
    prior: float,
) -> dict[str, dict]:
    controller = BayesianController.from_likelihood_tables(
        tables_path, costs=costs, horizon=10
    )

    b_res, f_res, t1, t2, t3 = [], [], [], [], []
    for patches in episodes.values():
        b_res.append(run_bayesian_policy(controller, patches, costs, prior))
        f_res.append(run_fixed_pipeline(patches, costs))
        t1.append(run_threshold_policy(patches, costs, "L1_lint"))
        t2.append(run_threshold_policy(patches, costs, "L2_fast_test"))
        t3.append(run_threshold_policy(patches, costs, "L3_llm_review"))

    def _summary(name: str, results: list) -> dict:
        n = len(results)
        resolved = sum(1 for r in results if r.resolved)
        utils = [(costs.reward if r.resolved else 0) - r.total_cost for r in results]
        avg_cost = sum(r.total_cost for r in results) / n
        return {
            "policy": name,
            "pass_rate": resolved / n,
            "avg_cost": avg_cost,
            "avg_utility": float(np.mean(utils)),
            "verify_per_ep": sum(r.n_verify_calls for r in results) / n,
            "critic_per_ep": sum(r.n_critic_calls for r in results) / n,
        }

    return {
        "Bayesian": _summary("Bayesian", b_res),
        "Fixed": _summary("Fixed", f_res),
        "Threshold(L1)": _summary("Threshold(L1)", t1),
        "Threshold(L2)": _summary("Threshold(L2)", t2),
        "Threshold(L3)": _summary("Threshold(L3)", t3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--tables", default=str(DEFAULT_TABLES))
    parser.add_argument(
        "--c-l2-values",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 5.0, 8.0, 12.0, 15.0, 18.0, 20.0, 25.0],
    )
    parser.add_argument("--out", default="l2_cost_ablation.json")
    args = parser.parse_args()

    episodes = load_episodes(Path(args.data))
    all_patches = [p for ps in episodes.values() for p in ps]
    prior = max(sum(1 for p in all_patches if p["ground_truth"] == 1) / len(all_patches), 0.05)
    print(f"Loaded {len(episodes)} instances, prior={prior:.3f}")

    results_by_cost: dict[float, dict] = {}

    for c_l2 in args.c_l2_values:
        costs = CostModel(c_crit_l2=c_l2)
        result = run_with_cost(episodes, args.tables, costs, prior)
        results_by_cost[c_l2] = result

        print(f"\nc_crit_l2 = {c_l2:.1f}")
        print(f"{'Policy':<20} {'Pass@1':>8} {'Cost':>8} {'Utility':>10} {'Verify/ep':>10}")
        print("-" * 65)
        for p in ["Bayesian", "Threshold(L2)", "Threshold(L1)", "Threshold(L3)", "Fixed"]:
            r = result[p]
            print(f"{p:<20} {100*r['pass_rate']:>7.1f}% {r['avg_cost']:>8.1f} "
                  f"{r['avg_utility']:>10.1f} {r['verify_per_ep']:>10.1f}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump({
            "c_l2_values": args.c_l2_values,
            "prior": prior,
            "results": {str(k): v for k, v in results_by_cost.items()},
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Find crossover
    print("\n" + "=" * 70)
    print("CROSSOVER: when does Bayesian beat Threshold(L2)?")
    print("=" * 70)
    for c_l2 in args.c_l2_values:
        bay = results_by_cost[c_l2]["Bayesian"]["avg_utility"]
        l2_th = results_by_cost[c_l2]["Threshold(L2)"]["avg_utility"]
        winner = "Bayesian" if bay > l2_th else "L2"
        print(f"c_l2={c_l2:>5.1f}  Bayesian={bay:>6.1f}  Threshold(L2)={l2_th:>6.1f}  winner={winner}")


if __name__ == "__main__":
    main()
