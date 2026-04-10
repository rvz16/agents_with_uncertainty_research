#!/usr/bin/env python3
"""Sweep c_crit_l2 on LCB data to find the best policy comparison."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import BayesianController, CostModel
from evaluation.run_simulation import (
    load_episodes, run_bayesian_policy, run_fixed_pipeline, run_threshold_policy,
)

DATA = Path(__file__).resolve().parents[1] / "calibration" / "data" / "lcb_results.jsonl"
TABLES = Path(__file__).resolve().parents[1] / "calibration" / "data" / "lcb_likelihood_tables.json"

episodes = load_episodes(DATA)
all_patches = [p for ps in episodes.values() for p in ps]
prior = max(sum(1 for p in all_patches if p["ground_truth"] == 1) / len(all_patches), 0.05)
print(f"n={len(all_patches)} instances={len(episodes)} prior={prior:.3f}")

# Sweep cost ratios
print(f"\n{'c_l2':>6} {'c_ver':>6} | {'Bay util':>10} {'Thr(L2) util':>14} {'Winner':>10}")
print("-" * 60)

scenarios = [
    (5.0, 20.0),   # Our current default
    (3.0, 20.0),   # Lower L2 cost
    (5.0, 25.0),   # Higher verifier cost
    (5.0, 30.0),   # Much higher verifier cost (more tests)
    (2.0, 30.0),   # Fast public, slow private
    (3.0, 30.0),   # Realistic LCB
    (4.0, 40.0),   # Very fast public, very slow private
]

for c_l2, c_ver in scenarios:
    costs = CostModel(c_crit_l2=c_l2, c_ver=c_ver)
    ctrl = BayesianController.from_likelihood_tables(TABLES, costs=costs, horizon=10)

    b_res, t1, t2, f_res = [], [], [], []
    for patches in episodes.values():
        b_res.append(run_bayesian_policy(ctrl, patches, costs, prior))
        t1.append(run_threshold_policy(patches, costs, "L1_lint"))
        t2.append(run_threshold_policy(patches, costs, "L2_fast_test"))
        f_res.append(run_fixed_pipeline(patches, costs))

    def _u(results):
        utils = [(costs.reward if r.resolved else 0) - r.total_cost for r in results]
        return np.mean(utils), np.std(utils) / np.sqrt(len(utils))

    b_u, b_se = _u(b_res)
    t2_u, t2_se = _u(t2)
    winner = "Bayesian" if b_u > t2_u else "Threshold(L2)"
    print(f"{c_l2:>6.1f} {c_ver:>6.1f} | {b_u:>+9.1f}±{b_se:.1f} {t2_u:>+12.1f}±{t2_se:.1f}  {winner}")
