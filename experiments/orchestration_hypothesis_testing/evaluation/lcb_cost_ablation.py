#!/usr/bin/env python3
"""Sweep c_crit_l2 on LCB data to find the best policy comparison.

Uses the fixed simulation semantics (verify is terminal for all policies) and
the iid transition kernel (fresh patch per generate, correct for our
calibration data which is iid samples). Paired differences reported alongside
marginal means to make the small effect sizes statistically interpretable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import BayesianController, CostModel
from evaluation.run_simulation import (
    load_episodes, run_bayesian_policy, run_fixed_pipeline, run_threshold_policy,
)

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "lcb_results_v2.jsonl"
)
DEFAULT_TABLES = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "lcb_likelihood_tables_v2.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--tables", default=str(DEFAULT_TABLES))
    parser.add_argument(
        "--iid-kernel",
        action="store_true",
        default=True,
        help="Use the iid-sampling transition kernel (default: True).",
    )
    parser.add_argument(
        "--no-iid-kernel",
        dest="iid_kernel",
        action="store_false",
        help="Use the calibrated within-problem kernel instead.",
    )
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    episodes = load_episodes(Path(args.data))
    all_patches = [p for ps in episodes.values() for p in ps]
    prior = max(
        sum(1 for p in all_patches if p["ground_truth"] == 1) / len(all_patches),
        0.05,
    )
    print(
        f"n={len(all_patches)} instances={len(episodes)} "
        f"prior={prior:.3f} iid_kernel={args.iid_kernel}"
    )

    header = (
        f"\n{'c_l2':>6} {'c_ver':>6} | "
        f"{'Bay util':>13} {'Thr(L2) util':>15} "
        f"{'Bay - Thr':>14} {'Winner':>14}"
    )
    print(header)
    print("-" * len(header))

    scenarios = [
        (5.0, 20.0),
        (3.0, 20.0),
        (5.0, 25.0),
        (5.0, 30.0),
        (2.0, 30.0),
        (3.0, 30.0),
        (4.0, 40.0),
    ]

    for c_l2, c_ver in scenarios:
        costs = CostModel(c_crit_l2=c_l2, c_ver=c_ver)
        ctrl = BayesianController.from_likelihood_tables(
            args.tables,
            costs=costs,
            horizon=args.horizon,
            iid_kernel=args.iid_kernel,
        )

        b_res, t2 = [], []
        for patches in episodes.values():
            b_res.append(run_bayesian_policy(ctrl, patches, costs, prior))
            t2.append(run_threshold_policy(patches, costs, "L2_fast_test"))

        def _u(results: list) -> tuple[np.ndarray, float, float]:
            utils = np.array(
                [(costs.reward if r.resolved else 0) - r.total_cost for r in results]
            )
            return utils, float(utils.mean()), float(utils.std() / np.sqrt(len(utils)))

        b_utils, b_u, b_se = _u(b_res)
        t_utils, t_u, t_se = _u(t2)
        diff = b_utils - t_utils
        diff_mean = float(diff.mean())
        diff_se = (
            float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        )
        t_stat = diff_mean / diff_se if diff_se > 0 else 0.0
        if abs(t_stat) < 2.0:
            winner = "tie"
        elif diff_mean > 0:
            winner = "Bayesian"
        else:
            winner = "Threshold(L2)"
        print(
            f"{c_l2:>6.1f} {c_ver:>6.1f} | "
            f"{b_u:>+8.1f}±{b_se:>3.1f} "
            f"{t_u:>+10.1f}±{t_se:>3.1f} "
            f"{diff_mean:>+9.2f}±{diff_se:>3.2f} "
            f"{winner:>14}"
        )


if __name__ == "__main__":
    main()
