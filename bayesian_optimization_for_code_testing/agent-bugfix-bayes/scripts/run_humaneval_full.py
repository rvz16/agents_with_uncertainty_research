#!/usr/bin/env python
"""End-to-end agent run on the full HumanEvalFix held-out split (124 tasks).

Train split: 40 tasks (used to fit theta_hat — same split as the calibration
test). Held-out: remaining 124 tasks for the agent comparison.

Resilience: saves results after every task. If interrupted (rate limit,
crash, manual stop), re-running this script will skip already-completed
(task, variant) pairs.

Usage:
    python scripts/run_humaneval_full.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.calibration import calibrate_likelihoods
from abbo.realworld.agents.humaneval_agent_runner import (
    aggregate, format_summary,
    run_simple, run_greedy, run_dp,
    HE_CRITIC_LIKELIHOODS, AgentRunResult,
)
from abbo.realworld.agents.humaneval_fix import (
    list_task_ids, collect_calibration_samples_from_pairs,
)
from abbo.realworld.agents.llm_provider import LLMConfig
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.bayes_agent import DPPlanner


# ---- Knobs ----
SPLIT_SEED = 42
N_TRAIN_FOR_THETA = 40
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
RESULTS_PATH = ROOT / "sim_results" / "humaneval_full_endtoend.json"
LLM_MODEL = "openai/gpt-oss-20b:free"

VARIANTS = ("simple", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted")


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {"results": {}, "fitted_theta": None}
    with open(path) as f:
        return json.load(f)


def save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def serialize_result(r: AgentRunResult) -> dict:
    return {
        "task_id": r.task_id, "variant": r.variant, "fixed": r.fixed,
        "total_cost": r.total_cost, "wall_clock": r.wall_clock,
        "n_llm_calls": r.n_llm_calls, "n_critic_runs": r.n_critic_runs,
        "n_full_tests": r.n_full_tests,
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "final_action": r.final_action,
        "actions": r.actions,
    }


def main() -> None:
    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN_FOR_THETA]
    test_ids = all_ids[N_TRAIN_FOR_THETA:]
    print(f"Train: {len(train_ids)}  Held-out: {len(test_ids)}")

    state = load_existing(RESULTS_PATH)

    # Fit theta_hat (cheap — re-run if missing)
    if not state.get("fitted_theta"):
        print("Fitting fitted_theta on 40 train tasks...")
        train_samples = collect_calibration_samples_from_pairs(train_ids, verbose=False)
        fitted_lk, _ = calibrate_likelihoods(train_samples)
        state["fitted_theta"] = fitted_lk
        save_progress(RESULTS_PATH, state)
        print(f"  fitted theta: {json.dumps(fitted_lk, indent=2)}")
    fitted_theta = state["fitted_theta"]

    # Pre-solve DP planners
    costs = AgentCostConfig()
    dp_hand = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                        critic_likelihoods=HE_CRITIC_LIKELIHOODS)
    dp_hand.solve()
    dp_fitted = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                          critic_likelihoods=fitted_theta)
    dp_fitted.solve()

    llm_cfg = LLMConfig(
        provider="openrouter",
        model=LLM_MODEL,
        base_url="https://openrouter.ai/api",
        temperature=0.1,
        max_tokens=2048,
        timeout=120,
    )

    results = state.setdefault("results", {})
    total = len(test_ids) * len(VARIANTS)
    done = sum(1 for tid in test_ids for v in VARIANTS if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} (task, variant) pairs already done.")

    started = time.time()
    for i, tid in enumerate(test_ids):
        elapsed = time.time() - started
        rate = (i + 1) / max(0.001, elapsed)
        eta_min = (len(test_ids) - i - 1) / max(0.0001, rate) / 60
        print(f"\n[{i+1}/{len(test_ids)}] task={tid}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min")
        for v in VARIANTS:
            key = f"{tid}|{v}"
            if results.get(key):
                continue
            try:
                if v == "simple":
                    r = run_simple(tid, llm_cfg, costs, n_retries=MAX_GENERATORS)
                elif v == "greedy_hand":
                    r = run_greedy(tid, HE_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, MAX_GENERATORS, PRIOR)
                elif v == "greedy_fitted":
                    r = run_greedy(tid, fitted_theta, "fitted",
                                   llm_cfg, costs, MAX_GENERATORS, PRIOR)
                elif v == "dp_hand":
                    r = run_dp(tid, HE_CRITIC_LIKELIHOODS, "hand",
                               llm_cfg, costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                               PRIOR, planner=dp_hand)
                elif v == "dp_fitted":
                    r = run_dp(tid, fitted_theta, "fitted",
                               llm_cfg, costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                               PRIOR, planner=dp_fitted)
                else:
                    continue
            except Exception as e:
                print(f"  [{v}] EXCEPTION: {e}")
                continue
            results[key] = serialize_result(r)
            tag = "OK" if r.fixed else "no"
            print(f"  {v:<16} fix={tag}  cost={r.total_cost:5.1f}  "
                  f"llm={r.n_llm_calls}  toks={r.completion_tokens}  "
                  f"wc={r.wall_clock:.1f}s  final={r.final_action}")
            save_progress(RESULTS_PATH, state)

    # Final aggregate
    print("\n=== Aggregate over completed tasks ===")
    by_variant: dict[str, list[AgentRunResult]] = {v: [] for v in VARIANTS}
    for key, rec in results.items():
        v = rec["variant"]
        if v in by_variant:
            by_variant[v].append(AgentRunResult(**{
                k: rec.get(k) for k in [
                    "task_id", "variant", "fixed", "total_cost", "wall_clock",
                    "n_llm_calls", "n_critic_runs", "n_full_tests",
                    "prompt_tokens", "completion_tokens", "final_action", "actions",
                ]
            }))
    agg = aggregate(by_variant)
    print(format_summary(agg))
    state["aggregate"] = agg
    state["llm_model"] = LLM_MODEL
    state["n_test_tasks"] = len(test_ids)
    state["n_train_tasks"] = len(train_ids)
    save_progress(RESULTS_PATH, state)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
