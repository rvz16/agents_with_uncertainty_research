#!/usr/bin/env python
"""End-to-end agent run on the full HumanEvalFix held-out split (124 tasks).

Train split: 40 tasks (used to fit theta_hat — same split as the calibration
test). Held-out: remaining 124 tasks for the agent comparison.

Resilience: saves results after every task. If interrupted (rate limit,
crash, manual stop), re-running this script will skip already-completed
(task, variant) pairs.

Usage:
    python scripts/run_humaneval_full.py
    python scripts/run_humaneval_full.py --model anthropic/claude-haiku-4.5 \\
        --results sim_results/humaneval_full__haiku45.json

Paper generator IDs (orchestration_hypothesis_testing) map to OpenRouter
``--model`` strings like ``openai/gpt-5-mini``, ``anthropic/claude-haiku-4.5``,
``qwen/qwen3-coder``, ``anthropic/claude-sonnet-4.5``. Local vLLM uses
``ABBO_LLM_PROVIDER=openai`` and ``ABBO_LLM_BASE_URL``.

Note: abbreviated policies in the paper (BoN, tL0–tL3, FP, SR, Rfx, BG, BDP)
are evaluated via calibration + replay under ``experiments/orchestration_*
`` on HumanEval+ / LCB. This script’s variants are bugfix agents: ``simple``,
``greedy_*`` (Bayesian one-step lookahead), ``dp_*`` (POMDP DP).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
from abbo.realworld.agents.llm_provider import build_llm_config_from_env
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.bayes_agent import DPPlanner


# ---- Knobs ----
SPLIT_SEED = 42
N_TRAIN_FOR_THETA = 40
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b:free"

DEFAULT_VARIANTS = ("simple", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted")


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEvalFix held-out agent comparison (resume-safe).")
    p.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model id (overrides ABBO_LLM_MODEL for this run).",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Output JSON path (default: sim_results/humaneval_full_endtoend.json).",
    )
    p.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated subset of: simple,greedy_hand,greedy_fitted,dp_hand,dp_fitted",
    )
    p.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Limit number of held-out tasks (default: all).",
    )
    p.add_argument(
        "--kernel-mode",
        choices=["measured", "online", "hardcoded", "thompson",
                  "conditional", "conditional_online"],
        default="measured",
        help="dp_fitted transition kernel mode. 'online' updates via "
             "Beta-Binomial after every verify (DP re-solves). "
             "'thompson' samples kernel from Beta posterior per instance, "
             "anchored by train raw_counts. 'conditional' uses a frozen "
             "ConditionalKernel fit from iter_records (per-critic-signature). "
             "'conditional_online' updates per-z buckets live.",
    )
    p.add_argument(
        "--conditional-records",
        type=Path,
        action="append",
        default=None,
        help="Path(s) to iter_records.jsonl used to fit ConditionalKernel.",
    )
    p.add_argument(
        "--kernel-seed-dir",
        type=Path,
        default=None,
        help="Dir containing transition_kernel.json used as seed for online mode "
             "or as the frozen kernel for measured mode.",
    )
    p.add_argument(
        "--epsilon-explore",
        type=float,
        default=0.0,
        help="With probability epsilon, override DP 'bail_out' with a forced "
             "generate action so the online estimator can observe (Y_t, Y_{t+1}) "
             "transitions. 0 = pure DP (current behavior).",
    )
    p.add_argument(
        "--explore-seed",
        type=int,
        default=12345,
        help="Seed for epsilon-greedy override RNG (deterministic exploration).",
    )
    p.add_argument(
        "--thompson-seed",
        type=int,
        default=12345,
        help="Seed for Thompson sampling RNG.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for v in variants:
        if v not in DEFAULT_VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}; allowed: {DEFAULT_VARIANTS}")
    results_path = args.results or (ROOT / "sim_results" / "humaneval_full_endtoend.json")
    llm_model = (args.model or "").strip() or DEFAULT_LLM_MODEL

    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    explore_rng = random.Random(args.explore_seed) if args.epsilon_explore > 0 else None
    if args.epsilon_explore > 0:
        print(f"ε-greedy bail override: epsilon={args.epsilon_explore} "
              f"seed={args.explore_seed}")
    thompson_rng = random.Random(args.thompson_seed) if args.kernel_mode == "thompson" else None
    if thompson_rng is not None:
        print(f"Thompson sampling enabled (seed={args.thompson_seed})")
    train_ids = all_ids[:N_TRAIN_FOR_THETA]
    test_ids = all_ids[N_TRAIN_FOR_THETA:]
    if args.n_tasks is not None:
        test_ids = test_ids[:args.n_tasks]
    print(f"Train: {len(train_ids)}  Held-out: {len(test_ids)}")
    print(f"Kernel mode: {args.kernel_mode}  Seed dir: {args.kernel_seed_dir}")

    state = load_existing(results_path)

    # Fit theta_hat (cheap — re-run if missing)
    if not state.get("fitted_theta"):
        print("Fitting fitted_theta on 40 train tasks...")
        train_samples = collect_calibration_samples_from_pairs(train_ids, verbose=False)
        fitted_lk, _ = calibrate_likelihoods(train_samples)
        state["fitted_theta"] = fitted_lk
        save_progress(results_path, state)
        print(f"  fitted theta: {json.dumps(fitted_lk, indent=2)}")
    fitted_theta = state["fitted_theta"]

    # Resolve transition kernel for dp_fitted via kernel-mode.
    from abbo.realworld.agents.kernel_helpers import resolve_kernel, OnlineKernelCalibration
    base_mode = args.kernel_mode
    if base_mode in {"conditional", "conditional_online"}:
        base_mode = "measured"
    if args.kernel_seed_dir is not None:
        active_kernel, kernel_src, online_kernel_obj = resolve_kernel(
            args.kernel_seed_dir, mode=base_mode)
    elif base_mode == "online":
        active_kernel = {"p_fix_broken": 0.5, "p_break_correct": 0.05}
        kernel_src = "hardcoded (no seed dir)"
        online_kernel_obj = OnlineKernelCalibration(init_kernel=active_kernel)
    else:
        active_kernel = None
        kernel_src = "default planner kernel"
        online_kernel_obj = None
    print(f"dp_fitted kernel source: {kernel_src}")
    if active_kernel is not None:
        print(f"  initial kernel: {active_kernel}")

    # Conditional kernel: load from iter_records.jsonl
    conditional_kernel_seed = None
    if args.kernel_mode in {"conditional", "conditional_online"}:
        rec_paths = args.conditional_records or (
            [args.kernel_seed_dir / "iter_records.jsonl"]
            if args.kernel_seed_dir else [])
        if not rec_paths:
            raise SystemExit("--kernel-mode conditional requires --conditional-records or --kernel-seed-dir")
        # Import from _common (synthesis side) using sys.path
        import sys as _sys
        _common_path = ROOT.parent.parent / "experiments" / "orchestration_hypothesis_testing"
        if not (_common_path / "_common").exists():
            raise SystemExit(f"_common dir not found at {_common_path}")
        _sys.path.insert(0, str(_common_path))
        from _common.kernel import (ConditionalKernel,
                                      conditional_pairs_from_trajectories,
                                      CRITIC_FIELDS_DEFAULT)
        from collections import defaultdict
        by_inst: dict[str, list[dict]] = defaultdict(list)
        n_loaded = 0
        for rp in rec_paths:
            if not Path(rp).exists():
                raise SystemExit(f"missing {rp}")
            for line in Path(rp).read_text().splitlines():
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = r.get("instance_id")
                if iid is None: continue
                by_inst[iid].append(r)
                n_loaded += 1
        for iid in by_inst:
            by_inst[iid].sort(key=lambda r: int(r.get("step", r.get("patch_id", 0))))
        # Use train_ids if they exist in iter_records, else all
        train_in_records = [i for i in train_ids if i in by_inst] or list(by_inst.keys())
        train_trajs = [by_inst[i] for i in train_in_records]
        triples = conditional_pairs_from_trajectories(
            train_trajs, critic_fields=CRITIC_FIELDS_DEFAULT)
        seed_marginal = active_kernel or {"p_fix_broken": 0.5, "p_break_correct": 0.05}
        conditional_kernel_seed = ConditionalKernel.from_triples(
            triples, critic_fields=CRITIC_FIELDS_DEFAULT,
            init_kernel=seed_marginal, min_obs=3,
        )
        print(f"ConditionalKernel: loaded {n_loaded} records, {len(train_trajs)} train trajectories, "
              f"{len(conditional_kernel_seed.counts)} buckets, {len(triples)} triples")

    # Pre-solve DP planners
    costs = AgentCostConfig()
    dp_hand = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                        critic_likelihoods=HE_CRITIC_LIKELIHOODS)
    dp_hand.solve()
    dp_fitted_kwargs = {
        "critic_likelihoods": fitted_theta,
    }
    if active_kernel is not None:
        dp_fitted_kwargs["transition_kernel"] = active_kernel
    dp_fitted = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                          **dp_fitted_kwargs)
    dp_fitted.solve()

    llm_cfg = build_llm_config_from_env(
        default_provider="openrouter",
        default_model=llm_model,
        default_base_url="https://openrouter.ai/api",
        default_temperature=0.1,
        default_max_tokens=2048,
        default_timeout=120,
    )
    if args.model:
        llm_cfg.model = args.model.strip()
    print(f"LLM provider={llm_cfg.provider} model={llm_cfg.model} base_url={llm_cfg.base_url}")

    results = state.setdefault("results", {})
    total = len(test_ids) * len(variants)
    done = sum(1 for tid in test_ids for v in variants if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} (task, variant) pairs already done.")

    started = time.time()
    for i, tid in enumerate(test_ids):
        elapsed = time.time() - started
        rate = (i + 1) / max(0.001, elapsed)
        eta_min = (len(test_ids) - i - 1) / max(0.0001, rate) / 60
        print(f"\n[{i+1}/{len(test_ids)}] task={tid}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min")
        for v in variants:
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
                    # Thompson: sample kernel for THIS instance, re-solve DP
                    inst_planner = dp_fitted
                    if thompson_rng is not None and online_kernel_obj is not None:
                        sampled = online_kernel_obj.sample(thompson_rng)
                        inst_planner = DPPlanner(
                            costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                            critic_likelihoods=fitted_theta,
                            transition_kernel=sampled,
                        )
                        inst_planner.solve()
                        print(f"  [thompson] p_fix={sampled['p_fix_broken']:.3f} "
                              f"p_break={sampled['p_break_correct']:.3f}")
                    # Conditional kernel per-instance
                    cond_kernel = None
                    if args.kernel_mode == "conditional":
                        import copy
                        cond_kernel = copy.deepcopy(conditional_kernel_seed)
                    elif args.kernel_mode == "conditional_online":
                        cond_kernel = conditional_kernel_seed
                    r = run_dp(tid, fitted_theta, "fitted",
                               llm_cfg, costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                               PRIOR, planner=inst_planner,
                               online_kernel=online_kernel_obj,
                               epsilon_explore=args.epsilon_explore,
                               explore_rng=explore_rng,
                               conditional_kernel=cond_kernel)
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
            save_progress(results_path, state)

    # Final aggregate
    print("\n=== Aggregate over completed tasks ===")
    by_variant: dict[str, list[AgentRunResult]] = {v: [] for v in variants}
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
    state["llm_model"] = llm_cfg.model
    state["n_test_tasks"] = len(test_ids)
    state["n_train_tasks"] = len(train_ids)
    save_progress(results_path, state)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
