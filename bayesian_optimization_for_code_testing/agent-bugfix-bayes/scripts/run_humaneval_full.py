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
import re
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
    ARM_PROMPTS, ARM_SEQUENCE, _get_test_output, extract_code,
)
from abbo.realworld.agents.humaneval_fix import (
    get_buggy_source, get_full_test_script,
    list_task_ids, collect_calibration_samples_from_pairs,
    run_critic, run_full_test,
)
from abbo.realworld.agents.llm_provider import build_llm_config_from_env, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.bayes_agent import DPPlanner
from abbo.realworld.agents.kernel_helpers import (
    DEFAULT_KERNEL, resolve_kernel, OnlineKernelCalibration,
)
from abbo.realworld.telemetry import TelemetryLogger, write_action
import tempfile, time


# ---- Knobs ----
SPLIT_SEED = 42
TRAIN_FRAC = 0.75   # default 75/25 train/test split
N_TRAIN_FOR_THETA = None   # if None, computed as int(round(TRAIN_FRAC * n_total)); override via --n-train
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b:free"

DEFAULT_VARIANTS = (
    "simple", "best_of_3",
    "threshold_L0", "threshold_L2", "threshold_L3", "fixed_pipeline",
    "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted",
    "self_refine", "reflexion",
)

HE_CRITICS_ORDERED = ["critic_syntax", "critic_lint", "critic_early", "critic_mid"]
THRESHOLD_CRITICS = {
    "threshold_L0": ["critic_syntax"],
    "threshold_L2": ["critic_syntax", "critic_early"],
    "threshold_L3": HE_CRITICS_ORDERED,
}


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
        help="Limit held-out tasks to first N (for cost measurement pilots).",
    )
    p.add_argument(
        "--kernel-mode",
        default="measured",
        choices=["measured", "online", "hardcoded"],
        help="Transition kernel source for greedy/dp variants.",
    )
    p.add_argument(
        "--kernel-dir",
        type=Path,
        default=None,
        help="Directory with transition_kernel.json (e.g. data/humanevalfix_iter/<gen>/). "
             "If not specified, falls back to sim_results/transition_kernels.json.",
    )
    p.add_argument(
        "--n-train",
        type=int,
        default=None,
        help=f"Number of train instances. Default: int(round({TRAIN_FRAC} * n_total)) "
             "= 75/25 train/test split. Pass --n-train 40 for legacy 40/124 split.",
    )
    return p.parse_args()


REFINE_SUFFIX = "\n\nThe previous attempt failed these checks:\n{feedback}\nProvide a revised COMPLETE corrected program. Code only."
REFLEXION_SUFFIX = "\n\nThe previous attempt failed the test suite. Test output:\n{test_feedback}\nReflect and provide a corrected complete program. Code only."


def _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id, belief=None):
    t0 = time.perf_counter()
    resp = call_llm_or_raise(prompt, llm_cfg)
    write_action(logger, run_id=run_id or task_id, dataset="humaneval",
                 instance_id=task_id, action_type="generate",
                 runtime_s=time.perf_counter() - t0,
                 model_name=llm_cfg.model, belief_before=belief)
    res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
    res.prompt_tokens += resp.prompt_tokens
    res.completion_tokens += resp.completion_tokens
    return resp


def _verify(workdir, task_id, costs, res, logger, run_id, belief=None):
    t0 = time.perf_counter()
    ok, _ = run_full_test(workdir, task_id)
    write_action(logger, run_id=run_id or task_id, dataset="humaneval",
                 instance_id=task_id, action_type="verify",
                 runtime_s=time.perf_counter() - t0, passed=ok,
                 belief_before=belief)
    res.n_full_tests += 1; res.total_cost += costs.c_full_test
    return ok


def _critic(workdir, cn, task_id, costs, res, logger, run_id, belief=None):
    t0 = time.perf_counter()
    passed, msg = run_critic(workdir, cn, task_id)
    write_action(logger, run_id=run_id or task_id, dataset="humaneval",
                 instance_id=task_id, action_type=cn,
                 runtime_s=time.perf_counter() - t0, passed=passed,
                 belief_before=belief)
    res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
    return passed, msg


def _build_prompt(task_id, current, workdir, arm_idx=0, suffix=""):
    test_out = _get_test_output(workdir, task_id)
    arm = ARM_SEQUENCE[arm_idx % len(ARM_SEQUENCE)]
    return ARM_PROMPTS[arm].format(
        source_code=current,
        test_output=test_out,
        test_code=get_full_test_script(task_id),
    ) + suffix


def run_best_of_n(task_id, llm_cfg, costs, n=3, logger=None, run_id=None):
    res = AgentRunResult(task_id=task_id, variant="best_of_3", fixed=False,
                         total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp); sol = workdir / "solution.py"
        for attempt in range(n):
            sol.write_text(buggy)
            prompt = _build_prompt(task_id, buggy, workdir, arm_idx=attempt)
            resp = _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id)
            current = extract_code(resp.text, buggy); sol.write_text(current)
            ok = _verify(workdir, task_id, costs, res, logger, run_id)
            res.actions.append({"step": attempt, "verify_pass": ok})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_threshold(task_id, variant_name, critics, llm_cfg, costs,
                  max_gen=3, logger=None, run_id=None):
    res = AgentRunResult(task_id=task_id, variant=variant_name, fixed=False,
                         total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp); sol = workdir / "solution.py"
        for attempt in range(max_gen):
            sol.write_text(buggy)
            prompt = _build_prompt(task_id, buggy, workdir, arm_idx=attempt)
            resp = _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id)
            current = extract_code(resp.text, buggy); sol.write_text(current)
            gate = True
            for cn in critics:
                passed, _ = _critic(workdir, cn, task_id, costs, res, logger, run_id)
                res.actions.append({"step": attempt, "action": f"critic:{cn}", "passed": passed})
                if not passed:
                    gate = False; break
            if gate:
                ok = _verify(workdir, task_id, costs, res, logger, run_id)
                res.actions.append({"step": attempt, "verify_pass": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_fixed_pipeline(task_id, llm_cfg, costs, max_gen=3, logger=None, run_id=None):
    res = run_threshold(task_id, "fixed_pipeline", HE_CRITICS_ORDERED,
                        llm_cfg, costs, max_gen, logger, run_id)
    res.variant = "fixed_pipeline"
    return res


def run_self_refine(task_id, llm_cfg, costs, max_rounds=2, logger=None, run_id=None):
    res = AgentRunResult(task_id=task_id, variant="self_refine", fixed=False,
                         total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp); sol = workdir / "solution.py"
        current = buggy; suffix = ""
        for rnd in range(max_rounds + 1):
            sol.write_text(buggy)
            prompt = _build_prompt(task_id, current, workdir, arm_idx=rnd, suffix=suffix)
            resp = _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id)
            current = extract_code(resp.text, current); sol.write_text(current)
            failures = []
            for cn in HE_CRITICS_ORDERED:
                passed, msg = _critic(workdir, cn, task_id, costs, res, logger, run_id)
                if not passed:
                    failures.append(f"- {cn}: {msg}")
            res.actions.append({"round": rnd, "n_failed": len(failures)})
            if not failures or rnd == max_rounds:
                ok = _verify(workdir, task_id, costs, res, logger, run_id)
                res.actions.append({"round": rnd, "verify_pass": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"
                break
            suffix = REFINE_SUFFIX.format(feedback="\n".join(failures))
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_reflexion(task_id, llm_cfg, costs, max_rounds=2, logger=None, run_id=None):
    res = AgentRunResult(task_id=task_id, variant="reflexion", fixed=False,
                         total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp); sol = workdir / "solution.py"
        current = buggy; suffix = ""
        for rnd in range(max_rounds + 1):
            sol.write_text(buggy)
            prompt = _build_prompt(task_id, current, workdir, arm_idx=rnd, suffix=suffix)
            resp = _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id)
            current = extract_code(resp.text, current); sol.write_text(current)
            ok = _verify(workdir, task_id, costs, res, logger, run_id)
            res.actions.append({"round": rnd, "verify_pass": ok})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            if rnd < max_rounds:
                test_fb = _get_test_output(workdir, task_id)
                suffix = REFLEXION_SUFFIX.format(test_feedback=test_fb[:600])
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def main() -> None:
    args = parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for v in variants:
        if v not in DEFAULT_VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}; allowed: {DEFAULT_VARIANTS}")
    results_path = args.results or (ROOT / "sim_results" / "humaneval_full_endtoend.json")
    llm_model = (args.model or "").strip() or DEFAULT_LLM_MODEL
    n_tasks = args.n_tasks

    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    if args.n_train is not None:
        n_train_active = args.n_train
    else:
        n_train_active = int(round(TRAIN_FRAC * len(all_ids)))
    if n_train_active < 1 or n_train_active >= len(all_ids):
        raise SystemExit(
            f"n_train {n_train_active} invalid (must be 1..{len(all_ids)-1}; "
            f"total instances = {len(all_ids)})"
        )
    train_ids = all_ids[:n_train_active]
    test_ids = all_ids[n_train_active:]
    if n_tasks is not None:
        test_ids = test_ids[:n_tasks]
    print(f"Train: {len(train_ids)}  Held-out: {len(test_ids)}")

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

    # Pre-solve DP planners
    costs = AgentCostConfig()

    # Resolve transition kernel
    kernel_dir = args.kernel_dir
    if kernel_dir is None:
        legacy_path = ROOT / "sim_results" / "transition_kernels.json"
        if legacy_path.exists() and args.kernel_mode != "hardcoded":
            jj = json.loads(legacy_path.read_text())
            hef = jj.get("humaneval_fix", {})
            if "p_fix_broken" in hef:
                measured = {"p_fix_broken": float(hef["p_fix_broken"]),
                            "p_break_correct": float(hef["p_break_correct"]
                                                     if hef.get("p_break_correct") is not None
                                                     else jj.get("prior_break_correct", 0.05))}
                kernel = measured
                kernel_src = "measured (sim_results/transition_kernels.json)"
                online_kernel = (OnlineKernelCalibration(init_kernel=measured)
                                  if args.kernel_mode == "online" else None)
            else:
                kernel = DEFAULT_KERNEL.copy()
                kernel_src = "default (no humaneval_fix kernel)"
                online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if args.kernel_mode == "online" else None)
        else:
            kernel = DEFAULT_KERNEL.copy()
            kernel_src = ("hardcoded (forced)" if args.kernel_mode == "hardcoded"
                          else "default (no kernel file)")
            online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                              if args.kernel_mode == "online" else None)
    else:
        kernel, kernel_src, online_kernel = resolve_kernel(kernel_dir, args.kernel_mode)
    print(f"Transition kernel: source={kernel_src}, "
          f"p_fix={kernel['p_fix_broken']:.3f}, p_break={kernel['p_break_correct']:.3f}, "
          f"mode={args.kernel_mode}")

    def make_dp(theta_dict, kern):
        p = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                      critic_likelihoods=theta_dict, transition_kernel=kern)
        p.solve()
        return p

    dp_hand = make_dp(HE_CRITIC_LIKELIHOODS, kernel)
    dp_fitted = make_dp(fitted_theta, kernel)

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

    _model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", llm_cfg.model)
    telemetry_path = ROOT / "logs" / f"action_telemetry_humaneval__{_model_slug}.jsonl"
    tlog = TelemetryLogger(telemetry_path)
    print(f"Telemetry → {telemetry_path}")

    started = time.time()
    try:
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
                run_id = f"{_model_slug}__{tid}__{v}"
                try:
                    if v == "simple":
                        r = run_simple(tid, llm_cfg, costs, n_retries=MAX_GENERATORS,
                                       logger=tlog, run_id=run_id)
                    elif v == "best_of_3":
                        r = run_best_of_n(tid, llm_cfg, costs, n=3,
                                          logger=tlog, run_id=run_id)
                    elif v in THRESHOLD_CRITICS:
                        r = run_threshold(tid, v, THRESHOLD_CRITICS[v],
                                          llm_cfg, costs, MAX_GENERATORS,
                                          logger=tlog, run_id=run_id)
                    elif v == "fixed_pipeline":
                        r = run_fixed_pipeline(tid, llm_cfg, costs,
                                               MAX_GENERATORS,
                                               logger=tlog, run_id=run_id)
                    elif v == "greedy_hand":
                        r = run_greedy(tid, HE_CRITIC_LIKELIHOODS, "hand",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR,
                                       logger=tlog, run_id=run_id,
                                       kernel=kernel, online_kernel=online_kernel)
                    elif v == "greedy_fitted":
                        r = run_greedy(tid, fitted_theta, "fitted",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR,
                                       logger=tlog, run_id=run_id,
                                       kernel=kernel, online_kernel=online_kernel)
                    elif v == "dp_hand":
                        r = run_dp(tid, HE_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                                   PRIOR, planner=dp_hand,
                                   logger=tlog, run_id=run_id,
                                   kernel=kernel, online_kernel=online_kernel,
                                   make_planner=(make_dp if online_kernel is not None else None))
                    elif v == "dp_fitted":
                        r = run_dp(tid, fitted_theta, "fitted",
                                   llm_cfg, costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                                   PRIOR, planner=dp_fitted,
                                   logger=tlog, run_id=run_id,
                                   kernel=kernel, online_kernel=online_kernel,
                                   make_planner=(make_dp if online_kernel is not None else None))
                    elif v == "self_refine":
                        r = run_self_refine(tid, llm_cfg, costs,
                                            logger=tlog, run_id=run_id)
                    elif v == "reflexion":
                        r = run_reflexion(tid, llm_cfg, costs,
                                          logger=tlog, run_id=run_id)
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
    finally:
        tlog.close()

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
