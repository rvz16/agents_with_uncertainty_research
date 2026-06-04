#!/usr/bin/env python
"""End-to-end agent run on CodeContests held-out (82 tasks).

Train split: same 30 tasks as test_codecontests_calibration.py used to fit
fitted_theta. Held-out: remaining ~82 tasks for the agent comparison.

Resilience: saves results after every (task, variant) pair. Re-running this
script skips already-completed pairs, so it's safe to interrupt + resume.

Usage:
    python scripts/run_codecontests_full.py
    python scripts/run_codecontests_full.py --model qwen/qwen3-coder \\
        --results sim_results/codecontests_full__qwen3_coder.json

See run_humaneval_full.py module docstring for paper generator ↔ model IDs
and policy naming caveats (bugfix variants vs orchestration replay).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES,
    get_solution_pool, get_test_cases, get_metadata,
    list_task_ids, run_critic, run_full_test,
)
from abbo.realworld.agents.llm_provider import build_llm_config_from_env, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig

# ---- Config ----
SPLIT_SEED = 42
N_TRAIN = 30        # tasks used by calibration to fit fitted_theta
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b:free"
# Backward compat for scripts that import LLM_MODEL from this module
LLM_MODEL = DEFAULT_LLM_MODEL

DEFAULT_VARIANTS = ("simple", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted")

# Cached fitted theta from the n=146 CodeContests calibration run
FITTED_THETA = {
    "critic_early": {"p_pass_y1": 0.9687500000000001, "p_pass_y0": 0.375},
    "critic_lint":  {"p_pass_y1": 0.728125,           "p_pass_y0": 0.678125},
    "critic_mid":   {"p_pass_y1": 0.9687500000000001, "p_pass_y0": 0.125},
    "critic_syntax":{"p_pass_y1": 0.9803571428571428, "p_pass_y0": 0.94375},
}

# ---- Prompt + extraction ----
PROMPT_TEMPLATE = """You are a competitive programming assistant. The Python solution below has a bug that makes it fail at least one test case. Return the COMPLETE corrected Python program — only the code, no explanation, no markdown fences.

Buggy code:
```python
{source_code}
```

Sample test cases (input → expected output):
{test_examples}

Recent failed run output:
```
{test_output}
```

Return only the corrected Python program."""

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)


def extract_code(llm_text, fallback):
    if not isinstance(llm_text, str):
        return fallback
    m = CODE_FENCE_RE.search(llm_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    s = llm_text.strip()
    return s if s else fallback


def format_test_examples(task_id, n=2):
    tests = get_test_cases(task_id, max_tests=n)
    parts = []
    for i, (inp, exp) in enumerate(tests, 1):
        parts.append(f"Test {i}:\n  input:\n{inp[:300]}\n  expected output:\n{exp[:200]}")
    return "\n\n".join(parts)


def get_test_output(workdir, task_id):
    tests = get_test_cases(task_id, max_tests=1)
    if not tests:
        return "(no test cases)"
    src = workdir / "solution.py"
    try:
        r = subprocess.run(
            [sys.executable, str(src)], input=tests[0][0],
            text=True, capture_output=True, timeout=4,
        )
        return f"stdout: {r.stdout[:300]}\nstderr: {r.stderr[:200]}\nexpected: {tests[0][1][:200]}"
    except Exception as e:
        return f"runtime error: {e}"


# ---- Result type ----
@dataclass
class Result:
    task_id: str
    variant: str
    cf_rating: int | None = None
    fixed: bool = False
    total_cost: float = 0.0
    wall_clock: float = 0.0
    n_llm_calls: int = 0
    n_critic_runs: int = 0
    n_full_tests: int = 0
    completion_tokens: int = 0
    final_action: str = ""
    actions: list = field(default_factory=list)


# ---- Variants ----
def run_simple(task_id, llm_cfg, costs, n_retries=3):
    res = Result(task_id=task_id, variant="simple",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        for attempt in range(n_retries):
            test_out = get_test_output(wd, task_id)
            prompt = PROMPT_TEMPLATE.format(
                source_code=current,
                test_examples=format_test_examples(task_id),
                test_output=test_out,
            )
            r = call_llm_or_raise(prompt, llm_cfg)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            current = extract_code(r.text, current)
            sol.write_text(current)
            ok, _ = run_full_test(wd, task_id)
            res.n_full_tests += 1; res.total_cost += costs.c_full_test
            res.actions.append({"step": attempt, "verify_pass": ok})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
        if not res.fixed:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def _q_one_step_critic(b, c, theta, costs):
    lk = theta[c]
    p = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    bp = bayes_update(b, c, True, likelihoods=theta)
    bf = bayes_update(b, c, False, likelihoods=theta)
    return -costs.c_critic_test \
        + p     * max(0.0, -costs.c_full_test + bp * costs.reward) \
        + (1-p) * max(0.0, -costs.c_full_test + bf * costs.reward)


def run_greedy(task_id, theta, label, llm_cfg, costs, max_gen=3, prior=0.5):
    res = Result(task_id=task_id, variant=f"greedy_{label}",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        belief = prior; gen_left = max_gen
        crit_used: set[str] = set(); step = 0
        while step < 12:
            Q_bail = 0.0
            Q_verify = -costs.c_full_test + belief * costs.reward
            Q_critics = {c: _q_one_step_critic(belief, c, theta, costs)
                         for c in theta if c not in crit_used}
            best_c, best_q = (max(Q_critics.items(), key=lambda x: x[1])
                              if Q_critics else (None, -math.inf))
            Q_gen = -math.inf
            if gen_left > 0:
                b_after = belief * 0.95 + (1-belief) * 0.50
                Q_gen = -costs.c_llm_call - costs.c_full_test + b_after * costs.reward
            choices = [("bail", Q_bail), ("verify", Q_verify)]
            if best_c: choices.append((f"critic:{best_c}", best_q))
            if gen_left > 0: choices.append(("generate", Q_gen))
            action, _q = max(choices, key=lambda x: x[1])

            if action == "bail":
                res.final_action = "bail"; break
            if action == "verify":
                ok, _ = run_full_test(wd, task_id)
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(wd, cn, task_id)
                res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used.add(cn)
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            else:  # generate
                test_out = get_test_output(wd, task_id)
                prompt = PROMPT_TEMPLATE.format(
                    source_code=current,
                    test_examples=format_test_examples(task_id),
                    test_output=test_out,
                )
                r = call_llm_or_raise(prompt, llm_cfg)
                res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left -= 1
                belief = belief * 0.95 + (1-belief) * 0.50
                crit_used = set()
                res.actions.append({"step": step, "action": "generate", "b": belief})
            step += 1
        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_dp(task_id, theta, label, llm_cfg, costs, planner,
           max_gen=3, max_ver=2, prior=0.5,
           kernel=None, online_kernel=None, planner_factory=None,
           conditional_kernel=None,
           critic_fields=("L0_syntax", "L2_public_tests", "L3_llm_review"),
           verify_on_bail: bool = False,
           refine_on_bail: bool = False,
           cascading_refine: bool = False,
           epsilon_explore: float = 0.0,
           explore_rng=None,
           gap_gated_refine: float = -1.0):
    res = Result(task_id=task_id, variant=f"dp_{label}",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    prev_Y = None  # for online kernel updates
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        belief = prior; gen_left = max_gen; ver_left = max_ver
        crit_used: frozenset[str] = frozenset(); step = 0
        z_obs: dict[str, int] = {}
        z_at_prev_gen: tuple | None = None
        # True when current code has never been verified since last generate
        # (or since start). Used by verify-on-bail to skip redundant verify.
        code_unverified = True
        while step < 16:
            action, _q = planner.choose_action(
                belief, gen_left, crit_used, ver_left,
            )
            if action == "bail_out":
                # ε-decay-on-bail: with probability ε override to generate.
                if (epsilon_explore > 0 and gen_left > 0 and ver_left > 0
                        and explore_rng is not None
                        and explore_rng.random() < epsilon_explore):
                    action = "generate:exploration"
                    res.actions.append({"step": step,
                                        "action": "epsilon_explore_override",
                                        "epsilon": epsilon_explore})
            if action == "bail_out":
                # cascading-refine: loop forced (gen+verify) until budget exhausted
                if (cascading_refine and gen_left > 0 and ver_left > 0
                        and prev_Y is not None):
                    iter_count = 0
                    while gen_left > 0 and ver_left > 0:
                        iter_count += 1
                        test_out = get_test_output(wd, task_id)
                        prompt = PROMPT_TEMPLATE.format(
                            source_code=current,
                            test_examples=format_test_examples(task_id),
                            test_output=test_out,
                        )
                        rr = call_llm_or_raise(prompt, llm_cfg)
                        res.n_llm_calls += 1
                        res.total_cost += costs.c_llm_call
                        res.completion_tokens += rr.completion_tokens
                        current = extract_code(rr.text, current)
                        sol.write_text(current)
                        gen_left -= 1
                        if conditional_kernel is not None:
                            z_at_prev_gen = tuple(int(z_obs.get(f, 0))
                                                    for f in critic_fields)
                        z_obs = {}
                        res.actions.append({"step": step + iter_count*2 - 1,
                                            "action": "generate_cascade",
                                            "iter": iter_count})
                        ok, _ = run_full_test(wd, task_id)
                        y_now = 1 if ok else 0
                        if online_kernel is not None:
                            online_kernel.update(prev_Y, y_now)
                        if (conditional_kernel is not None
                                and z_at_prev_gen is not None):
                            conditional_kernel.update(prev_Y, z_at_prev_gen, y_now)
                        res.n_full_tests += 1
                        res.total_cost += costs.c_full_test
                        ver_left -= 1
                        res.actions.append({"step": step + iter_count*2,
                                            "action": "verify_cascade",
                                            "iter": iter_count, "ok": ok})
                        prev_Y = y_now
                        if ok:
                            res.fixed = True
                            res.final_action = f"cascade_pass_iter{iter_count}"
                            break
                    if res.fixed:
                        break
                    res.final_action = f"cascade_exhausted_iter{iter_count}"
                    break

                # refine-on-bail: force one generate + verify before bail to
                # harvest the (Y_t, Y_{t+1}) transition pair the online kernel
                # needs. This directly addresses the "near-zero online updates"
                # issue: every bail trajectory now contributes one observation.
                if (refine_on_bail and gen_left > 0 and ver_left > 0
                        and prev_Y is not None):
                    # Gap-gated refine: skip forced refine when Q(bail) is
                    # decisively better than Q(best non-bail action).
                    # Q(bail) = 0 by construction; gap = 0 − Q_second_best.
                    # Large gap → bail clearly correct → don't waste cost.
                    # Small gap → planner uncertain → forced retry may catch.
                    if gap_gated_refine >= 0:
                        qs = planner.q_values_at(belief, gen_left, crit_used,
                                                  ver_left)
                        non_bail_qs = [v for kk, v in qs.items()
                                        if kk != "bail_out"]
                        if non_bail_qs:
                            q_second = max(non_bail_qs)
                            gap_to_bail = -q_second  # ≥ 0 because bail won
                            if gap_to_bail >= gap_gated_refine:
                                # bail decisively correct, skip refine
                                res.actions.append({
                                    "step": step,
                                    "action": "bail_gap_skip",
                                    "gap": gap_to_bail,
                                })
                                res.final_action = "bail_gap_skip"
                                break
                    test_out = get_test_output(wd, task_id)
                    prompt = PROMPT_TEMPLATE.format(
                        source_code=current,
                        test_examples=format_test_examples(task_id),
                        test_output=test_out,
                    )
                    rr = call_llm_or_raise(prompt, llm_cfg)
                    res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                    res.completion_tokens += rr.completion_tokens
                    current = extract_code(rr.text, current)
                    sol.write_text(current)
                    gen_left -= 1
                    # Track z_at_prev_gen for conditional kernel
                    if conditional_kernel is not None:
                        z_at_prev_gen = tuple(int(z_obs.get(f, 0))
                                                for f in critic_fields)
                    z_obs = {}
                    res.actions.append({"step": step, "action": "generate_on_bail"})
                    # forced verify
                    ok, _ = run_full_test(wd, task_id)
                    y_now = 1 if ok else 0
                    if online_kernel is not None:
                        online_kernel.update(prev_Y, y_now)
                    if (conditional_kernel is not None
                            and z_at_prev_gen is not None):
                        conditional_kernel.update(prev_Y, z_at_prev_gen, y_now)
                    res.n_full_tests += 1; res.total_cost += costs.c_full_test
                    ver_left -= 1
                    res.actions.append({"step": step + 1,
                                        "action": "verify_on_bail", "ok": ok})
                    if ok:
                        res.fixed = True
                        res.final_action = "refine_on_bail_pass"
                        break
                # verify-on-bail: forced verify ONLY if current code never verified
                # and we still have a verify budget. Guarantees an observation
                # for online/conditional kernel even when planner wants to bail.
                elif verify_on_bail and code_unverified and ver_left > 0:
                    ok, _ = run_full_test(wd, task_id)
                    y_now = 1 if ok else 0
                    if online_kernel is not None and prev_Y is not None:
                        online_kernel.update(prev_Y, y_now)
                    if (conditional_kernel is not None and prev_Y is not None
                            and z_at_prev_gen is not None):
                        conditional_kernel.update(prev_Y, z_at_prev_gen, y_now)
                    prev_Y = y_now
                    res.n_full_tests += 1; res.total_cost += costs.c_full_test
                    ver_left -= 1
                    res.actions.append({"step": step, "action": "verify_on_bail",
                                        "ok": ok})
                    if ok:
                        res.fixed = True; res.final_action = "verify_on_bail_pass"
                        break
                res.final_action = "bail"; break
            if action == "verify":
                ok, _ = run_full_test(wd, task_id)
                y_now = 1 if ok else 0
                # Online kernel update + planner re-solve
                if online_kernel is not None and prev_Y is not None:
                    online_kernel.update(prev_Y, y_now)
                    if planner_factory is not None:
                        planner = planner_factory()
                if (conditional_kernel is not None and prev_Y is not None
                        and z_at_prev_gen is not None):
                    conditional_kernel.update(prev_Y, z_at_prev_gen, y_now)
                prev_Y = y_now
                code_unverified = False
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                ver_left -= 1
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(wd, cn, task_id)
                res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used = crit_used | frozenset([cn])
                z_obs[cn] = 1 if passed else 0
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            elif action.startswith("generate:"):
                test_out = get_test_output(wd, task_id)
                prompt = PROMPT_TEMPLATE.format(
                    source_code=current,
                    test_examples=format_test_examples(task_id),
                    test_output=test_out,
                )
                r = call_llm_or_raise(prompt, llm_cfg)
                res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left -= 1
                # Apply kernel transition: prefer conditional > online > frozen.
                if conditional_kernel is not None:
                    z_at_prev_gen = tuple(int(z_obs.get(f, 0)) for f in critic_fields)
                    k = conditional_kernel.kernel_for(z_at_prev_gen)
                    belief = belief * (1 - k["p_break_correct"]) + (1 - belief) * k["p_fix_broken"]
                elif online_kernel is not None:
                    k = online_kernel.get()
                    belief = belief * (1 - k["p_break_correct"]) + (1 - belief) * k["p_fix_broken"]
                elif kernel is not None:
                    belief = belief * (1 - kernel["p_break_correct"]) + (1 - belief) * kernel["p_fix_broken"]
                else:
                    belief = belief * 0.95 + (1-belief) * 0.50
                crit_used = frozenset()
                z_obs = {}
                code_unverified = True
                res.actions.append({"step": step, "action": "generate", "b": belief})
            step += 1
        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


# ---- Resume-safe save/load ----
def load_existing(path):
    if not path.exists():
        return {"results": {}}
    with open(path) as f:
        return json.load(f)


def save_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def serialize(r):
    return {
        "task_id": r.task_id, "variant": r.variant, "cf_rating": r.cf_rating,
        "fixed": r.fixed, "total_cost": r.total_cost, "wall_clock": r.wall_clock,
        "n_llm_calls": r.n_llm_calls, "n_critic_runs": r.n_critic_runs,
        "n_full_tests": r.n_full_tests,
        "completion_tokens": r.completion_tokens,
        "final_action": r.final_action, "actions": r.actions,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CodeContests held-out agent comparison (resume-safe).")
    p.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model id (overrides ABBO_LLM_MODEL for this run).",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Output JSON path (default: sim_results/codecontests_full_endtoend.json).",
    )
    p.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated subset of: simple,greedy_hand,greedy_fitted,dp_hand,dp_fitted",
    )
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Limit number of held-out tasks (default: all).")
    p.add_argument("--kernel-mode",
                   choices=["measured", "online", "hardcoded", "thompson",
                             "conditional", "conditional_online",
                             "thompson_conditional"],
                   default="measured",
                   help="dp_fitted/greedy_fitted transition kernel mode. "
                        "'thompson': Beta posterior anchored by train raw_counts; "
                        "DP re-solved per instance with sampled kernel. "
                        "'conditional'[/_online]: per-critic-signature kernel "
                        "fit from iter_records. "
                        "'thompson_conditional': sample from per-z Beta posterior.")
    p.add_argument("--kernel-seed-dir", type=Path, default=None,
                   help="Dir containing transition_kernel.json (seed for online/thompson/conditional).")
    p.add_argument("--epsilon-thompson", type=float, default=1.0,
                   help="ε for ε-Thompson: with prob ε sample kernel from "
                        "Beta posterior, else use posterior mean. ε=0 = mean-"
                        "only (online), ε=1 = pure Thompson (default). Only "
                        "effective when --kernel-mode thompson.")
    p.add_argument("--action-gap-threshold", type=float, default=-1.0,
                   help="Gap-gated Thompson: if best vs 2nd-best Q-gap under "
                        "the mean kernel < threshold, do Thompson sample; "
                        "else exploit the mean. -1 = off. Active only when "
                        "--kernel-mode thompson.")
    p.add_argument("--uq-samples", type=int, default=0,
                   help="Posterior-Thompson UQ: at each instance, draw N "
                        "kernel samples, solve DP under each, record action "
                        "distribution + max-Q distribution + gap stats per "
                        "instance into result['uq']. 0 = off.")
    p.add_argument("--thompson-seed", type=int, default=12345,
                   help="Seed for Thompson sampling RNG (deterministic).")
    p.add_argument("--conditional-records", type=Path, action="append", default=None,
                   help="Path(s) to iter_records.jsonl for ConditionalKernel fit.")
    p.add_argument("--verify-on-bail", action="store_true",
                   help="When DP bails, force one verify (if code never verified) "
                        "to harvest a (Y_t, Y_{t+1}) observation for online/cond "
                        "kernel. Adds c_full_test cost per bail instance.")
    p.add_argument("--refine-on-bail", action="store_true",
                   help="When DP bails, force one full refinement step "
                        "(generate + verify) to harvest a real (Y_t, Y_{t+1}) "
                        "transition pair. Adds c_llm_call + c_full_test per bail.")
    p.add_argument("--gap-gated-refine", type=float, default=-1.0,
                   help="Gate refine-on-bail by Q-gap to bail. Force refine "
                        "only if Q(bail) − Q(best non-bail) < threshold. "
                        "Skip refine when bail is decisively correct. "
                        "-1 = off (always refine if --refine-on-bail).")
    p.add_argument("--cascading-refine", action="store_true",
                   help="Like refine-on-bail but loops until budget exhausted "
                        "or verify passes. Each iteration adds a kernel update.")
    p.add_argument("--max-verifications", type=int, default=MAX_VERIFICATIONS,
                   help=f"Override MAX_VERIFICATIONS (default {MAX_VERIFICATIONS}). "
                        f"Useful with --cascading-refine.")
    p.add_argument("--epsilon-explore-init", type=float, default=0.0,
                   help="Initial ε for ε-decay-on-bail. With this probability, "
                        "override bail with a forced generate. Decays linearly "
                        "to 0 over the test set.")
    p.add_argument("--epsilon-explore-seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for v in variants:
        if v not in DEFAULT_VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}; allowed: {DEFAULT_VARIANTS}")
    results_path = args.results or (ROOT / "sim_results" / "codecontests_full_endtoend.json")
    llm_model = (args.model or "").strip() or DEFAULT_LLM_MODEL

    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:]
    if args.n_tasks is not None:
        test_ids = test_ids[:args.n_tasks]
    print(f"Train: {len(train_ids)}  Held-out: {len(test_ids)}")
    print(f"Kernel mode: {args.kernel_mode}  Seed dir: {args.kernel_seed_dir}")

    state = load_existing(results_path)
    results = state.setdefault("results", {})

    # Resolve transition kernel + optional online estimator
    from abbo.realworld.agents.kernel_helpers import resolve_kernel, OnlineKernelCalibration
    base_mode = args.kernel_mode
    if base_mode in {"conditional", "conditional_online", "thompson_conditional"}:
        base_mode = "measured"
    if args.kernel_seed_dir is not None:
        active_kernel, kernel_src, online_kernel_obj = resolve_kernel(
            args.kernel_seed_dir, mode=base_mode)
    elif base_mode in ("online", "thompson") or args.kernel_mode == "thompson_conditional":
        active_kernel = {"p_fix_broken": 0.5, "p_break_correct": 0.05}
        kernel_src = "hardcoded (no seed dir)"
        online_kernel_obj = OnlineKernelCalibration(init_kernel=active_kernel)
    elif args.kernel_mode == "hardcoded":
        active_kernel = {"p_fix_broken": 0.5, "p_break_correct": 0.05}
        kernel_src = "hardcoded"
        online_kernel_obj = None
    else:
        active_kernel = None
        kernel_src = "legacy planner default"
        online_kernel_obj = None
    print(f"Kernel source: {kernel_src}  initial: {active_kernel}")
    thompson_rng = (random.Random(args.thompson_seed)
                    if args.kernel_mode in ("thompson", "thompson_conditional") else None)
    if thompson_rng is not None:
        print(f"Thompson sampling enabled (seed={args.thompson_seed})")
    explore_rng = (random.Random(args.epsilon_explore_seed)
                   if args.epsilon_explore_init > 0 else None)
    if explore_rng is not None:
        print(f"ε-decay-on-bail enabled (init={args.epsilon_explore_init}, "
              f"seed={args.epsilon_explore_seed})")

    # Conditional kernel: fit from iter_records.jsonl
    conditional_kernel_seed = None
    if args.kernel_mode in {"conditional", "conditional_online", "thompson_conditional"}:
        rec_paths = args.conditional_records or (
            [args.kernel_seed_dir / "iter_records.jsonl"]
            if args.kernel_seed_dir else [])
        if not rec_paths:
            raise SystemExit("--kernel-mode conditional requires iter_records.jsonl path")
        import sys as _sys
        _common_path = ROOT.parent.parent / "experiments" / "orchestration_hypothesis_testing"
        if not (_common_path / "_common").exists():
            raise SystemExit(f"_common not found at {_common_path}")
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
        train_in_records = [i for i in train_ids if i in by_inst] or list(by_inst.keys())
        train_trajs = [by_inst[i] for i in train_in_records]
        triples = conditional_pairs_from_trajectories(
            train_trajs, critic_fields=CRITIC_FIELDS_DEFAULT)
        seed_marginal = active_kernel or {"p_fix_broken": 0.5, "p_break_correct": 0.05}
        conditional_kernel_seed = ConditionalKernel.from_triples(
            triples, critic_fields=CRITIC_FIELDS_DEFAULT,
            init_kernel=seed_marginal, min_obs=3,
        )
        print(f"ConditionalKernel: loaded {n_loaded} records, "
              f"{len(train_trajs)} train trajectories, "
              f"{len(conditional_kernel_seed.counts)} buckets, "
              f"{len(triples)} triples")

    costs = AgentCostConfig()
    dp_hand = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                        critic_likelihoods=CC_CRITIC_LIKELIHOODS); dp_hand.solve()
    dp_fitted_kwargs = {"critic_likelihoods": FITTED_THETA}
    if active_kernel is not None:
        dp_fitted_kwargs["transition_kernel"] = active_kernel
    dp_fitted = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                          **dp_fitted_kwargs); dp_fitted.solve()

    # Factory for re-solving dp_fitted with updated online kernel
    def _factory_fitted():
        kwargs = {"critic_likelihoods": FITTED_THETA}
        if online_kernel_obj is not None:
            kwargs["transition_kernel"] = online_kernel_obj.get()
        elif active_kernel is not None:
            kwargs["transition_kernel"] = active_kernel
        p = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS, **kwargs)
        p.solve()
        return p

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

    total = len(test_ids) * len(variants)
    done = sum(1 for tid in test_ids for v in variants if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} (task, variant) pairs already done.\n")

    started = time.time()
    for i, tid in enumerate(test_ids):
        elapsed = time.time() - started
        rate = (i + 1) / max(0.001, elapsed)
        eta_min = (len(test_ids) - i - 1) / max(0.0001, rate) / 60
        m = get_metadata(tid)
        print(f"\n[{i+1}/{len(test_ids)}] task={tid}  cf={m.get('cf_rating')}  "
              f"diff={m.get('difficulty')}  elapsed={elapsed/60:.1f}min  "
              f"ETA={eta_min:.1f}min")
        for v in variants:
            key = f"{tid}|{v}"
            if results.get(key):
                continue
            try:
                if v == "simple":
                    r = run_simple(tid, llm_cfg, costs)
                elif v == "greedy_hand":
                    r = run_greedy(tid, CC_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, MAX_GENERATORS, PRIOR)
                elif v == "greedy_fitted":
                    r = run_greedy(tid, FITTED_THETA, "fitted",
                                   llm_cfg, costs, MAX_GENERATORS, PRIOR)
                elif v == "dp_hand":
                    r = run_dp(tid, CC_CRITIC_LIKELIHOODS, "hand",
                               llm_cfg, costs, dp_hand,
                               MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR)
                elif v == "dp_fitted":
                    # Thompson: sample kernel for THIS instance, re-solve DP
                    inst_kernel = active_kernel
                    inst_planner = dp_fitted
                    if args.kernel_mode == "thompson_conditional":
                        # Sample from conditional posterior at z=(0,0,0) — no critics yet
                        z_start = (0, 0, 0)
                        inst_kernel = conditional_kernel_seed.sample(thompson_rng, z_start)
                        inst_planner = DPPlanner(costs, MAX_GENERATORS,
                                                  MAX_VERIFICATIONS,
                                                  critic_likelihoods=FITTED_THETA,
                                                  transition_kernel=inst_kernel)
                        inst_planner.solve()
                        print(f"    [thomp_cond z={z_start}] "
                              f"p_fix={inst_kernel['p_fix_broken']:.3f} "
                              f"p_break={inst_kernel['p_break_correct']:.3f}")
                    elif thompson_rng is not None and online_kernel_obj is not None:
                        # ε-Thompson / gap-gated Thompson
                        mean_kernel = online_kernel_obj.get()
                        use_thompson = thompson_rng.random() < args.epsilon_thompson
                        gap_info = ""
                        if args.action_gap_threshold >= 0:
                            # Build mean-kernel planner, query Q-gap at initial state
                            mean_planner = DPPlanner(
                                costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                                critic_likelihoods=FITTED_THETA,
                                transition_kernel=mean_kernel,
                            )
                            mean_planner.solve()
                            gap, best_a, second_a = mean_planner.action_gap_at(
                                PRIOR, MAX_GENERATORS,
                                frozenset(), MAX_VERIFICATIONS)
                            use_thompson = gap < args.action_gap_threshold
                            gap_info = (f" gap={gap:.2f} "
                                        f"({best_a}↔{second_a})")
                        if use_thompson:
                            inst_kernel = online_kernel_obj.sample(thompson_rng)
                            mode_tag = "sample"
                        else:
                            inst_kernel = mean_kernel
                            mode_tag = "mean"
                        inst_kwargs = {"critic_likelihoods": FITTED_THETA,
                                        "transition_kernel": inst_kernel}
                        inst_planner = DPPlanner(costs, MAX_GENERATORS,
                                                  MAX_VERIFICATIONS, **inst_kwargs)
                        inst_planner.solve()
                        print(f"    [thompson ε={args.epsilon_thompson:.2f} "
                              f"({mode_tag}){gap_info}] "
                              f"p_fix={inst_kernel['p_fix_broken']:.3f} "
                              f"p_break={inst_kernel['p_break_correct']:.3f}")
                    cond_kernel = None
                    if args.kernel_mode == "conditional":
                        import copy
                        cond_kernel = copy.deepcopy(conditional_kernel_seed)
                    elif args.kernel_mode == "conditional_online":
                        cond_kernel = conditional_kernel_seed
                    # ε-decay: linearly decay from ε_init to 0 over test set
                    eps_now = 0.0
                    if args.epsilon_explore_init > 0:
                        progress = i / max(1, len(test_ids) - 1)
                        eps_now = max(0.0, args.epsilon_explore_init * (1.0 - progress))

                    # UQ via Thompson posterior sampling: draw N samples and
                    # collect action distribution + Q-stats at initial state.
                    uq_record = None
                    if args.uq_samples > 0 and online_kernel_obj is not None:
                        from collections import Counter
                        from math import log
                        actions_seen = []
                        max_qs = []
                        gaps = []
                        for _ in range(args.uq_samples):
                            k_smp = online_kernel_obj.sample(thompson_rng) if thompson_rng else online_kernel_obj.get()
                            tmp = DPPlanner(costs, MAX_GENERATORS,
                                              MAX_VERIFICATIONS,
                                              critic_likelihoods=FITTED_THETA,
                                              transition_kernel=k_smp)
                            tmp.solve()
                            qs = tmp.q_values_at(PRIOR, MAX_GENERATORS,
                                                  frozenset(),
                                                  MAX_VERIFICATIONS)
                            ranked = sorted(qs.items(), key=lambda x: -x[1])
                            actions_seen.append(ranked[0][0])
                            max_qs.append(ranked[0][1])
                            gaps.append(ranked[0][1] - ranked[1][1]
                                         if len(ranked) > 1 else 0.0)
                        cnt = Counter(actions_seen)
                        top_a, top_n = cnt.most_common(1)[0]
                        N = args.uq_samples
                        probs = [c / N for c in cnt.values()]
                        entropy = -sum(p * log(p) for p in probs if p > 0)
                        max_q_sorted = sorted(max_qs)
                        ci_lo = max_q_sorted[int(0.025 * N)]
                        ci_hi = max_q_sorted[int(0.975 * N) if int(0.975*N) < N else N-1]
                        p_gap_small_05 = sum(1 for g in gaps if g < 0.5) / N
                        p_gap_small_10 = sum(1 for g in gaps if g < 1.0) / N
                        p_gap_small_20 = sum(1 for g in gaps if g < 2.0) / N
                        uq_record = {
                            "n_samples": N,
                            "action_freq": dict(cnt),
                            "top_action": top_a,
                            "top_action_prob": top_n / N,
                            "action_entropy": entropy,
                            "max_q_mean": sum(max_qs) / N,
                            "max_q_std": (sum((q - sum(max_qs)/N)**2 for q in max_qs) / N) ** 0.5,
                            "max_q_ci95": [ci_lo, ci_hi],
                            "gap_mean": sum(gaps) / N,
                            "p_gap_lt_0_5": p_gap_small_05,
                            "p_gap_lt_1_0": p_gap_small_10,
                            "p_gap_lt_2_0": p_gap_small_20,
                        }
                        print(f"    [UQ N={N}] top={top_a} p={top_n/N:.2f} "
                              f"H={entropy:.2f} max_Q_CI=[{ci_lo:.1f},{ci_hi:.1f}] "
                              f"P(gap<0.5)={p_gap_small_05:.2f}")

                    r = run_dp(tid, FITTED_THETA, "fitted",
                               llm_cfg, costs, inst_planner,
                               MAX_GENERATORS, args.max_verifications, PRIOR,
                               kernel=inst_kernel,
                               online_kernel=online_kernel_obj,
                               planner_factory=_factory_fitted,
                               conditional_kernel=cond_kernel,
                               verify_on_bail=args.verify_on_bail,
                               refine_on_bail=args.refine_on_bail,
                               gap_gated_refine=args.gap_gated_refine,
                               cascading_refine=args.cascading_refine,
                               epsilon_explore=eps_now,
                               explore_rng=explore_rng)
                else:
                    continue
            except Exception as e:
                print(f"  [{v}] EXCEPTION: {e}")
                continue
            results[key] = serialize(r)
            if v == "dp_fitted" and uq_record is not None:
                results[key]["uq"] = uq_record
            tag = "OK" if r.fixed else "no"
            print(f"  {v:<16} fix={tag}  cost={r.total_cost:5.1f}  "
                  f"llm={r.n_llm_calls}  crit={r.n_critic_runs}  "
                  f"toks={r.completion_tokens}  wc={r.wall_clock:.1f}s  "
                  f"final={r.final_action}")
            save_progress(results_path, state)

    # Final aggregate
    print("\n=== Final aggregate ===")
    R = 100
    from collections import defaultdict
    by_v = defaultdict(list)
    for rec in results.values():
        by_v[rec["variant"]].append(rec)
    print(f"{'variant':<16} {'n':>4} {'fix%':>6} {'cost':>7} {'Ū_π':>8} {'Δ_π':>8}")
    print('-' * 55)
    if "simple" in by_v and by_v["simple"]:
        baseline = sum((R if r["fixed"] else 0) - r["total_cost"]
                       for r in by_v["simple"]) / len(by_v["simple"])
    else:
        baseline = 0.0
    for v in variants:
        rs = by_v.get(v, [])
        if not rs: continue
        n = len(rs)
        fix = sum(1 for r in rs if r["fixed"]) / n * 100
        c = sum(r["total_cost"] for r in rs) / n
        u = sum((R if r["fixed"] else 0) - r["total_cost"] for r in rs) / n
        d = u - baseline
        print(f"{v:<16} {n:>4} {fix:>5.1f}% {c:>7.2f} {u:>8.2f} {d:>+8.2f}")

    state["llm_model"] = llm_cfg.model
    state["fitted_theta"] = FITTED_THETA
    state["n_train"] = len(train_ids)
    state["n_test"] = len(test_ids)
    save_progress(results_path, state)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
