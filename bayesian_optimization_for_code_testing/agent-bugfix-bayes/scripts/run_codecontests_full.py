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
import re  # noqa: F401 (also used for model slug)
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.kernel_helpers import (
    DEFAULT_KERNEL, kernel_update, resolve_kernel, OnlineKernelCalibration,
)
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES,
    get_solution_pool, get_test_cases, get_metadata,
    list_task_ids, run_critic, run_full_test,
)
from abbo.realworld.agents.llm_provider import build_llm_config_from_env, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.telemetry import TelemetryLogger, write_action

# ---- Config ----
SPLIT_SEED = 42
TRAIN_FRAC = 0.75   # default train/test split: 75% train, 25% test
N_TRAIN = None      # if None, computed as int(round(TRAIN_FRAC * n_total)); override via --n-train
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b:free"
# Backward compat for scripts that import LLM_MODEL from this module
LLM_MODEL = DEFAULT_LLM_MODEL

DEFAULT_VARIANTS = (
    "simple", "best_of_3",
    "threshold_L0", "threshold_L2", "threshold_L3", "fixed_pipeline",
    "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted",
    "self_refine", "reflexion",
)

# CodeContests critic ordering (L0 → L1 → L2 → L3 equivalents)
CC_CRITICS_ORDERED = ["critic_syntax", "critic_lint", "critic_early", "critic_mid"]
THRESHOLD_CRITICS = {
    "threshold_L0": ["critic_syntax"],
    "threshold_L2": ["critic_syntax", "critic_early"],
    "threshold_L3": CC_CRITICS_ORDERED,
}

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
def run_simple(task_id, llm_cfg, costs, n_retries=3, logger=None, run_id=None):
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
            t0 = time.perf_counter()
            r = call_llm_or_raise(prompt, llm_cfg)
            write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                         instance_id=task_id, action_type="generate",
                         runtime_s=time.perf_counter() - t0, model_name=llm_cfg.model)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            current = extract_code(r.text, current)
            sol.write_text(current)
            t0 = time.perf_counter()
            ok, _ = run_full_test(wd, task_id)
            write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                         instance_id=task_id, action_type="verify",
                         runtime_s=time.perf_counter() - t0, passed=ok)
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


def run_greedy(task_id, theta, label, llm_cfg, costs, max_gen=3, prior=0.5,
               logger=None, run_id=None, kernel=None, online_kernel=None):
    """kernel: dict {p_fix_broken, p_break_correct}. If None, uses DEFAULT_KERNEL.
    online_kernel: OnlineKernelCalibration; if provided, transitions are recorded
                    and the kernel updates after each verify event."""
    if kernel is None:
        kernel = DEFAULT_KERNEL
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
        prev_Y = 0  # seed is buggy by construction
        while step < 12:
            active_kernel = online_kernel.get() if online_kernel is not None else kernel
            Q_bail = 0.0
            Q_verify = -costs.c_full_test + belief * costs.reward
            Q_critics = {c: _q_one_step_critic(belief, c, theta, costs)
                         for c in theta if c not in crit_used}
            best_c, best_q = (max(Q_critics.items(), key=lambda x: x[1])
                              if Q_critics else (None, -math.inf))
            Q_gen = -math.inf
            if gen_left > 0:
                b_after = kernel_update(belief, active_kernel)
                Q_gen = -costs.c_llm_call - costs.c_full_test + b_after * costs.reward
            choices = [("bail", Q_bail), ("verify", Q_verify)]
            if best_c: choices.append((f"critic:{best_c}", best_q))
            if gen_left > 0: choices.append(("generate", Q_gen))
            action, _q = max(choices, key=lambda x: x[1])

            if action == "bail":
                res.final_action = "bail"; break
            if action == "verify":
                t0 = time.perf_counter()
                ok, _ = run_full_test(wd, task_id)
                write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                             instance_id=task_id, action_type="verify",
                             runtime_s=time.perf_counter() - t0, passed=ok,
                             belief_before=belief)
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                y_now = 1 if ok else 0
                if online_kernel is not None and prev_Y is not None:
                    online_kernel.update(prev_Y, y_now)
                prev_Y = y_now
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                t0 = time.perf_counter()
                passed, _ = run_critic(wd, cn, task_id)
                write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                             instance_id=task_id, action_type=cn,
                             runtime_s=time.perf_counter() - t0, passed=passed,
                             belief_before=belief)
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
                t0 = time.perf_counter()
                r = call_llm_or_raise(prompt, llm_cfg)
                write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                             instance_id=task_id, action_type="generate",
                             runtime_s=time.perf_counter() - t0,
                             model_name=llm_cfg.model, belief_before=belief)
                res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left -= 1
                belief = kernel_update(belief, active_kernel)
                crit_used = set()
                res.actions.append({"step": step, "action": "generate", "b": belief})
            step += 1
        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_dp(task_id, theta, label, llm_cfg, costs, make_planner,
           max_gen=3, max_ver=2, prior=0.5, logger=None, run_id=None,
           kernel=None, online_kernel=None):
    """make_planner: callable(critic_likelihoods, transition_kernel) -> DPPlanner (pre-solved).
    kernel: dict {p_fix_broken, p_break_correct}; if None, uses DEFAULT_KERNEL.
    online_kernel: OnlineKernelCalibration; if provided, kernel updates after each verify
                    and planner is re-solved with the new kernel."""
    if kernel is None:
        kernel = DEFAULT_KERNEL
    res = Result(task_id=task_id, variant=f"dp_{label}",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    # Initial planner solve (with current kernel)
    active_kernel = online_kernel.get() if online_kernel is not None else kernel
    planner = make_planner(theta, active_kernel)
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        belief = prior; gen_left = max_gen; ver_left = max_ver
        crit_used: frozenset[str] = frozenset(); step = 0
        prev_Y = 0  # seed is buggy by construction
        while step < 16:
            action, _q = planner.choose_action(
                belief, gen_left, crit_used, ver_left,
            )
            if action == "bail_out":
                res.final_action = "bail"; break
            if action == "verify":
                t0 = time.perf_counter()
                ok, _ = run_full_test(wd, task_id)
                write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                             instance_id=task_id, action_type="verify",
                             runtime_s=time.perf_counter() - t0, passed=ok,
                             belief_before=belief)
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                ver_left -= 1
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                y_now = 1 if ok else 0
                if online_kernel is not None and prev_Y is not None:
                    online_kernel.update(prev_Y, y_now)
                    # Re-solve planner with updated kernel for subsequent decisions
                    planner = make_planner(theta, online_kernel.get())
                prev_Y = y_now
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                t0 = time.perf_counter()
                passed, _ = run_critic(wd, cn, task_id)
                write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                             instance_id=task_id, action_type=cn,
                             runtime_s=time.perf_counter() - t0, passed=passed,
                             belief_before=belief)
                res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used = crit_used | frozenset([cn])
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            elif action.startswith("generate:"):
                test_out = get_test_output(wd, task_id)
                prompt = PROMPT_TEMPLATE.format(
                    source_code=current,
                    test_examples=format_test_examples(task_id),
                    test_output=test_out,
                )
                t0 = time.perf_counter()
                r = call_llm_or_raise(prompt, llm_cfg)
                write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                             instance_id=task_id, action_type="generate",
                             runtime_s=time.perf_counter() - t0,
                             model_name=llm_cfg.model, belief_before=belief)
                res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left -= 1
                active_kernel = online_kernel.get() if online_kernel is not None else kernel
                belief = kernel_update(belief, active_kernel)
                crit_used = frozenset()
                res.actions.append({"step": step, "action": "generate", "b": belief})
            step += 1
        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


REFINE_SUFFIX = """

The previous attempt did not pass critics. Specifically:
{feedback}

Provide a revised COMPLETE Python program addressing these issues.
Return ONLY the code, no explanation."""

REFLEXION_SUFFIX = """

The previous attempt failed the test suite. Test output:
{test_feedback}

Reflect on what went wrong and provide a corrected complete program.
Return ONLY the code, no explanation."""


def _make_prompt(task_id, source_code, test_out, suffix=""):
    return PROMPT_TEMPLATE.format(
        source_code=source_code,
        test_examples=format_test_examples(task_id),
        test_output=test_out,
    ) + suffix


def _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id, belief=None):
    t0 = time.perf_counter()
    r = call_llm_or_raise(prompt, llm_cfg)
    write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                 instance_id=task_id, action_type="generate",
                 runtime_s=time.perf_counter() - t0,
                 model_name=llm_cfg.model, belief_before=belief)
    res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
    res.completion_tokens += r.completion_tokens
    return r


def _verify(wd, task_id, costs, res, logger, run_id, belief=None):
    t0 = time.perf_counter()
    ok, _ = run_full_test(wd, task_id)
    write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                 instance_id=task_id, action_type="verify",
                 runtime_s=time.perf_counter() - t0, passed=ok,
                 belief_before=belief)
    res.n_full_tests += 1; res.total_cost += costs.c_full_test
    return ok


def _critic(wd, cn, task_id, costs, res, logger, run_id, belief=None):
    t0 = time.perf_counter()
    passed, msg = run_critic(wd, cn, task_id)
    write_action(logger, run_id=run_id or task_id, dataset="codecontests",
                 instance_id=task_id, action_type=cn,
                 runtime_s=time.perf_counter() - t0, passed=passed,
                 belief_before=belief)
    res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
    return passed, msg


def run_best_of_n(task_id, llm_cfg, costs, n=3, logger=None, run_id=None):
    res = Result(task_id=task_id, variant="best_of_3",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start; return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        for attempt in range(n):
            sol.write_text(buggy)
            test_out = get_test_output(wd, task_id)
            r = _gen(_make_prompt(task_id, buggy, test_out), llm_cfg, costs,
                     res, logger, run_id, task_id)
            current = extract_code(r.text, buggy); sol.write_text(current)
            ok = _verify(wd, task_id, costs, res, logger, run_id)
            res.actions.append({"step": attempt, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_threshold(task_id, variant_name, critics, llm_cfg, costs,
                  max_gen=3, logger=None, run_id=None):
    res = Result(task_id=task_id, variant=variant_name,
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start; return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        for attempt in range(max_gen):
            sol.write_text(buggy)
            test_out = get_test_output(wd, task_id)
            r = _gen(_make_prompt(task_id, buggy, test_out), llm_cfg, costs,
                     res, logger, run_id, task_id)
            current = extract_code(r.text, buggy); sol.write_text(current)
            gate = True
            for cn in critics:
                passed, _ = _critic(wd, cn, task_id, costs, res, logger, run_id)
                res.actions.append({"step": attempt, "action": f"critic:{cn}", "passed": passed})
                if not passed:
                    gate = False; break
            if gate:
                ok = _verify(wd, task_id, costs, res, logger, run_id)
                res.actions.append({"step": attempt, "action": "verify", "ok": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_fixed_pipeline(task_id, llm_cfg, costs, max_gen=3, logger=None, run_id=None):
    res = run_threshold(task_id, "fixed_pipeline", CC_CRITICS_ORDERED,
                        llm_cfg, costs, max_gen, logger, run_id)
    res.variant = "fixed_pipeline"
    return res


def run_self_refine(task_id, llm_cfg, costs, max_rounds=2, logger=None, run_id=None):
    res = Result(task_id=task_id, variant="self_refine",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start; return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        current_source = buggy
        prompt_suffix = ""
        for rnd in range(max_rounds + 1):
            sol.write_text(buggy)
            test_out = get_test_output(wd, task_id)
            prompt = _make_prompt(task_id, current_source, test_out, prompt_suffix)
            r = _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id)
            current_source = extract_code(r.text, current_source)
            sol.write_text(current_source)
            failures = []
            for cn in CC_CRITICS_ORDERED:
                passed, msg = _critic(wd, cn, task_id, costs, res, logger, run_id)
                if not passed:
                    failures.append(f"- {cn}: {msg}")
            res.actions.append({"round": rnd, "action": "critics",
                                "n_failed": len(failures)})
            if not failures or rnd == max_rounds:
                ok = _verify(wd, task_id, costs, res, logger, run_id)
                res.actions.append({"round": rnd, "action": "verify", "ok": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"
                break
            prompt_suffix = REFINE_SUFFIX.format(feedback="\n".join(failures))
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_reflexion(task_id, llm_cfg, costs, max_rounds=2, logger=None, run_id=None):
    res = Result(task_id=task_id, variant="reflexion",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start; return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        current_source = buggy
        prompt_suffix = ""
        for rnd in range(max_rounds + 1):
            sol.write_text(buggy)
            test_out = get_test_output(wd, task_id)
            prompt = _make_prompt(task_id, current_source, test_out, prompt_suffix)
            r = _gen(prompt, llm_cfg, costs, res, logger, run_id, task_id)
            current_source = extract_code(r.text, current_source)
            sol.write_text(current_source)
            ok = _verify(wd, task_id, costs, res, logger, run_id)
            res.actions.append({"round": rnd, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            if rnd < max_rounds:
                test_fb = get_test_output(wd, task_id)
                prompt_suffix = REFLEXION_SUFFIX.format(test_feedback=test_fb[:600])
        if not res.fixed:
            res.final_action = res.final_action or "exhausted"
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
        help="Transition kernel source for greedy/dp variants. "
             "'measured' (default): load transition_kernel.json from --kernel-dir; "
             "fallback to hardcoded (0.50, 0.05) if missing. "
             "'online': start from measured and update via Beta-Binomial after each verify. "
             "'hardcoded': force legacy (0.50, 0.05).",
    )
    p.add_argument(
        "--kernel-dir",
        type=Path,
        default=None,
        help="Directory containing transition_kernel.json (typically "
             "data/codecontests_iter/<generator>/ from iter_refine_bugfix.py output, "
             "or sim_results/transition_kernels.json). "
             "If not specified, looks at sim_results/transition_kernels.json "
             "and uses the 'code_contests' key.",
    )
    p.add_argument(
        "--refit-theta-from",
        type=Path,
        default=None,
        help="Path to critic_results.jsonl from a calibration run. If provided, "
             "refits FITTED_THETA from records with instance_id in train_ids only "
             "(no leakage). Overrides the hardcoded FITTED_THETA constant.",
    )
    p.add_argument(
        "--refit-kernel-from",
        type=Path,
        default=None,
        help="Path to iter_records.jsonl from a calibration run. If provided, "
             "refits transition kernel from records with instance_id in train_ids "
             "only (no leakage). Overrides --kernel-mode/--kernel-dir loading.",
    )
    p.add_argument(
        "--n-train",
        type=int,
        default=None,
        help=f"Number of train instances. Default: int(round({TRAIN_FRAC} * n_total)) "
             "= 75/25 train/test split. Pass e.g. --n-train 30 to reproduce legacy "
             "30/82 split on CodeContests.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for v in variants:
        if v not in DEFAULT_VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}; allowed: {DEFAULT_VARIANTS}")
    results_path = args.results or (ROOT / "sim_results" / "codecontests_full_endtoend.json")
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
    # Sanity: no overlap (guaranteed by slice but assert anyway)
    assert set(train_ids).isdisjoint(set(test_ids)), \
        "train/test overlap — leakage in split!"

    # Optionally refit FITTED_THETA from train_ids only (no leakage)
    fitted_theta_active = FITTED_THETA
    refit_source = "hardcoded constant"
    if args.refit_theta_from:
        crit_path = args.refit_theta_from
        if not crit_path.exists():
            raise SystemExit(f"--refit-theta-from path missing: {crit_path}")
        # Read calibration records, filter to train_ids, refit per critic
        train_id_set = set(train_ids)
        records = []
        for line in crit_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(r.get("instance_id") or r.get("task_id"))
            if iid in train_id_set:
                records.append(r)
        # Beta(1,1) smoothing
        refit = {}
        for cn in CC_CRITIC_NAMES:
            tp = fp = tn = fn = 0
            for r in records:
                z = r.get(cn)
                if z is None:
                    continue
                y = int(r.get("Y", 0))
                p = bool(z)
                if   y == 1 and p: tp += 1
                elif y == 0 and p: fp += 1
                elif y == 0 and not p: tn += 1
                elif y == 1 and not p: fn += 1
            p_y1 = (tp + 1) / (tp + fn + 2) if (tp + fn + 2) > 0 else 0.5
            p_y0 = (fp + 1) / (fp + tn + 2) if (fp + tn + 2) > 0 else 0.5
            refit[cn] = {"p_pass_y1": p_y1, "p_pass_y0": p_y0}
        fitted_theta_active = refit
        refit_source = f"refit from train ({len(records)} records, {len(train_id_set)} train_ids)"
        print(f"FITTED_THETA refit from {crit_path}:")
        for cn, lk in refit.items():
            print(f"  {cn}: p_pass_y1={lk['p_pass_y1']:.4f}  p_pass_y0={lk['p_pass_y0']:.4f}")
    print(f"FITTED_THETA source: {refit_source}")

    state = load_existing(results_path)
    results = state.setdefault("results", {})

    costs = AgentCostConfig()

    # Resolve transition kernel — refit from train_ids if requested, else load
    kernel_dir = args.kernel_dir
    if args.refit_kernel_from is not None:
        if not args.refit_kernel_from.exists():
            raise SystemExit(f"--refit-kernel-from path missing: {args.refit_kernel_from}")
        train_id_set = set(train_ids)
        by_inst: dict[str, list[dict]] = {}
        for line in args.refit_kernel_from.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(r.get("instance_id") or r.get("task_id"))
            if iid not in train_id_set:
                continue
            by_inst.setdefault(iid, []).append(r)
        # Sort by step / patch_id and count Y[i]->Y[i+1]
        counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        for rows in by_inst.values():
            rows.sort(key=lambda r: int(r.get("step", r.get("patch_id", 0))))
            for i in range(len(rows) - 1):
                y0 = rows[i].get("Y"); y1 = rows[i+1].get("Y")
                if y0 is None or y1 is None:
                    continue
                counts[f"{int(y0)}->{int(y1)}"] += 1
        n_broken = counts["0->0"] + counts["0->1"]
        n_correct = counts["1->0"] + counts["1->1"]
        p_fix = (counts["0->1"] + 1) / (n_broken + 2) if (n_broken + 2) > 0 else 0.5
        p_break = (counts["1->0"] + 1) / (n_correct + 2) if (n_correct + 2) > 0 else 0.5
        kernel = {"p_fix_broken": p_fix, "p_break_correct": p_break}
        kernel_src = (f"refit from train ({sum(counts.values())} pairs, "
                      f"{len(by_inst)} train_ids, raw_counts={counts})")
        online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                         if args.kernel_mode == "online" else None)
    elif kernel_dir is None:
        # Try sim_results/transition_kernels.json (per-benchmark) as fallback
        legacy_path = ROOT / "sim_results" / "transition_kernels.json"
        if legacy_path.exists():
            jj = json.loads(legacy_path.read_text())
            cc = jj.get("code_contests", {})
            if "p_fix_broken" in cc and "p_break_correct" in cc:
                # Write to a tmp gen_dir-like place so resolve_kernel can pick it up
                # OR just construct kernel directly:
                if args.kernel_mode == "hardcoded":
                    kernel = DEFAULT_KERNEL.copy(); kernel_src = "hardcoded (forced)"
                    online_kernel = None
                else:
                    measured = {"p_fix_broken": float(cc["p_fix_broken"]),
                                 "p_break_correct": float(cc["p_break_correct"]
                                                          if cc.get("p_break_correct") is not None
                                                          else jj.get("prior_break_correct", 0.05))}
                    kernel = measured
                    kernel_src = "measured (sim_results/transition_kernels.json)"
                    online_kernel = (OnlineKernelCalibration(init_kernel=measured)
                                      if args.kernel_mode == "online" else None)
            else:
                kernel = DEFAULT_KERNEL.copy(); kernel_src = "default (no kernel data)"
                online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if args.kernel_mode == "online" else None)
        else:
            kernel = DEFAULT_KERNEL.copy(); kernel_src = "default (no kernel file)"
            online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                              if args.kernel_mode == "online" else None)
    else:
        kernel, kernel_src, online_kernel = resolve_kernel(kernel_dir, args.kernel_mode)
    print(f"Transition kernel: source={kernel_src}, "
          f"p_fix={kernel['p_fix_broken']:.3f}, p_break={kernel['p_break_correct']:.3f}, "
          f"mode={args.kernel_mode}")

    # Factory for DPPlanner: builds (and solves) a fresh planner given
    # (critic_likelihoods, transition_kernel). Used both for initial solve
    # and for online-mode re-solves inside run_dp.
    def make_dp(theta_dict, kern):
        p = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                      critic_likelihoods=theta_dict,
                      transition_kernel=kern)
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

    _model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", llm_cfg.model)
    telemetry_path = ROOT / "logs" / f"action_telemetry_codecontests__{_model_slug}.jsonl"
    tlog = TelemetryLogger(telemetry_path)
    print(f"Telemetry → {telemetry_path}")

    started = time.time()
    try:
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
                run_id = f"{_model_slug}__{tid}__{v}"
                try:
                    if v == "simple":
                        r = run_simple(tid, llm_cfg, costs,
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
                        r = run_greedy(tid, CC_CRITIC_LIKELIHOODS, "hand",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR,
                                       logger=tlog, run_id=run_id,
                                       kernel=kernel, online_kernel=online_kernel)
                    elif v == "greedy_fitted":
                        r = run_greedy(tid, fitted_theta_active, "fitted",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR,
                                       logger=tlog, run_id=run_id,
                                       kernel=kernel, online_kernel=online_kernel)
                    elif v == "dp_hand":
                        r = run_dp(tid, CC_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, make_dp,
                                   MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR,
                                   logger=tlog, run_id=run_id,
                                   kernel=kernel, online_kernel=online_kernel)
                    elif v == "dp_fitted":
                        r = run_dp(tid, fitted_theta_active, "fitted",
                                   llm_cfg, costs, make_dp,
                                   MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR,
                                   logger=tlog, run_id=run_id,
                                   kernel=kernel, online_kernel=online_kernel)
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
                results[key] = serialize(r)
                tag = "OK" if r.fixed else "no"
                print(f"  {v:<16} fix={tag}  cost={r.total_cost:5.1f}  "
                      f"llm={r.n_llm_calls}  crit={r.n_critic_runs}  "
                      f"toks={r.completion_tokens}  wc={r.wall_clock:.1f}s  "
                      f"final={r.final_action}")
                save_progress(results_path, state)
    finally:
        tlog.close()

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
    state["fitted_theta"] = fitted_theta_active
    state["fitted_theta_source"] = refit_source
    state["n_train"] = len(train_ids)
    state["n_test"] = len(test_ids)
    state["kernel_mode"] = args.kernel_mode
    state["kernel_source"] = kernel_src
    state["transition_kernel_init"] = kernel
    if online_kernel is not None:
        state["transition_kernel_final"] = online_kernel.get()
        state["online_kernel_summary"] = online_kernel.summary()
    save_progress(results_path, state)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
