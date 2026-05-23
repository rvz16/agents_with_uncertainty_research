#!/usr/bin/env python
"""End-to-end agent run on SWE-Bench Lite, mirroring run_codecontests_full.py.

Per instance:
  1. Pull image + start container (cached after first run)
  2. Reset to base + apply test_patch (Y=0 starting state)
  3. For each agent variant:
        a. Reset state again (clean slate per variant)
        b. Run agent loop: critic / verify / generate (LLM produces unified diff)
        c. Apply LLM patches via git apply inside container
        d. Verify = run FAIL_TO_PASS + PASS_TO_PASS

Resume-safe: saves after every (instance, variant) pair.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.kernel_helpers import (
    DEFAULT_KERNEL, kernel_update, resolve_kernel, OnlineKernelCalibration,
)
from abbo.realworld.agents.llm_provider import build_llm_config_from_env, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.swe_bench import (
    SWE_CRITIC_LIKELIHOODS, SWE_INSTANCE_POOL, SWE_CRITIC_NAMES,
    PYTEST_ENV_PREFIX,
    _exec, _exec_stdin,
    changed_files_from_patch, get_ftp, get_instance,
    list_instance_ids, prepare_repo, pull_image,
    run_critic, run_full_test,
    start_container, stop_container,
)
from abbo.realworld.telemetry import TelemetryLogger, measured_action


# ---- Knobs ----
SPLIT_SEED = 42
TRAIN_FRAC = 0.75   # default 75/25 train/test split
N_TRAIN = None      # if None, computed as int(round(TRAIN_FRAC * n_total)); override via ABBO_N_TRAIN env or --n-train
PRIOR = 0.5
MAX_GENERATORS = 2       # SWE patches are big — keep budget small
MAX_VERIFICATIONS = 1
LLM_MODEL = os.environ.get("ABBO_LLM_MODEL", "openai/gpt-oss-20b:free")
_model_slug = LLM_MODEL.split("/")[-1].replace(":", "_").replace(".", "_")
RESULTS_PATH = ROOT / "sim_results" / f"swebench_full_endtoend__{_model_slug}.json"

VARIANTS = (
    "simple",
    "best_of_3",
    "threshold_L0", "threshold_L2", "threshold_L3",
    "fixed_pipeline",
    "greedy_hand", "greedy_fitted",
    "dp_hand", "dp_fitted",
    "self_refine", "reflexion",
)
TELEMETRY_PATH = ROOT / "logs" / f"action_telemetry_swebench__{_model_slug}.jsonl"

# Cached fitted theta from the SWE calibration run (allure artifact 9e7fd0d7)
FITTED_THETA = {
    "critic_syntax": {"p_pass_y1": 0.914, "p_pass_y0": 0.864},
    "critic_lint":   {"p_pass_y1": 0.247, "p_pass_y0": 0.197},
    "critic_early":  {"p_pass_y1": 0.667, "p_pass_y0": 0.444},
    "critic_mid":    {"p_pass_y1": 0.667, "p_pass_y0": 0.222},
}

# Measured kernel placeholder (we have NO iter data for SWE yet).
# Fall back to literature-prior numbers from the supervisor deck (gpt5_mini
# SWE-Verified): P(fix|broken)=0.13, P(break|correct)=0.06.
SWE_KERNEL_LITERATURE = {"p_fix_broken": 0.13, "p_break_correct": 0.06}


PROMPT_TEMPLATE = """You are a software engineer fixing a bug in a Python repository.

Issue:
{issue}

Files you may need to modify (current contents shown below):
{files_block}

Produce one or more SEARCH/REPLACE blocks that fix the bug. Each block must be:

```
<<<<<<< SEARCH path/to/file.py
exact lines to find
(must match file contents byte-for-byte including indentation)
=======
exact replacement lines
>>>>>>> REPLACE
```

Return ONLY the SEARCH/REPLACE blocks (no explanation, no markdown fence around the whole thing).
Keep blocks small and targeted; one block per change-site."""


# SEARCH/REPLACE block parser — tolerates 5-7 angle-bracket chars and case variations
SR_BLOCK_RE = re.compile(
    r"<{5,7}\s*SEARCH\s+([^\n]+)\n(.*?)\n={5,7}\n(.*?)\n>{5,7}\s*REPLACE",
    re.DOTALL | re.IGNORECASE,
)

# Unified-diff code-fence extractor (```diff … ``` or ```patch … ```)
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)[^\n]*\n(.*?)```", re.DOTALL)

# Thinking-block stripper (qwen3-coder wraps CoT in <think>…</think>)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_sr_blocks(llm_text: str) -> list[tuple[str, str, str]]:
    """Returns list of (file_path, search_text, replace_text). Empty if none found."""
    return [(m.group(1).strip(), m.group(2), m.group(3))
            for m in SR_BLOCK_RE.finditer(llm_text)]


def _looks_like_diff(text: str) -> bool:
    return bool(re.search(r"^(?:---\s|\+\+\+\s|@@\s)", text, re.MULTILINE))


def _apply_unified_diff(cname: str, diff_text: str) -> bool:
    r = _exec_stdin(
        cname,
        "cd /testbed && git apply --ignore-whitespace --recount -",
        diff_text,
        timeout=30,
    )
    return r.returncode == 0


def get_files_block(cname: str, instance: dict, max_chars_per_file: int = 6000) -> str:
    """Cat the files touched by the gold patch (we tell the LLM which files
    to look at — that's a fair scaffolding, not the answer)."""
    try:
        paths = changed_files_from_patch(instance.get("patch") or "")[:3]
    except Exception:
        paths = []
    parts = []
    for p in paths:
        if not isinstance(p, str) or not p:
            continue
        try:
            r = _exec(cname, f"cat /testbed/{p} 2>&1 | head -200", timeout=30)
            body = (r.stdout or "")[:max_chars_per_file]
            parts.append(f"### {p}\n```python\n{body}\n```")
        except Exception:
            continue
    return "\n\n".join(parts) if parts else "(no files identified)"


def apply_llm_patch(cname: str, llm_text: str) -> tuple[bool, int, int]:
    """Apply LLM patch to /testbed.

    Tries SEARCH/REPLACE blocks first, then falls back to unified diff via
    git apply. Strips <think>…</think> blocks before parsing (qwen3-coder).

    Returns (any_applied, n_sr_blocks_found, n_sr_blocks_applied).
    """
    import base64
    llm_text = _strip_thinking(llm_text or "")

    # --- Method 1: SEARCH/REPLACE blocks ---
    try:
        blocks = extract_sr_blocks(llm_text)
    except Exception:
        blocks = []

    n_applied = 0
    for path, search, replace in blocks:
        try:
            if not path or not path.endswith(".py"):
                continue
            r = _exec(cname, f"cat /testbed/{path}", timeout=20)
            if r.returncode != 0 or r.stdout is None:
                continue
            content = r.stdout
            if not search or search not in content:
                continue
            new_content = content.replace(search, replace or "", 1)
            b64 = base64.b64encode(new_content.encode("utf-8")).decode("ascii")
            wr = _exec(cname, f"echo '{b64}' | base64 -d > /testbed/{path}", timeout=20)
            if wr.returncode == 0:
                n_applied += 1
        except Exception:
            continue

    if n_applied > 0:
        return True, len(blocks), n_applied

    # --- Method 2: unified diff via git apply ---
    # Prefer explicit ```diff fences; fall back to raw text.
    diff_candidates = [m.group(1) for m in _DIFF_FENCE_RE.finditer(llm_text)]
    diff_candidates.append(llm_text)
    for candidate in diff_candidates:
        if _looks_like_diff(candidate) and _apply_unified_diff(cname, candidate):
            return True, len(blocks), 1

    return False, len(blocks), 0


def reset_repo_for_variant(cname: str, instance: dict) -> None:
    """Reset to base + test_patch (no fix). Run between variants."""
    prepare_repo(cname, instance, apply_fix=False)


def _tlog(logger, run_id, instance_id, action_type, runtime_s,
          passed=None, belief_before=None, model_name=None, metadata=None):
    """Write one telemetry record if logger is provided."""
    if logger is None:
        return
    logger.write({
        "run_id": run_id,
        "dataset": "swebench",
        "instance_id": instance_id,
        "action_type": action_type,
        "runtime_seconds": runtime_s,
        "model_name": model_name,
        "passed": passed,
        "belief_before": belief_before,
        "metadata": metadata or {},
    })


# ---- Result ----
@dataclass
class Result:
    instance_id: str
    variant: str
    fixed: bool = False
    total_cost: float = 0.0
    wall_clock: float = 0.0
    n_llm_calls: int = 0
    n_critic_runs: int = 0
    n_full_tests: int = 0
    n_patch_apply_fails: int = 0
    completion_tokens: int = 0
    final_action: str = ""
    actions: list = field(default_factory=list)


# ---- Variants ----
def run_simple(instance, cname, llm_cfg, costs, n_retries=2,
               logger=None, run_id=None):
    res = Result(instance_id=instance["instance_id"], variant="simple")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    for attempt in range(n_retries):
        files = get_files_block(cname, instance)
        prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
        t0 = time.perf_counter()
        r = call_llm_or_raise(prompt, llm_cfg)
        _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0,
              model_name=llm_cfg.model)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"step": attempt, "n_blocks": n_blocks,
                                "n_applied": n_ok, "applied": False})
            reset_repo_for_variant(cname, instance)
            continue
        t0 = time.perf_counter()
        ok, _ = run_full_test(cname, instance)
        _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0, passed=ok)
        res.n_full_tests += 1
        res.total_cost += costs.c_full_test
        res.actions.append({"step": attempt, "n_blocks": n_blocks,
                            "n_applied": n_ok, "applied": True,
                            "verify_pass": ok})
        if ok:
            res.fixed = True
            res.final_action = "verify_pass"
            break
        reset_repo_for_variant(cname, instance)
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
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


def run_greedy(instance, cname, theta, label, llm_cfg, costs, max_gen=2, prior=0.5,
               logger=None, run_id=None, kernel=None, online_kernel=None):
    """kernel: dict {p_fix_broken, p_break_correct}; if None, uses DEFAULT_KERNEL.
    online_kernel: OnlineKernelCalibration; if provided, kernel updates after each verify."""
    if kernel is None:
        kernel = DEFAULT_KERNEL
    res = Result(instance_id=instance["instance_id"], variant=f"greedy_{label}")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    belief = prior; gen_left = max_gen
    crit_used: set[str] = set(); step = 0
    has_patch = False
    prev_Y = 0  # SWE-bench seed is buggy by construction
    while step < 10:
        active_kernel = online_kernel.get() if online_kernel is not None else kernel
        Q_bail = 0.0
        Q_verify = (-costs.c_full_test + belief * costs.reward) if has_patch else -math.inf
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
            ok, _ = run_full_test(cname, instance)
            _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0,
                  passed=ok, belief_before=belief)
            res.n_full_tests += 1; res.total_cost += costs.c_full_test
            res.actions.append({"step": step, "action": "verify", "ok": ok, "b": belief})
            y_now = 1 if ok else 0
            if online_kernel is not None and prev_Y is not None:
                online_kernel.update(prev_Y, y_now)
            prev_Y = y_now
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            belief = 0.05
            has_patch = False
            crit_used = set()
            reset_repo_for_variant(cname, instance)
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            t0 = time.perf_counter()
            passed, _ = run_critic(cname, cn, instance)
            _tlog(logger, run_id, iid, cn, time.perf_counter() - t0,
                  passed=passed, belief_before=belief)
            res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used.add(cn)
            res.actions.append({"step": step, "action": action, "passed": passed, "b": belief})
        else:  # generate
            files = get_files_block(cname, instance)
            prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
            t0 = time.perf_counter()
            r = call_llm_or_raise(prompt, llm_cfg)
            _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0,
                  belief_before=belief, model_name=llm_cfg.model)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
            gen_left -= 1
            if applied:
                has_patch = True
                belief = kernel_update(belief, active_kernel)
                crit_used = set()
            else:
                res.n_patch_apply_fails += 1
                reset_repo_for_variant(cname, instance)
                has_patch = False
                crit_used = set()
            res.actions.append({"step": step, "action": "generate", "applied": applied, "b": belief})
        step += 1
    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_dp(instance, cname, theta, label, llm_cfg, costs, make_planner,
           max_gen=2, max_ver=1, prior=0.5,
           logger=None, run_id=None, kernel=None, online_kernel=None):
    """make_planner: callable(critic_likelihoods, transition_kernel) -> DPPlanner (pre-solved).
    kernel/online_kernel: same as run_greedy."""
    if kernel is None:
        kernel = DEFAULT_KERNEL
    res = Result(instance_id=instance["instance_id"], variant=f"dp_{label}")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    belief = prior; gen_left = max_gen; ver_left = max_ver
    crit_used: frozenset[str] = frozenset(); step = 0
    has_patch = False
    prev_Y = 0
    active_kernel = online_kernel.get() if online_kernel is not None else kernel
    planner = make_planner(theta, active_kernel)
    while step < 12:
        action, _q = planner.choose_action(belief, gen_left, crit_used, ver_left)
        if action == "verify" and not has_patch:
            action = "generate:override" if gen_left > 0 else "bail_out"
        if action == "bail_out":
            res.final_action = "bail"; break
        if action == "verify":
            t0 = time.perf_counter()
            ok, _ = run_full_test(cname, instance)
            _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0,
                  passed=ok, belief_before=belief)
            res.n_full_tests += 1; res.total_cost += costs.c_full_test
            ver_left -= 1
            res.actions.append({"step": step, "action": "verify", "ok": ok, "b": belief})
            y_now = 1 if ok else 0
            if online_kernel is not None and prev_Y is not None:
                online_kernel.update(prev_Y, y_now)
                planner = make_planner(theta, online_kernel.get())
            prev_Y = y_now
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            belief = 0.05
            has_patch = False
            crit_used = frozenset()
            reset_repo_for_variant(cname, instance)
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            t0 = time.perf_counter()
            passed, _ = run_critic(cname, cn, instance)
            _tlog(logger, run_id, iid, cn, time.perf_counter() - t0,
                  passed=passed, belief_before=belief)
            res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used = crit_used | frozenset([cn])
            res.actions.append({"step": step, "action": action, "passed": passed, "b": belief})
        elif action.startswith("generate:"):
            files = get_files_block(cname, instance)
            prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
            t0 = time.perf_counter()
            r = call_llm_or_raise(prompt, llm_cfg)
            _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0,
                  belief_before=belief, model_name=llm_cfg.model)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
            gen_left -= 1
            active_kernel = online_kernel.get() if online_kernel is not None else kernel
            if applied:
                has_patch = True
                belief = kernel_update(belief, active_kernel)
                crit_used = frozenset()
            else:
                res.n_patch_apply_fails += 1
                reset_repo_for_variant(cname, instance)
                has_patch = False
                crit_used = frozenset()
            res.actions.append({"step": step, "action": "generate", "applied": applied, "b": belief})
        step += 1
    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


# ---- Ordered critic sequence (L0 → L1 → L2 → L3) ----
SWE_CRITICS_ORDERED = ["critic_syntax", "critic_lint", "critic_early", "critic_mid"]

# Maps threshold variant name → critics to gate on
THRESHOLD_CRITICS = {
    "threshold_L0": ["critic_syntax"],
    "threshold_L2": ["critic_syntax", "critic_early"],
    "threshold_L3": SWE_CRITICS_ORDERED,
}

REFINE_PROMPT_TEMPLATE = """You are a software engineer fixing a bug in a Python repository.

Issue:
{issue}

Files you may need to modify (current contents shown below):
{files_block}

Your previous patch did not pass the following checks:
{feedback}

Please provide a revised patch that addresses these issues.

Produce one or more SEARCH/REPLACE blocks that fix the bug. Each block must be:

```
<<<<<<< SEARCH path/to/file.py
exact lines to find
(must match file contents byte-for-byte including indentation)
=======
exact replacement lines
>>>>>>> REPLACE
```

Return ONLY the SEARCH/REPLACE blocks (no explanation, no markdown fence around the whole thing).
Keep blocks small and targeted; one block per change-site."""

REFLEXION_PROMPT_TEMPLATE = """You are a software engineer fixing a bug in a Python repository.

Issue:
{issue}

Files you may need to modify (current contents shown below):
{files_block}

Your previous patch failed the test suite. Test output:
{test_feedback}

Reflect on what went wrong and provide a corrected patch.

Produce one or more SEARCH/REPLACE blocks that fix the bug. Each block must be:

```
<<<<<<< SEARCH path/to/file.py
exact lines to find
(must match file contents byte-for-byte including indentation)
=======
exact replacement lines
>>>>>>> REPLACE
```

Return ONLY the SEARCH/REPLACE blocks (no explanation, no markdown fence around the whole thing).
Keep blocks small and targeted; one block per change-site."""


def _collect_critic_feedback(cname: str, instance: dict, critics: list[str]) -> str:
    """Run listed critics, return human-readable failure summary."""
    failures = []
    for cn in critics:
        passed, msg = run_critic(cname, cn, instance)
        if not passed:
            failures.append(f"- {cn}: {msg}")
    return "\n".join(failures) if failures else "All checks passed."


def _get_test_feedback(cname: str, instance: dict, timeout: int = 90) -> str:
    """Run first FTP test and return short pytest output for reflection prompt."""
    ftp = get_ftp(instance["instance_id"])[:2]
    if not ftp:
        return "No fail-to-pass tests available."
    args = " ".join(ftp)
    r = _exec(cname, PYTEST_ENV_PREFIX + f"python -m pytest --tb=short -q {args} 2>&1 | tail -40",
              timeout=timeout)
    return ((r.stdout or "") + (r.stderr or ""))[:600].strip() or "No output captured."


# ---- New variant runners ----

def run_best_of_n(instance, cname, llm_cfg, costs, n=3, logger=None, run_id=None):
    """Generate up to n patches independently, verify each, return on first success."""
    res = Result(instance_id=instance["instance_id"], variant="best_of_3")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    for attempt in range(n):
        reset_repo_for_variant(cname, instance)
        files = get_files_block(cname, instance)
        prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
        t0 = time.perf_counter()
        r = call_llm_or_raise(prompt, llm_cfg)
        _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0, model_name=llm_cfg.model)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"step": attempt, "action": "generate", "applied": False})
            continue
        t0 = time.perf_counter()
        ok, _ = run_full_test(cname, instance)
        _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0, passed=ok)
        res.n_full_tests += 1
        res.total_cost += costs.c_full_test
        res.actions.append({"step": attempt, "action": "verify", "applied": True, "ok": ok})
        if ok:
            res.fixed = True
            res.final_action = "verify_pass"
            break
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_threshold(instance, cname, variant_name, critics, llm_cfg, costs,
                  max_gen=2, logger=None, run_id=None):
    """Generate, gate full verification on all threshold critics passing."""
    res = Result(instance_id=instance["instance_id"], variant=variant_name)
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    for attempt in range(max_gen):
        reset_repo_for_variant(cname, instance)
        files = get_files_block(cname, instance)
        prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
        t0 = time.perf_counter()
        r = call_llm_or_raise(prompt, llm_cfg)
        _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0, model_name=llm_cfg.model)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, _, _ = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"step": attempt, "action": "generate", "applied": False})
            continue
        gate = True
        for cn in critics:
            t0 = time.perf_counter()
            passed, _ = run_critic(cname, cn, instance)
            _tlog(logger, run_id, iid, cn, time.perf_counter() - t0, passed=passed)
            res.n_critic_runs += 1
            res.total_cost += costs.c_critic_test
            res.actions.append({"step": attempt, "action": f"critic:{cn}", "passed": passed})
            if not passed:
                gate = False
                break
        if gate:
            t0 = time.perf_counter()
            ok, _ = run_full_test(cname, instance)
            _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0, passed=ok)
            res.n_full_tests += 1
            res.total_cost += costs.c_full_test
            res.actions.append({"step": attempt, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
                break
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_fixed_pipeline(instance, cname, llm_cfg, costs, max_gen=2, logger=None, run_id=None):
    """Run L0→L1→L2→L3 in fixed order before each verify attempt."""
    res = Result(instance_id=instance["instance_id"], variant="fixed_pipeline")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    for attempt in range(max_gen):
        reset_repo_for_variant(cname, instance)
        files = get_files_block(cname, instance)
        prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
        t0 = time.perf_counter()
        r = call_llm_or_raise(prompt, llm_cfg)
        _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0, model_name=llm_cfg.model)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, _, _ = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"step": attempt, "action": "generate", "applied": False})
            continue
        gate = True
        for cn in SWE_CRITICS_ORDERED:
            t0 = time.perf_counter()
            passed, _ = run_critic(cname, cn, instance)
            _tlog(logger, run_id, iid, cn, time.perf_counter() - t0, passed=passed)
            res.n_critic_runs += 1
            res.total_cost += costs.c_critic_test
            res.actions.append({"step": attempt, "action": f"critic:{cn}", "passed": passed})
            if not passed:
                gate = False
                break
        if gate:
            t0 = time.perf_counter()
            ok, _ = run_full_test(cname, instance)
            _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0, passed=ok)
            res.n_full_tests += 1
            res.total_cost += costs.c_full_test
            res.actions.append({"step": attempt, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
                break
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_self_refine(instance, cname, llm_cfg, costs, max_rounds=2, logger=None, run_id=None):
    """Generate → run critics → if fail, refine with feedback → verify."""
    res = Result(instance_id=instance["instance_id"], variant="self_refine")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    current_prompt = PROMPT_TEMPLATE.format(
        issue=issue, files_block=get_files_block(cname, instance))
    for rnd in range(max_rounds + 1):
        reset_repo_for_variant(cname, instance)
        t0 = time.perf_counter()
        r = call_llm_or_raise(current_prompt, llm_cfg)
        _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0, model_name=llm_cfg.model)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, _, _ = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"round": rnd, "action": "generate", "applied": False})
            break
        # Run all critics to decide: verify or refine?
        feedback_parts = []
        for cn in SWE_CRITICS_ORDERED:
            t0 = time.perf_counter()
            passed, msg = run_critic(cname, cn, instance)
            _tlog(logger, run_id, iid, cn, time.perf_counter() - t0, passed=passed)
            res.n_critic_runs += 1
            res.total_cost += costs.c_critic_test
            if not passed:
                feedback_parts.append(f"- {cn}: {msg}")
        res.actions.append({"round": rnd, "action": "critics",
                            "n_failed": len(feedback_parts)})
        if not feedback_parts or rnd == max_rounds:
            # Critics passed (or last round) → verify
            t0 = time.perf_counter()
            ok, _ = run_full_test(cname, instance)
            _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0, passed=ok)
            res.n_full_tests += 1
            res.total_cost += costs.c_full_test
            res.actions.append({"round": rnd, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
            break
        else:
            feedback = "\n".join(feedback_parts)
            current_prompt = REFINE_PROMPT_TEMPLATE.format(
                issue=issue,
                files_block=get_files_block(cname, instance),
                feedback=feedback,
            )
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_reflexion(instance, cname, llm_cfg, costs, max_rounds=2, logger=None, run_id=None):
    """Generate → verify → if fail, reflect on test output → regenerate → verify."""
    res = Result(instance_id=instance["instance_id"], variant="reflexion")
    start = time.perf_counter()
    issue = instance["problem_statement"][:4000]
    iid = instance["instance_id"]
    current_prompt = PROMPT_TEMPLATE.format(
        issue=issue, files_block=get_files_block(cname, instance))
    for rnd in range(max_rounds + 1):
        reset_repo_for_variant(cname, instance)
        t0 = time.perf_counter()
        r = call_llm_or_raise(current_prompt, llm_cfg)
        _tlog(logger, run_id, iid, "generate", time.perf_counter() - t0, model_name=llm_cfg.model)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, _, _ = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"round": rnd, "action": "generate", "applied": False})
            break
        t0 = time.perf_counter()
        ok, _ = run_full_test(cname, instance)
        _tlog(logger, run_id, iid, "verify", time.perf_counter() - t0, passed=ok)
        res.n_full_tests += 1
        res.total_cost += costs.c_full_test
        res.actions.append({"round": rnd, "action": "verify", "ok": ok})
        if ok:
            res.fixed = True
            res.final_action = "verify_pass"
            break
        if rnd < max_rounds:
            test_feedback = _get_test_feedback(cname, instance)
            current_prompt = REFLEXION_PROMPT_TEMPLATE.format(
                issue=issue,
                files_block=get_files_block(cname, instance),
                test_feedback=test_feedback,
            )
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


# ---- Resume-safe helpers ----
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
        "instance_id": r.instance_id, "variant": r.variant,
        "fixed": r.fixed, "total_cost": r.total_cost, "wall_clock": r.wall_clock,
        "n_llm_calls": r.n_llm_calls, "n_critic_runs": r.n_critic_runs,
        "n_full_tests": r.n_full_tests,
        "n_patch_apply_fails": r.n_patch_apply_fails,
        "completion_tokens": r.completion_tokens,
        "final_action": r.final_action, "actions": r.actions,
    }


def main():
    rng = random.Random(SPLIT_SEED)
    all_ids = SWE_INSTANCE_POOL[:]   # 11 small-deps instances
    rng.shuffle(all_ids)
    # Resolve n_train: env var override > computed from TRAIN_FRAC
    env_n_train = os.environ.get("ABBO_N_TRAIN")
    if env_n_train is not None:
        n_train_active = int(env_n_train)
    else:
        n_train_active = int(round(TRAIN_FRAC * len(all_ids)))
    if n_train_active < 1 or n_train_active >= len(all_ids):
        raise SystemExit(
            f"n_train {n_train_active} invalid (must be 1..{len(all_ids)-1}; "
            f"total instances = {len(all_ids)})"
        )
    test_ids = all_ids[n_train_active:]
    # Drop instances that don't exist in the currently-loaded dataset
    # (e.g. Lite-only IDs when running on SWE-bench_Verified).
    from abbo.realworld.agents.swe_bench import list_instance_ids as _present_ids
    _present = set(_present_ids())
    _skipped = [tid for tid in test_ids if tid not in _present]
    if _skipped:
        print(f"Skipping {len(_skipped)} instance(s) not in current dataset: {_skipped}")
    test_ids = [tid for tid in test_ids if tid in _present]
    print(f"Held-out: {len(test_ids)} instances")
    for tid in test_ids:
        print(f"  {tid}")

    state = load_existing(RESULTS_PATH)
    results = state.setdefault("results", {})

    costs = AgentCostConfig()

    # Resolve transition kernel from env var ABBO_KERNEL_MODE
    # ('measured' default; 'online'; 'hardcoded') and ABBO_KERNEL_DIR (optional).
    kernel_mode = os.environ.get("ABBO_KERNEL_MODE", "measured").lower()
    if kernel_mode not in ("measured", "online", "hardcoded"):
        raise SystemExit(f"invalid ABBO_KERNEL_MODE: {kernel_mode!r}")
    kernel_dir_env = os.environ.get("ABBO_KERNEL_DIR", "").strip()
    if kernel_dir_env:
        kernel, kernel_src, online_kernel = resolve_kernel(Path(kernel_dir_env), kernel_mode)
    else:
        # Fallback: try sim_results/transition_kernels.json
        legacy_path = ROOT / "sim_results" / "transition_kernels.json"
        if legacy_path.exists() and kernel_mode != "hardcoded":
            jj = json.loads(legacy_path.read_text())
            sw = jj.get("swebench", {})
            if "p_fix_broken" in sw:
                measured = {"p_fix_broken": float(sw["p_fix_broken"]),
                            "p_break_correct": float(sw["p_break_correct"]
                                                     if sw.get("p_break_correct") is not None
                                                     else jj.get("prior_break_correct", 0.05))}
                kernel = measured
                kernel_src = "measured (sim_results/transition_kernels.json)"
                online_kernel = (OnlineKernelCalibration(init_kernel=measured)
                                  if kernel_mode == "online" else None)
            else:
                kernel = DEFAULT_KERNEL.copy(); kernel_src = "default (no swebench kernel)"
                online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if kernel_mode == "online" else None)
        else:
            kernel = DEFAULT_KERNEL.copy()
            kernel_src = ("hardcoded (forced)" if kernel_mode == "hardcoded"
                          else "default (no kernel file)")
            online_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                              if kernel_mode == "online" else None)
    print(f"Transition kernel: source={kernel_src}, "
          f"p_fix={kernel['p_fix_broken']:.3f}, p_break={kernel['p_break_correct']:.3f}, "
          f"mode={kernel_mode}")

    # Factory used both for initial solve and for online-mode re-solves
    def make_dp(theta_dict, kern):
        p = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                      critic_likelihoods=theta_dict, transition_kernel=kern)
        p.solve()
        return p

    llm_cfg = build_llm_config_from_env(
        default_provider="openrouter",
        default_model=LLM_MODEL,
        default_base_url="https://openrouter.ai/api",
        default_temperature=0.1,
        default_max_tokens=2048,
        default_timeout=180,
    )
    print(f"LLM provider={llm_cfg.provider} model={llm_cfg.model} base_url={llm_cfg.base_url}")

    total = len(test_ids) * len(VARIANTS)
    done = sum(1 for tid in test_ids for v in VARIANTS if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} pairs already done.\n")

    tlog = TelemetryLogger(TELEMETRY_PATH)
    print(f"Telemetry → {TELEMETRY_PATH}")
    started = time.time()
    for i, tid in enumerate(test_ids):
        # Skip if all variants done for this instance
        if all(results.get(f"{tid}|{v}") for v in VARIANTS):
            print(f"\n[{i+1}/{len(test_ids)}] {tid}: all variants done, skipping")
            continue

        elapsed = time.time() - started
        print(f"\n[{i+1}/{len(test_ids)}] {tid}  elapsed={elapsed/60:.1f}min")
        try:
            pull_image(tid, verbose=True)
            cname = start_container(tid)
        except Exception as e:
            print(f"  FAILED to pull/start container: {e}")
            continue

        try:
            instance = get_instance(tid)
            for v in VARIANTS:
                key = f"{tid}|{v}"
                if results.get(key):
                    continue
                # Reset state before each variant
                try:
                    reset_repo_for_variant(cname, instance)
                except Exception as e:
                    print(f"  [{v}] reset failed: {e}")
                    continue

                run_id = f"{_model_slug}__{tid}__{v}"
                try:
                    if v == "simple":
                        r = run_simple(instance, cname, llm_cfg, costs,
                                       logger=tlog, run_id=run_id)
                    elif v == "best_of_3":
                        r = run_best_of_n(instance, cname, llm_cfg, costs, n=3,
                                          logger=tlog, run_id=run_id)
                    elif v in THRESHOLD_CRITICS:
                        r = run_threshold(instance, cname, v, THRESHOLD_CRITICS[v],
                                          llm_cfg, costs, MAX_GENERATORS,
                                          logger=tlog, run_id=run_id)
                    elif v == "fixed_pipeline":
                        r = run_fixed_pipeline(instance, cname, llm_cfg, costs,
                                               MAX_GENERATORS,
                                               logger=tlog, run_id=run_id)
                    elif v == "greedy_hand":
                        r = run_greedy(instance, cname, SWE_CRITIC_LIKELIHOODS, "hand",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR,
                                       logger=tlog, run_id=run_id,
                                       kernel=kernel, online_kernel=online_kernel)
                    elif v == "greedy_fitted":
                        r = run_greedy(instance, cname, FITTED_THETA, "fitted",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR,
                                       logger=tlog, run_id=run_id,
                                       kernel=kernel, online_kernel=online_kernel)
                    elif v == "dp_hand":
                        r = run_dp(instance, cname, SWE_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, make_dp,
                                   MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR,
                                   logger=tlog, run_id=run_id,
                                   kernel=kernel, online_kernel=online_kernel)
                    elif v == "dp_fitted":
                        r = run_dp(instance, cname, FITTED_THETA, "fitted",
                                   llm_cfg, costs, make_dp,
                                   MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR,
                                   logger=tlog, run_id=run_id,
                                   kernel=kernel, online_kernel=online_kernel)
                    elif v == "self_refine":
                        r = run_self_refine(instance, cname, llm_cfg, costs,
                                            logger=tlog, run_id=run_id)
                    elif v == "reflexion":
                        r = run_reflexion(instance, cname, llm_cfg, costs,
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
                      f"toks={r.completion_tokens}  apply_fail={r.n_patch_apply_fails}  "
                      f"wc={r.wall_clock:.1f}s  final={r.final_action}")
                save_progress(RESULTS_PATH, state)
        finally:
            stop_container(cname)

    # Final aggregate
    print("\n=== Final aggregate ===")
    R = 100
    from collections import defaultdict
    by_v = defaultdict(list)
    for rec in results.values():
        by_v[rec["variant"]].append(rec)
    print(f"{'variant':<16} {'n':>3} {'fix%':>6} {'cost':>7} {'Ū_π':>8} {'Δ_π':>8}")
    print('-' * 55)
    if "simple" in by_v and by_v["simple"]:
        baseline = sum((R if r["fixed"] else 0) - r["total_cost"]
                       for r in by_v["simple"]) / len(by_v["simple"])
    else:
        baseline = 0.0
    for v in (*VARIANTS, *[k for k in by_v if k not in VARIANTS]):
        rs = by_v.get(v, [])
        if not rs: continue
        n = len(rs)
        fix = sum(1 for r in rs if r["fixed"]) / n * 100
        c = sum(r["total_cost"] for r in rs) / n
        u = sum((R if r["fixed"] else 0) - r["total_cost"] for r in rs) / n
        d = u - baseline
        print(f"{v:<16} {n:>3} {fix:>5.1f}% {c:>7.2f} {u:>+8.2f} {d:>+8.2f}")

    state["llm_model"] = LLM_MODEL
    state["fitted_theta"] = FITTED_THETA
    state["n_train"] = n_train_active
    state["n_test"] = len(test_ids)
    save_progress(RESULTS_PATH, state)
    print(f"\nSaved: {RESULTS_PATH}")
    tlog.close()


if __name__ == "__main__":
    main()
