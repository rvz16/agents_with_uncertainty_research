#!/usr/bin/env python
"""K-shot active calibration on CodeContests live runs.

Hypothesis: when DP wants to bail, force exactly the first K bails per cell
to (generate, verify). Each forced (Y_t=0, Y_{t+1}) pair updates a Beta
posterior on p_fix_broken (online mode) or is logged-only (offline mode).
After K forced refines, the planner returns to its normal bail behavior.

Question: does online+K-shot beat offline+K-shot for small K? I.e. does a
small amount of guaranteed transition acquisition unlock adaptation?

Cells (CC / gpt5_mini, n=20):
- online K∈{2,5,10,20}  — Beta posterior updates after each forced refine
- offline K=20          — frozen kernel; collects observations for bail-risk UQ

Derivations (no extra runs needed):
- online K=0 ≡ offline K=0 ≡ baseline dp_fitted (no forced refines → no updates)
- offline K∈{0,2,5,10,20}: same offline kernel, just different #forced refines;
  the K<all variants can be computed from the K=all log by truncating to the
  first K forced refines.

Reuses everything from run_codecontests_full.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES,
    get_solution_pool, get_test_cases, get_metadata,
    list_task_ids, run_critic, run_full_test,
)
from abbo.realworld.agents.llm_provider import (
    build_llm_config_from_env, call_llm_or_raise,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig

from run_codecontests_full import (
    Result, PROMPT_TEMPLATE, FITTED_THETA,
    MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR, SPLIT_SEED, N_TRAIN,
    DEFAULT_LLM_MODEL,
    extract_code, format_test_examples, get_test_output,
    load_existing, save_progress, serialize,
)
from experiment_logger import ExperimentLogger


# ----------------------------------------------------------------------
# K-shot state (cell-level)
# ----------------------------------------------------------------------

@dataclass
class KShotState:
    K: int                           # max forced refines per cell
    online: bool                     # True → update Beta posterior; False → frozen
    initial_p_fix: float = 0.3       # starting Beta mean (centered prior)
    prior_ess: float = 4.0           # effective sample size of Beta prior
    alpha_fix: float = 0.0           # filled in __post_init__
    beta_fix: float = 0.0
    n_forced: int = 0                # forced refines done so far in this cell
    obs_log: list = field(default_factory=list)  # bail-risk UQ data
    p_break: float = 0.05            # held fixed throughout

    def __post_init__(self) -> None:
        # Initialize Beta(α,β) so mean = initial_p_fix and α+β = prior_ess
        if self.alpha_fix == 0.0 and self.beta_fix == 0.0:
            self.alpha_fix = self.initial_p_fix * self.prior_ess
            self.beta_fix = (1.0 - self.initial_p_fix) * self.prior_ess

    @property
    def p_fix_mean(self) -> float:
        """Beta mean for the current posterior on p_fix_broken."""
        if self.online:
            return self.alpha_fix / (self.alpha_fix + self.beta_fix)
        return self.initial_p_fix

    def update(self, y_t1: int, instance_id: str, belief_at_bail: float,
               gen_left: int, ver_left: int) -> None:
        """Record one forced refine outcome. Update Beta only if online."""
        self.obs_log.append({
            "instance_id": instance_id,
            "n_forced_so_far": self.n_forced + 1,
            "belief_at_bail": belief_at_bail,
            "y_t1": int(y_t1),
            "gen_left_pre": gen_left,
            "ver_left_pre": ver_left,
            "p_fix_mean_before_update": self.p_fix_mean,
        })
        if self.online:
            if y_t1:
                self.alpha_fix += 1
            else:
                self.beta_fix += 1
        self.n_forced += 1


# ----------------------------------------------------------------------
# Planner cache: avoid re-solving DP every instance when kernel mean is
# stable. Online mode only re-solves when the mean changes by >ε.
# ----------------------------------------------------------------------

_PLANNER_CACHE: dict = {}


def _planner_for(theta_lk, p_fix: float, p_break: float, costs) -> DPPlanner:
    key = (round(p_fix, 3), round(p_break, 3))
    if key not in _PLANNER_CACHE:
        pl = DPPlanner(
            costs, MAX_GENERATORS, MAX_VERIFICATIONS,
            critic_likelihoods=theta_lk,
            transition_kernel={"p_fix_broken": key[0],
                               "p_break_correct": key[1]},
        )
        pl.solve()
        _PLANNER_CACHE[key] = pl
    return _PLANNER_CACHE[key]


# ----------------------------------------------------------------------
# K-shot DP episode runner
# ----------------------------------------------------------------------

def run_dp_kshot(
    task_id: str,
    theta_lk: dict,
    llm_cfg,
    costs: AgentCostConfig,
    kshot_state: KShotState,
    max_gen: int = MAX_GENERATORS,
    max_ver: int = MAX_VERIFICATIONS,
    prior: float = PRIOR,
    logger: ExperimentLogger | None = None,
) -> Result:
    """DP episode with optional forced refine-on-bail (first K bails in cell)."""
    variant = f"kshot_K{kshot_state.K}_{'online' if kshot_state.online else 'offline'}"
    res = Result(task_id=task_id, variant=variant,
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]

    # Build planner with current kernel mean
    planner = _planner_for(
        theta_lk, kshot_state.p_fix_mean, kshot_state.p_break, costs,
    )

    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        belief = prior; gen_left = max_gen; ver_left = max_ver
        crit_used: frozenset[str] = frozenset(); step = 0
        forced_done = False  # we force at most one refine per episode

        while step < 16:
            action, _q = planner.choose_action(
                belief, gen_left, crit_used, ver_left,
            )

            # K-shot intercept: when planner wants to bail and we still have
            # forced-refine budget on this cell, force one (generate, verify).
            # NOTE: we ignore gen_left/ver_left here — the forced refine is a
            # K-shot override of the planner's bail, not a planner-budgeted
            # action. After the forced refine we set forced_done=True so the
            # next bail isn't intercepted again.
            if (action == "bail_out"
                    and not forced_done
                    and kshot_state.n_forced < kshot_state.K):
                belief_at_bail = belief
                gen_left_pre, ver_left_pre = gen_left, ver_left

                # Forced generate
                test_out = get_test_output(wd, task_id)
                prompt = PROMPT_TEMPLATE.format(
                    source_code=current,
                    test_examples=format_test_examples(task_id),
                    test_output=test_out,
                )
                r = call_llm_or_raise(prompt, llm_cfg)
                res.n_llm_calls += 1
                res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left = max(0, gen_left - 1)  # don't go negative
                belief_before_gen = belief
                belief = belief * 0.95 + (1 - belief) * 0.50
                crit_used = frozenset()
                res.actions.append({
                    "step": step, "action": "generate_on_bail", "b": belief,
                    "forced": True,
                })
                if logger:
                    logger.llm_usage(getattr(r, "prompt_tokens", 0),
                                     r.completion_tokens)
                    logger.action(step=step, action="generate_on_bail",
                                  belief_before=belief_before_gen,
                                  belief_after=belief,
                                  reason="K-shot intercept (forced)")
                step += 1

                # Forced verify
                ok, _ = run_full_test(wd, task_id)
                res.n_full_tests += 1
                res.total_cost += costs.c_full_test
                ver_left = max(0, ver_left - 1)  # don't go negative
                res.actions.append({
                    "step": step, "action": "verify_on_bail", "ok": ok,
                    "forced": True,
                })
                if logger:
                    logger.action(step=step, action="verify_on_bail", ok=ok,
                                  belief_before=belief)

                # Log the (Y_t=0, Y_{t+1}) pair and update Beta if online
                alpha_before, beta_before = kshot_state.alpha_fix, kshot_state.beta_fix
                p_fix_before = kshot_state.p_fix_mean
                kshot_state.update(
                    y_t1=int(ok),
                    instance_id=task_id,
                    belief_at_bail=belief_at_bail,
                    gen_left=gen_left_pre,
                    ver_left=ver_left_pre,
                )
                if logger:
                    logger.forced_refine(
                        catch=bool(ok),
                        belief_at_bail=belief_at_bail,
                        alpha_before=alpha_before, beta_before=beta_before,
                        alpha_after=kshot_state.alpha_fix,
                        beta_after=kshot_state.beta_fix,
                        p_fix_before=p_fix_before,
                        p_fix_after=kshot_state.p_fix_mean,
                        n_forced=kshot_state.n_forced, K=kshot_state.K,
                    )

                if ok:
                    res.fixed = True
                    res.final_action = "verify_on_bail_pass"
                    break

                # If online, refresh planner with the updated posterior
                if kshot_state.online:
                    planner = _planner_for(
                        theta_lk, kshot_state.p_fix_mean,
                        kshot_state.p_break, costs,
                    )
                    if logger and abs(kshot_state.p_fix_mean - p_fix_before) > 1e-6:
                        logger.kernel_update(
                            p_fix_before=p_fix_before,
                            p_fix_after=kshot_state.p_fix_mean,
                            delta=kshot_state.p_fix_mean - p_fix_before,
                            alpha=kshot_state.alpha_fix,
                            beta=kshot_state.beta_fix,
                        )

                belief = 0.05  # verify failed → low belief
                forced_done = True
                step += 1
                # Re-enter loop; planner will likely bail again, but now
                # forced_done=True so we accept the bail.
                continue

            if action == "bail_out":
                res.final_action = "bail"
                if logger:
                    logger.action(step=step, action="bail_out",
                                  belief_before=belief)
                break

            if action == "verify":
                belief_before = belief
                ok, _ = run_full_test(wd, task_id)
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                ver_left -= 1
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                if logger:
                    logger.action(step=step, action="verify", ok=ok,
                                  belief_before=belief_before,
                                  belief_after=(1.0 if ok else 0.05))
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05

            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                belief_before = belief
                passed, _ = run_critic(wd, cn, task_id)
                res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta_lk)
                crit_used = crit_used | frozenset([cn])
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
                if logger:
                    lk = theta_lk.get(cn, {})
                    logger.action(step=step, action=action, passed=passed,
                                  belief_before=belief_before,
                                  belief_after=belief,
                                  likelihood_y1=lk.get("p_pass_y1"),
                                  likelihood_y0=lk.get("p_pass_y0"))

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
                belief_before = belief
                belief = belief * 0.95 + (1 - belief) * 0.50
                crit_used = frozenset()
                res.actions.append({"step": step, "action": "generate", "b": belief})
                if logger:
                    logger.llm_usage(getattr(r, "prompt_tokens", 0),
                                     r.completion_tokens)
                    logger.action(step=step, action=action,
                                  belief_before=belief_before,
                                  belief_after=belief)

            step += 1

        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"

    res.wall_clock = time.perf_counter() - start
    return res


# ----------------------------------------------------------------------
# Sweep main()
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="OpenAI-compatible model id.")
    p.add_argument("--results", type=Path, required=True,
                   help="Output JSON path (resume-safe).")
    p.add_argument("--K", type=int, required=True,
                   help="K (max forced refines per cell). Use 999 for 'all'.")
    p.add_argument("--mode", choices=["online", "offline"], required=True,
                   help="online = Beta posterior updates; offline = frozen kernel.")
    p.add_argument("--initial-p-fix", type=float, default=0.3,
                   help="Starting mean for p_fix_broken (default 0.3).")
    p.add_argument("--n-tasks", type=int, default=20,
                   help="Number of test tasks (default 20).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    llm_model = (args.model or "").strip() or DEFAULT_LLM_MODEL

    # Same train/test split as run_codecontests_full.py
    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    test_ids = all_ids[N_TRAIN:N_TRAIN + args.n_tasks]

    state = load_existing(args.results)
    results = state.setdefault("results", {})

    costs = AgentCostConfig()
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

    # Cell-level K-shot state (persists across instances within one cell)
    online = (args.mode == "online")
    K = args.K
    kshot_state = KShotState(K=K, online=online, initial_p_fix=args.initial_p_fix)

    # If resuming, replay logged obs into kshot_state so the cell state is correct
    variant_name = f"kshot_K{K}_{args.mode}"
    for tid in test_ids:
        key = f"{tid}|{variant_name}"
        rec = results.get(key)
        if not rec:
            continue
        acts = rec.get("actions", [])
        for i, a in enumerate(acts):
            if a.get("action") == "verify_on_bail":
                kshot_state.update(
                    y_t1=int(a.get("ok", False)),
                    instance_id=tid,
                    belief_at_bail=acts[i-1].get("b", 0.0) if i > 0 else 0.0,
                    gen_left=0, ver_left=0,
                )

    # Set up ExperimentLogger: events JSONL goes next to the results JSON
    logger = ExperimentLogger(
        name=variant_name, model=llm_cfg.model,
        output_dir=args.results.parent / "events",
        n_total=len(test_ids),
    )
    logger.boot({
        "K": K, "mode": args.mode,
        "initial_p_fix": args.initial_p_fix,
        "n_train": N_TRAIN, "split_seed": SPLIT_SEED,
        "max_generators": MAX_GENERATORS, "max_verifications": MAX_VERIFICATIONS,
        "prior": PRIOR, "costs": str(costs),
        "resume_n_forced": kshot_state.n_forced,
        "resume_alpha": kshot_state.alpha_fix,
        "resume_beta": kshot_state.beta_fix,
        "resume_p_fix_mean": kshot_state.p_fix_mean,
        "results_path": str(args.results),
    })

    for i, tid in enumerate(test_ids):
        key = f"{tid}|{variant_name}"
        if results.get(key):
            continue
        m = get_metadata(tid)
        logger.instance_start(i, tid, meta={
            "cf_rating": m.get("cf_rating"),
            "difficulty": m.get("difficulty"),
            "n_forced_in": kshot_state.n_forced,
            "K": K,
            "p_fix_mean_in": round(kshot_state.p_fix_mean, 3),
        })

        try:
            r = run_dp_kshot(tid, FITTED_THETA, llm_cfg, costs, kshot_state,
                             logger=logger)
        except Exception as e:
            logger.exception(str(e))
            continue

        results[key] = serialize(r)
        logger.instance_done(results[key])

        # Persist progress + kshot_state
        state["kshot_state"] = {
            "K": K, "mode": args.mode,
            "alpha_fix": kshot_state.alpha_fix,
            "beta_fix": kshot_state.beta_fix,
            "n_forced": kshot_state.n_forced,
            "obs_log": kshot_state.obs_log,
            "initial_p_fix": kshot_state.initial_p_fix,
            "p_break": kshot_state.p_break,
        }
        state["llm_model"] = llm_cfg.model
        state["fitted_theta"] = FITTED_THETA
        state["n_test"] = len(test_ids)
        save_progress(args.results, state)

    # Final aggregate via logger
    rs = [r for r in results.values() if r.get("variant") == variant_name]
    n_forced_obs = sum(1 for r in rs for a in r.get("actions", [])
                       if a.get("action") == "verify_on_bail")
    n_fixed_by_forced = sum(1 for r in rs
                            if r.get("final_action") == "verify_on_bail_pass")
    logger.cell_done(extras={
        "variant": variant_name,
        "forced_refines": f"{n_forced_obs} (of which {n_fixed_by_forced} caught)",
        "beta_alpha": f"{kshot_state.alpha_fix:.2f}",
        "beta_beta": f"{kshot_state.beta_fix:.2f}",
        "beta_mean": f"{kshot_state.p_fix_mean:.3f}",
        "results_json": str(args.results),
    })
    logger.close()


if __name__ == "__main__":
    main()
