"""Online-BDP replay over iter trajectories.

The static BDP (in `controller.py:BayesianController`) is solved ONCE per
(generator, benchmark) cell with a fixed measured kernel. This module
gives the counter-factual answer to "what if the BDP planner had been
learning the kernel ONLINE as it consumed each instance's iter trajectory?"

Methodology (per-instance reset variant):

  For each instance's iter trajectory (steps 0..N-1 with Y known at every
  step because the iter script always ran the oracle):

    online_kernel = OnlineKernelCalibration(init_kernel=measured_seed)
    belief = prior
    for step in trajectory:
      belief ← Bayes-update on all observed critic outcomes for this step
      action ← argmax over {verify, give_up, generate} given current
               online_kernel + remaining steps
      if action == verify:    pay c_ver, reward = R · Y[step], DONE
      elif action == give_up: DONE
      elif action == generate:
        pay c_gen
        online_kernel.update(Y[step], Y[step+1])    # the agent observes
                                                    # both Ys (iter data
                                                    # always has them)
        belief ← propagate via online_kernel.get()
        advance to next step

Why per-instance reset rather than cross-instance accumulation:

  Reset matches a single-task agent picture — "drop the agent into a new
  problem with a calibration prior, watch it learn within that problem."
  Cross-instance accumulation is a different research question (does the
  agent get smarter as it sees more problems?) and would need its own
  paper section.

Why "online" is meaningful here when it wasn't on calibration data:

  Iter trajectories observe Y at every step (the iter script always runs
  the oracle to backfill Y). So every step → step+1 boundary provides a
  real (Y_t, Y_{t+1}) transition for the online kernel to learn from.
  Calibration data only observes Y once per verify, so there are no
  in-trajectory transitions to learn from.

The decision-time DP is intentionally NOT the full BayesianController
solve. In iter replay:
  - All critic outcomes at each step are pre-observed (the iter script
    ran them; cost is sunk in `step_cost_usd`).
  - The only choice the agent makes is verify vs. give_up vs. generate.
  - So we collapse the action space to those three and solve a tiny
    horizon-N backward induction in closed form.

If you instead want the full BayesianController behavior (with critic
actions in the planner's lookahead and full belief discretization), use
`analysis.controller.simulate_policy` with `make_bayesian_policy` and
a `BayesianController` rebuilt per step. That version is slower and
methodologically less clean for iter replay (the planner imagines paying
for critics it can already see for free).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

# Allow direct import from the package root (orchestration_hypothesis_testing)
# even when this file is invoked as a notebook helper.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from _common.kernel import OnlineKernelCalibration  # noqa: E402

from analysis.controller import CostModel  # noqa: E402


# ---------------------------------------------------------------------------
# Backward-induction DP over {verify, give_up, generate} only (no critics).
# Iter-replay specific: critics are pre-observed so they don't appear in
# the action space — only the verify-or-continue decision matters.
# ---------------------------------------------------------------------------

def _online_bdp_decide(
    belief: float,
    k_left: int,
    kernel: Mapping[str, float],
    cost: CostModel,
) -> str:
    """Optimal action at this step given current belief and k_left remaining
    steps (counts the current step, so k_left=1 means this is the last step).

    Backward induction:
      V(b, 1)   = max(R·b − c_ver, 0)                                # terminal
      V(b, k>1) = max(R·b − c_ver, 0, −c_gen + V(b', k−1))           # general
      where b' = b · (1 − p_break) + (1 − b) · p_fix.

    Returns one of "verify", "give_up", "generate".
    """
    R = cost.reward
    cv = cost.c_ver
    cg = cost.c_gen
    p_fix = kernel.get("p_fix_broken", kernel.get("P_fix_given_broken"))
    p_break = kernel.get("p_break_correct", kernel.get("P_break_given_correct"))
    if p_fix is None or p_break is None:
        raise ValueError(f"kernel missing p_fix / p_break: {dict(kernel)!r}")

    # Forward belief trajectory under the always-generate path
    beliefs = [belief]
    for _ in range(k_left - 1):
        b = beliefs[-1]
        beliefs.append(b * (1.0 - p_break) + (1.0 - b) * p_fix)

    # Backward value-function pass
    # V[i] = value at step i (i=0 is current step, i=k_left-1 is last step)
    V = [0.0] * k_left
    V[-1] = max(R * beliefs[-1] - cv, 0.0)
    for i in range(k_left - 2, -1, -1):
        Q_verify = R * beliefs[i] - cv
        Q_give_up = 0.0
        Q_generate = -cg + V[i + 1]
        V[i] = max(Q_verify, Q_give_up, Q_generate)

    # Action at step 0 (now)
    Q_verify = R * belief - cv
    Q_give_up = 0.0
    if k_left > 1:
        Q_generate = -cg + V[1]
    else:
        # Last step — generate would land in "the void", no future value
        Q_generate = float("-inf")

    actions = {"verify": Q_verify, "give_up": Q_give_up, "generate": Q_generate}
    return max(actions, key=actions.get)


# ---------------------------------------------------------------------------
# Per-instance online-BDP simulator
# ---------------------------------------------------------------------------

_CRITIC_FIELDS = ("L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review")


def _bayes_update_belief(
    belief: float,
    critic_likes: Mapping[str, Mapping[str, float]],
    rec: Mapping[str, object],
) -> float:
    """Bayes-update belief on ALL critic outcomes available in `rec`.

    Skips critics that are missing from `critic_likes` (e.g. L2 on SWE-bench
    Lite, where public-tests don't exist) or that have None likelihoods.
    Matches BayesianController._bayes_update arithmetic so the iter-replay
    belief is comparable to the static-BDP belief.
    """
    b = belief
    for cname in _CRITIC_FIELDS:
        if cname not in critic_likes or cname not in rec:
            continue
        l = critic_likes[cname]
        p1 = l.get("P_pass_given_Y1")
        p0 = l.get("P_pass_given_Y0")
        if p1 is None or p0 is None:
            continue
        obs = bool(rec[cname])
        if obs:
            num = p1 * b
            den = num + p0 * (1.0 - b)
        else:
            num = (1.0 - p1) * b
            den = num + (1.0 - p0) * (1.0 - b)
        b = num / max(den, 1e-12)
    return b


def simulate_online_bdp_on_iter(
    iter_traj: Sequence[Mapping[str, object]],
    likes: Mapping[str, object],
    prior: float,
    kernel_seed: Mapping[str, float],
    cost: CostModel,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict:
    """Replay one instance's iter trajectory under a per-step online-BDP planner.

    Parameters
    ----------
    iter_traj : sorted-by-step list of step records. Each record should
        carry `Y` (0/1), and any subset of L0_syntax, L1_lint,
        L2_public_tests, L3_llm_review. Records with `Y` not in (0, 1) are
        tolerated mid-trajectory (the kernel update for that boundary is
        skipped); at the terminal-verify step a missing Y is recorded as
        a `verify_but_y_missing` give-up.

    likes : likelihood tables in the canonical schema:
            {"critic_likelihoods": {cname: {"P_pass_given_Y1", "P_pass_given_Y0"}}}

    prior : P(Y=1) prior used at step 0 before any critic updates.

    kernel_seed : starting kernel for the online estimator. Accepts both
        lowercase ({"p_fix_broken", "p_break_correct"}) and uppercase
        ({"P_fix_given_broken", "P_break_given_correct"}) schemas.

    cost : CostModel with reward, c_ver, c_gen.

    alpha, beta : Beta prior for the online estimator (defaults: Laplace).

    Returns
    -------
    dict with keys:
        utility, reward, cost — paired with the static BDP simulator
        stop_step, stop_reason — when/why the trajectory ended
        verified — bool
        n_kernel_updates — how many (Y_t, Y_{t+1}) pairs were absorbed
        final_kernel — {"p_fix_broken", "p_break_correct"} at end of run
        belief_trajectory — list of beliefs at each decision point (for
                            diagnostics)
    """
    ok = OnlineKernelCalibration(init_kernel=dict(kernel_seed),
                                  alpha=alpha, beta=beta)
    critic_likes = likes.get("critic_likelihoods", {}) if isinstance(likes, Mapping) else {}

    cum_cost = 0.0
    reward = 0.0
    belief = float(prior)
    stop_step = None
    stop_reason = None
    verified = False
    given_up = False
    n_kernel_updates = 0
    belief_trajectory = []

    if not iter_traj:
        return {
            "utility": 0.0, "reward": 0.0, "cost": 0.0,
            "stop_step": None, "stop_reason": "empty_trajectory",
            "verified": False, "n_kernel_updates": 0,
            "final_kernel": ok.get(), "belief_trajectory": [],
        }

    for step_idx, rec in enumerate(iter_traj):
        # 1. Bayes-update belief on this step's pre-observed critic outcomes
        belief = _bayes_update_belief(belief, critic_likes, rec)
        belief_trajectory.append(belief)

        # 2. Decide using online-BDP (verify / give_up / generate)
        k_left = len(iter_traj) - step_idx
        action = _online_bdp_decide(belief, k_left, ok.get(), cost)

        if action == "verify":
            cum_cost += cost.c_ver
            y = rec.get("Y")
            if y in (0, 1):
                reward = cost.reward * int(y)
                verified = True
                stop_reason = "verify"
            else:
                # Y not available — treat as give_up but still pay c_ver
                # (the agent paid to verify; the harness just didn't return Y)
                given_up = True
                stop_reason = "verify_but_y_missing"
            stop_step = step_idx
            break
        elif action == "give_up":
            given_up = True
            stop_reason = "give_up"
            stop_step = step_idx
            break
        elif action == "generate":
            cum_cost += cost.c_gen
            # Online kernel update: in iter replay the agent observes Y at
            # every step (the iter script always backfills Y), so each
            # step→step+1 boundary yields a (Y_t, Y_{t+1}) transition.
            yt = rec.get("Y")
            if step_idx + 1 < len(iter_traj):
                yt1 = iter_traj[step_idx + 1].get("Y")
                if yt in (0, 1) and yt1 in (0, 1):
                    ok.update(int(yt), int(yt1))
                    n_kernel_updates += 1
            # Propagate belief via the (just-updated) online kernel
            k = ok.get()
            belief = (belief * (1.0 - k["p_break_correct"])
                       + (1.0 - belief) * k["p_fix_broken"])
        else:
            raise ValueError(f"Unknown action from _online_bdp_decide: {action!r}")

    if not verified and not given_up:
        # Trajectory ended without a terminal decision (shouldn't happen
        # because k_left=1 forces verify-or-give-up, but defensive).
        stop_reason = "trajectory_end"
        stop_step = len(iter_traj) - 1

    return {
        "utility": float(reward - cum_cost),
        "reward": float(reward),
        "cost": float(cum_cost),
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        "verified": verified,
        "n_kernel_updates": n_kernel_updates,
        "final_kernel": ok.get(),
        "belief_trajectory": belief_trajectory,
    }


__all__ = [
    "simulate_online_bdp_on_iter",
    "_online_bdp_decide",   # exported for testing
    "_bayes_update_belief",  # exported for testing
]
