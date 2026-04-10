#!/usr/bin/env python3
"""Multi-critic Bayesian controller.

Extends the single-critic Bellman controller by making the set of "critics
already run on the current patch" part of the state. This lets the solver
plan sequences like "run L3, and if it passes, run L2 on the same patch to
combine independent evidence before verifying".

State: (belief, used_critics_mask, step)
  - belief: continuous, discretized on a 1001-point grid
  - used_critics_mask: bitmask over the available critics
  - step: integer in [0, horizon)

Transitions:
  - GENERATE: belief -> transition_kernel(belief); mask -> 0; step += 1
  - VERIFY:   terminal, reward = b*reward - c_ver
  - CRITIC_k: belief -> Bayes update; mask |= 1<<k; step += 1
              (only allowed if bit k is not already set in mask)

The action space grows as 2 + n_critics, but the effective per-state action
set is 2 + (n_critics - popcount(mask)). Memory: O(grid × 2^k × horizon)
which is ~32k states per step for k=5 — well within budget.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from controller.bayesian_controller import (
    Action,
    CostModel,
    CriticLikelihood,
    TransitionKernel,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiCriticDecision:
    """Result of a policy query."""
    action: Optional[Action]
    expected_value: float


class MultiCriticBayesianController:
    """Bayesian controller with per-patch critic exhaustion in the state.

    Unlike the single-critic controller, this class does NOT precompute a
    single policy table on belief alone. Instead it solves the full Bellman
    on (belief, mask, step) and exposes ``select_action(belief, mask, step)``.

    The simulation is expected to track ``used_critics_mask`` per patch,
    resetting it to 0 on a generate step. See ``run_multi_critic_policy`` in
    ``evaluation/run_simulation.py`` (added alongside this controller).
    """

    # Fixed ordering so the mask bit positions are stable.
    CRITIC_ORDER = [
        "L0_syntax",
        "L1_lint",
        "L2_fast_test",
        "L3_llm_review",
        "L4_mypy",
    ]

    def __init__(
        self,
        critic_likelihoods: dict[str, CriticLikelihood],
        transition: TransitionKernel,
        costs: CostModel = CostModel(),
        horizon: int = 10,
        grid_size: int = 500,
    ) -> None:
        self.critics = critic_likelihoods
        self.transition = transition
        self.costs = costs
        self.horizon = horizon
        self.grid_size = grid_size
        self.grid = np.linspace(0, 1, grid_size + 1)

        self._active_critics: list[str] = [
            c for c in self.CRITIC_ORDER if c in critic_likelihoods
        ]
        self._critic_to_bit = {c: i for i, c in enumerate(self._active_critics)}
        self._n_masks = 1 << len(self._active_critics)

        self._critic_cost = {
            "L0_syntax": costs.c_crit_l0,
            "L1_lint": costs.c_crit_l1,
            "L2_fast_test": costs.c_crit_l2,
            "L3_llm_review": costs.c_crit_l3,
            "L4_mypy": costs.c_crit_l4,
        }
        self._critic_action = {
            "L0_syntax": Action.CRITIC_L0,
            "L1_lint": Action.CRITIC_L1,
            "L2_fast_test": Action.CRITIC_L2,
            "L3_llm_review": Action.CRITIC_L3,
            "L4_mypy": Action.CRITIC_L4,
        }

        self._value, self._policy = self._solve_bellman()

    @classmethod
    def from_likelihood_tables(
        cls,
        path: str | Path,
        costs: Optional[CostModel] = None,
        horizon: int = 10,
        grid_size: int = 500,
        iid_kernel: bool = False,
    ) -> "MultiCriticBayesianController":
        with open(path) as f:
            data = json.load(f)
        critics = {
            level: CriticLikelihood(
                p_pass_given_correct=lk["p_pass_given_correct"],
                p_pass_given_incorrect=lk["p_pass_given_incorrect"],
            )
            for level, lk in data["critic_likelihoods"].items()
        }
        if iid_kernel:
            counts = data.get("sample_counts", {})
            total = counts.get("total_patches", 0)
            correct = counts.get("correct", 0)
            if total <= 0:
                raise ValueError("iid_kernel requires sample_counts.total_patches>0")
            base_rate = correct / total
            transition = TransitionKernel(p_fix=base_rate, p_break=1.0 - base_rate)
        else:
            transition = TransitionKernel(
                p_fix=data["generator_transition"]["p_fix_given_broken"],
                p_break=data["generator_transition"]["p_break_given_correct"],
            )
        return cls(
            critic_likelihoods=critics,
            transition=transition,
            costs=costs or CostModel(),
            horizon=horizon,
            grid_size=grid_size,
        )

    def _belief_index(self, b: float) -> int:
        return int(round(float(np.clip(b, 0.0, 1.0)) * self.grid_size))

    def _q_verify(self, b: float) -> float:
        return b * self.costs.reward - self.costs.c_ver

    def _belief_after_generate(self, b: float) -> float:
        return b * (1 - self.transition.p_break) + (1 - b) * self.transition.p_fix

    def _solve_bellman(self) -> tuple[np.ndarray, np.ndarray]:
        """Backward induction over (step, mask, belief).

        Value/policy shape: (horizon+1, n_masks, grid_size+1).
        Policy encoding:
          -2 = verify
          -1 = give up
           0 = generate
           1+i = critic bit i (from CRITIC_ORDER among active critics)
        """
        n_b = self.grid_size + 1
        value = np.zeros((self.horizon + 1, self._n_masks, n_b), dtype=np.float64)
        policy = np.full(
            (self.horizon, self._n_masks, n_b), fill_value=-1, dtype=np.int8
        )

        # Terminal: V_T(b, mask) = max(Q_ver(b), 0)
        q_ver = np.maximum(self.grid * self.costs.reward - self.costs.c_ver, 0.0)
        for m in range(self._n_masks):
            value[self.horizon, m, :] = q_ver

        # Precompute critic Bayes updates and p_pass as vectors over belief
        critic_updates = {}
        for name, lk in self.critics.items():
            b_arr = self.grid
            p_pass = b_arr * lk.p_pass_given_correct + (1 - b_arr) * lk.p_pass_given_incorrect
            p_fail = 1 - p_pass
            with np.errstate(divide="ignore", invalid="ignore"):
                b_pass = np.where(
                    p_pass > 1e-12,
                    (b_arr * lk.p_pass_given_correct) / np.where(p_pass > 1e-12, p_pass, 1.0),
                    b_arr,
                )
                b_fail = np.where(
                    p_fail > 1e-12,
                    (b_arr * (1 - lk.p_pass_given_correct)) / np.where(p_fail > 1e-12, p_fail, 1.0),
                    b_arr,
                )
            b_pass = np.clip(b_pass, 0.0, 1.0)
            b_fail = np.clip(b_fail, 0.0, 1.0)
            idx_pass = np.rint(b_pass * self.grid_size).astype(np.int64)
            idx_fail = np.rint(b_fail * self.grid_size).astype(np.int64)
            critic_updates[name] = {
                "p_pass": p_pass,
                "p_fail": p_fail,
                "idx_pass": idx_pass,
                "idx_fail": idx_fail,
                "cost": self._critic_cost[name],
            }

        # Generate belief update (vectorized).
        gen_b = self.grid * (1 - self.transition.p_break) + (1 - self.grid) * self.transition.p_fix
        gen_b = np.clip(gen_b, 0.0, 1.0)
        gen_idx = np.rint(gen_b * self.grid_size).astype(np.int64)
        gen_cost = self.costs.c_gen

        # Backward induction
        for t in range(self.horizon - 1, -1, -1):
            future = value[t + 1]  # shape (n_masks, n_b)
            for mask in range(self._n_masks):
                # Q_ver(b) is independent of mask and step (terminal)
                q_ver_arr = self.grid * self.costs.reward - self.costs.c_ver

                # Q_gen(b) uses future[0] because generate resets the mask
                q_gen_arr = -gen_cost + future[0][gen_idx]

                # Stack candidate action values: [verify, generate, (critics...)]
                cand_values = [q_ver_arr, q_gen_arr]
                cand_codes = [-2, 0]  # verify=-2, generate=0

                for name in self._active_critics:
                    bit = self._critic_to_bit[name]
                    if mask & (1 << bit):
                        continue  # already used -> not available
                    u = critic_updates[name]
                    new_mask = mask | (1 << bit)
                    v_pass = future[new_mask][u["idx_pass"]]
                    v_fail = future[new_mask][u["idx_fail"]]
                    q_crit = -u["cost"] + u["p_pass"] * v_pass + u["p_fail"] * v_fail
                    cand_values.append(q_crit)
                    cand_codes.append(1 + bit)

                # Also consider "give up" = 0.0
                cand_values.append(np.zeros_like(q_ver_arr))
                cand_codes.append(-1)

                stacked = np.stack(cand_values, axis=0)  # (n_actions, n_b)
                argmax = np.argmax(stacked, axis=0)
                best_values = stacked[argmax, np.arange(n_b)]
                value[t, mask, :] = best_values
                policy[t, mask, :] = np.array(cand_codes, dtype=np.int8)[argmax]

        return value, policy

    def select_action(
        self,
        belief: float,
        used_critics_mask: int,
        step: int,
    ) -> Optional[Action]:
        idx = self._belief_index(belief)
        step = min(step, self.horizon - 1)
        code = int(self._policy[step, used_critics_mask, idx])
        if code == -1:
            return None  # give up
        if code == -2:
            return Action.VERIFY
        if code == 0:
            return Action.GENERATE
        bit = code - 1
        level = self._active_critics[bit]
        return self._critic_action[level]

    def get_value(self, belief: float, used_critics_mask: int, step: int) -> float:
        idx = self._belief_index(belief)
        step = min(step, self.horizon)
        return float(self._value[step, used_critics_mask, idx])

    def update_belief(self, belief: float, critic_level: str, passed: bool) -> float:
        critic = self.critics[critic_level]
        if passed:
            pz1 = critic.p_pass_given_correct
            pz0 = critic.p_pass_given_incorrect
        else:
            pz1 = 1 - critic.p_pass_given_correct
            pz0 = 1 - critic.p_pass_given_incorrect
        denom = belief * pz1 + (1 - belief) * pz0
        if denom < 1e-12:
            return belief
        return (belief * pz1) / denom

    def update_belief_after_generation(self, belief: float) -> float:
        return (
            belief * (1 - self.transition.p_break)
            + (1 - belief) * self.transition.p_fix
        )

    def bit_for(self, level: str) -> int:
        return self._critic_to_bit[level]

    def active_critics(self) -> list[str]:
        return list(self._active_critics)

    def print_policy_at_mask(self, mask: int = 0, step: int = 0) -> None:
        """Print policy at step=0, mask=0 in human-readable regions."""
        decode = {
            -2: "verify",
            -1: "give_up",
            0: "generate",
        }
        for name in self._active_critics:
            decode[1 + self._critic_to_bit[name]] = f"critic_{name}"
        row = self._policy[step, mask]
        print(f"\nMulti-critic policy [step={step}, mask={mask:0{len(self._active_critics)}b}]:")
        start = 0.0
        prev = None
        for i, b in enumerate(self.grid):
            act = decode[int(row[i])]
            if act != prev:
                if prev is not None:
                    print(f"  b in [{start:.3f}, {b:.3f}): {prev}")
                prev = act
                start = b
        if prev is not None:
            print(f"  b in [{start:.3f}, 1.000]: {prev}")
