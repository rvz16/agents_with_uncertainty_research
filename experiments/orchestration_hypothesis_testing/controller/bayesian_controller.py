#!/usr/bin/env python3
"""Bayesian Code Generation Controller.

Implements the Bellman equation from Section 3.3 of the paper:
"Agentic AI Orchestration as Sequential Hypothesis Testing for Code Generation"

The controller maintains a belief state b_t = P(patch correct | observations)
and selects between three actions:
  - Generator (a_gen): refine the current patch via LLM
  - Critic (a_crit): run a cheap diagnostic to update belief
  - Verifier (a_ver): run full test suite (expensive, reveals ground truth)

The optimal policy is computed via backward induction on a discretized belief
space, solving:
    V(b) = max{ Q_gen(b), Q_crit(b), Q_ver(b) }

Usage:
    from controller.bayesian_controller import BayesianController

    ctrl = BayesianController.from_likelihood_tables("calibration/data/likelihood_tables.json")
    action = ctrl.select_action(belief=0.3)
    new_belief = ctrl.update_belief(belief=0.3, critic_level="L1_lint", passed=True)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class Action(Enum):
    GENERATE = "generate"
    VERIFY = "verify"
    CRITIC_L0 = "critic_L0"
    CRITIC_L1 = "critic_L1"
    CRITIC_L2 = "critic_L2"
    CRITIC_L3 = "critic_L3"
    CRITIC_L4 = "critic_L4"


@dataclass(frozen=True)
class CostModel:
    """Cost parameters for the POMDP.

    Costs are in abstract units. The reward lambda for successful verification
    must be large enough to justify the cumulative costs.
    """
    c_gen: float = 5.0       # Cost of one LLM refinement call
    c_ver: float = 20.0      # Cost of full test suite execution
    c_crit_l0: float = 0.1   # Cost of syntax check
    c_crit_l1: float = 1.0   # Cost of lint
    c_crit_l2: float = 5.0   # Cost of fast test
    c_crit_l3: float = 2.0   # Cost of LLM review (cheap model, ~2s + $0.001)
    c_crit_l4: float = 3.0   # Cost of mypy type check (~2-5s)
    reward: float = 200.0    # Reward for successful verification


@dataclass(frozen=True)
class CriticLikelihood:
    """P(pass | Y) for a single critic level."""
    p_pass_given_correct: float   # P(pass | Y=1)
    p_pass_given_incorrect: float  # P(pass | Y=0)

    @property
    def informativeness(self) -> float:
        """Gap between TPR and FPR. Higher = more informative."""
        return self.p_pass_given_correct - self.p_pass_given_incorrect


@dataclass(frozen=True)
class TransitionKernel:
    """Generator transition probabilities."""
    p_fix: float   # P(Y'=1 | Y=0, a_gen) — probability of fixing a broken patch
    p_break: float  # P(Y'=0 | Y=1, a_gen) — probability of breaking a correct patch


class BayesianController:
    """Bayesian decision-theoretic controller for code generation.

    Solves the Bellman equation on a discretized belief space to compute
    the optimal policy: which action to take at each belief level.
    """

    def __init__(
        self,
        critic_likelihoods: dict[str, CriticLikelihood],
        transition: TransitionKernel,
        costs: CostModel = CostModel(),
        horizon: int = 10,
        grid_size: int = 1000,
    ) -> None:
        self.critics = critic_likelihoods
        self.transition = transition
        self.costs = costs
        self.horizon = horizon
        self.grid_size = grid_size

        # Discretized belief grid
        self.grid = np.linspace(0, 1, grid_size + 1)

        # Solve the Bellman equation
        self._value_table, self._policy_table = self._solve_bellman()

    @classmethod
    def from_likelihood_tables(
        cls,
        path: str | Path,
        costs: Optional[CostModel] = None,
        horizon: int = 10,
        grid_size: int = 1000,
    ) -> "BayesianController":
        """Load likelihood tables from compute_likelihoods.py output."""
        with open(path) as f:
            data = json.load(f)

        critics = {}
        for level, lk in data["critic_likelihoods"].items():
            critics[level] = CriticLikelihood(
                p_pass_given_correct=lk["p_pass_given_correct"],
                p_pass_given_incorrect=lk["p_pass_given_incorrect"],
            )

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
        """Map continuous belief to nearest grid index."""
        return int(round(b * self.grid_size))

    def _q_verify(self, b: float) -> float:
        """Q-value for verification: Q_ver(b) = b * lambda - C_ver."""
        return b * self.costs.reward - self.costs.c_ver

    def _q_generate(self, b: float, value_fn: np.ndarray) -> float:
        """Q-value for generation.

        After generation, belief transitions via the transition kernel:
        b' = [b * (1 - p_break) + (1 - b) * p_fix]
        (simplified: the new belief is the expected correctness after transition)
        """
        b_new = (
            b * (1 - self.transition.p_break)
            + (1 - b) * self.transition.p_fix
        )
        b_new = np.clip(b_new, 0, 1)
        idx = self._belief_index(b_new)
        return -self.costs.c_gen + value_fn[idx]

    def _q_critic(
        self,
        b: float,
        critic: CriticLikelihood,
        cost: float,
        value_fn: np.ndarray,
    ) -> float:
        """Q-value for running a critic.

        The critic returns pass/fail. We compute expected V over both outcomes:
        Q_crit(b) = -C_crit + P(pass|b) * V(b_pass) + P(fail|b) * V(b_fail)

        where b_pass and b_fail are Bayes updates.
        """
        p_pass = b * critic.p_pass_given_correct + (1 - b) * critic.p_pass_given_incorrect
        p_fail = 1 - p_pass

        # Bayes update for pass
        if p_pass > 1e-12:
            b_pass = (b * critic.p_pass_given_correct) / p_pass
        else:
            b_pass = b

        # Bayes update for fail
        if p_fail > 1e-12:
            b_fail = (b * (1 - critic.p_pass_given_correct)) / p_fail
        else:
            b_fail = b

        b_pass = np.clip(b_pass, 0, 1)
        b_fail = np.clip(b_fail, 0, 1)

        idx_pass = self._belief_index(b_pass)
        idx_fail = self._belief_index(b_fail)

        return -cost + p_pass * value_fn[idx_pass] + p_fail * value_fn[idx_fail]

    def _solve_bellman(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve the Bellman equation via backward induction.

        Returns:
            value_table: shape (horizon+1, grid_size+1) — V_t(b) for each step
            policy_table: shape (horizon, grid_size+1) — optimal action index at each step
        """
        n = self.grid_size + 1
        value_table = np.zeros((self.horizon + 1, n))
        policy_table = np.zeros((self.horizon, n), dtype=int)

        # Terminal values: V_T(b) = max(Q_ver(b), 0)
        # At the end, we must either verify or give up
        for i, b in enumerate(self.grid):
            value_table[self.horizon, i] = max(self._q_verify(b), 0)

        # Map action names to indices for the policy table
        action_list = [
            Action.GENERATE,
            Action.VERIFY,
        ]
        critic_costs = {
            "L0_syntax": self.costs.c_crit_l0,
            "L1_lint": self.costs.c_crit_l1,
            "L2_fast_test": self.costs.c_crit_l2,
            "L3_llm_review": self.costs.c_crit_l3,
            "L4_mypy": self.costs.c_crit_l4,
        }
        critic_actions = {
            "L0_syntax": Action.CRITIC_L0,
            "L1_lint": Action.CRITIC_L1,
            "L2_fast_test": Action.CRITIC_L2,
            "L3_llm_review": Action.CRITIC_L3,
            "L4_mypy": Action.CRITIC_L4,
        }
        for level in self.critics:
            if level in critic_actions:
                action_list.append(critic_actions[level])

        self._action_list = action_list

        # Backward induction
        for t in range(self.horizon - 1, -1, -1):
            future_v = value_table[t + 1]

            for i, b in enumerate(self.grid):
                q_values = []

                # Q_gen
                q_values.append(self._q_generate(b, future_v))

                # Q_ver
                q_values.append(self._q_verify(b))

                # Q_crit for each critic level
                for level, critic in self.critics.items():
                    cost = critic_costs.get(level, 1.0)
                    q_values.append(self._q_critic(b, critic, cost, future_v))

                # Also consider "give up" (utility = 0)
                q_values.append(0.0)

                best = int(np.argmax(q_values))
                if best == len(q_values) - 1:
                    # "Give up" maps to no action — use verify with 0 expected value
                    value_table[t, i] = 0.0
                    policy_table[t, i] = -1  # give up
                else:
                    value_table[t, i] = q_values[best]
                    policy_table[t, i] = best

        return value_table, policy_table

    def select_action(self, belief: float, step: int = 0) -> Optional[Action]:
        """Select the optimal action at the given belief and step.

        Returns None if the optimal action is to give up.
        """
        idx = self._belief_index(belief)
        action_idx = self._policy_table[min(step, self.horizon - 1), idx]

        if action_idx == -1:
            return None  # give up

        return self._action_list[action_idx]

    def get_value(self, belief: float, step: int = 0) -> float:
        """Get the value function V(b) at the given belief and step."""
        idx = self._belief_index(belief)
        return float(self._value_table[min(step, self.horizon), idx])

    def update_belief(
        self,
        belief: float,
        critic_level: str,
        passed: bool,
    ) -> float:
        """Update belief after observing a critic outcome.

        Uses Bayes' rule:
            b' = b * P(z|Y=1) / [b * P(z|Y=1) + (1-b) * P(z|Y=0)]
        """
        critic = self.critics.get(critic_level)
        if critic is None:
            log.warning("Unknown critic level: %s", critic_level)
            return belief

        if passed:
            p_z_correct = critic.p_pass_given_correct
            p_z_incorrect = critic.p_pass_given_incorrect
        else:
            p_z_correct = 1 - critic.p_pass_given_correct
            p_z_incorrect = 1 - critic.p_pass_given_incorrect

        denominator = belief * p_z_correct + (1 - belief) * p_z_incorrect
        if denominator < 1e-12:
            return belief

        return (belief * p_z_correct) / denominator

    def update_belief_after_generation(self, belief: float) -> float:
        """Update belief after a generation step.

        b' = b * (1 - p_break) + (1 - b) * p_fix
        """
        return (
            belief * (1 - self.transition.p_break)
            + (1 - belief) * self.transition.p_fix
        )

    def get_policy_summary(self) -> dict[str, list[float]]:
        """Get belief thresholds where optimal action changes.

        Returns a dict mapping action names to [lower_bound, upper_bound]
        belief ranges (at step 0).
        """
        regions: dict[str, list[float]] = {}
        policy_row = self._policy_table[0]

        current_action = None
        current_start = 0.0

        for i, b in enumerate(self.grid):
            action_idx = policy_row[i]
            if action_idx == -1:
                action_name = "give_up"
            else:
                action_name = self._action_list[action_idx].value

            if action_name != current_action:
                if current_action is not None:
                    if current_action not in regions:
                        regions[current_action] = []
                    regions[current_action].append(current_start)
                    regions[current_action].append(b)
                current_action = action_name
                current_start = b

        # Final region
        if current_action is not None:
            if current_action not in regions:
                regions[current_action] = []
            regions[current_action].append(current_start)
            regions[current_action].append(1.0)

        return regions

    def print_policy(self) -> None:
        """Print a human-readable summary of the policy at step 0."""
        print("\n" + "=" * 60)
        print("BAYESIAN CONTROLLER POLICY (step 0)")
        print("=" * 60)
        print(f"Horizon: {self.horizon}, Grid: {self.grid_size}")
        print(f"Costs: C_gen={self.costs.c_gen}, C_ver={self.costs.c_ver}, "
              f"C_crit=[{self.costs.c_crit_l0}, {self.costs.c_crit_l1}, {self.costs.c_crit_l2}]")
        print(f"Reward: {self.costs.reward}")
        print(f"Transition: P(fix)={self.transition.p_fix:.4f}, P(break)={self.transition.p_break:.4f}")

        print("\nCritic likelihoods:")
        for level, critic in self.critics.items():
            print(f"  {level}: P(pass|Y=1)={critic.p_pass_given_correct:.4f}, "
                  f"P(pass|Y=0)={critic.p_pass_given_incorrect:.4f}, "
                  f"gap={critic.informativeness:.4f}")

        print("\nOptimal action by belief region:")
        policy_row = self._policy_table[0]
        prev_action = None
        start_b = 0.0

        for i, b in enumerate(self.grid):
            action_idx = policy_row[i]
            action_name = "give_up" if action_idx == -1 else self._action_list[action_idx].value

            if action_name != prev_action:
                if prev_action is not None:
                    print(f"  b in [{start_b:.3f}, {b:.3f}): {prev_action}")
                prev_action = action_name
                start_b = b

        if prev_action is not None:
            print(f"  b in [{start_b:.3f}, 1.000]: {prev_action}")

        print("=" * 60)


# ============================================================================
# Heuristic baselines for comparison
# ============================================================================

class FixedPipelineBaseline:
    """Fixed pipeline: generate -> lint -> if fail, regenerate -> verify.

    No belief tracking. Always runs the same sequence.
    """

    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts

    def get_action_sequence(self) -> list[Action]:
        """Return the fixed action sequence for one episode."""
        actions = []
        for _ in range(self.max_attempts):
            actions.append(Action.GENERATE)
            actions.append(Action.CRITIC_L1)  # lint check
        actions.append(Action.VERIFY)
        return actions


class ConfidenceThresholdBaseline:
    """Confidence threshold: verify if cheapest critic passes, else regenerate.

    Uses a single threshold tau on the critic outcome.
    """

    def __init__(self, tau: float = 0.5, critic_level: str = "L1_lint") -> None:
        self.tau = tau
        self.critic_level = critic_level

    def select_action(self, critic_passed: Optional[bool] = None) -> Action:
        if critic_passed is None:
            return Action.CRITIC_L1
        if critic_passed:
            return Action.VERIFY
        return Action.GENERATE


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        table_path = sys.argv[1]
    else:
        table_path = str(
            Path(__file__).resolve().parent.parent
            / "calibration" / "data" / "likelihood_tables.json"
        )

    print(f"Loading likelihood tables from: {table_path}")
    controller = BayesianController.from_likelihood_tables(table_path)
    controller.print_policy()

    # Demo: trace a belief trajectory
    print("\n--- Demo: belief trajectory ---")
    b = 0.2  # initial belief
    for step in range(5):
        action = controller.select_action(b, step)
        print(f"  Step {step}: b={b:.4f} -> action={action.value if action else 'give_up'}")

        if action == Action.VERIFY:
            print(f"  -> Verify! Expected utility = {b * controller.costs.reward - controller.costs.c_ver:.1f}")
            break
        elif action and action.value.startswith("critic"):
            # Simulate critic pass
            level = {Action.CRITIC_L0: "L0_syntax", Action.CRITIC_L1: "L1_lint", Action.CRITIC_L2: "L2_fast_test"}[action]
            b = controller.update_belief(b, level, passed=True)
            print(f"     Critic passed -> b={b:.4f}")
        elif action == Action.GENERATE:
            b = controller.update_belief_after_generation(b)
            print(f"     Generated -> b={b:.4f}")
        else:
            print("  -> Give up")
            break
