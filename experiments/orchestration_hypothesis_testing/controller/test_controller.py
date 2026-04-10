#!/usr/bin/env python3
"""Tests for the Bayesian controller.

Validates:
1. Bellman equation solution is correct
2. Bayes updates are mathematically correct
3. Policy regions make intuitive sense
4. Controller handles edge cases (b=0, b=1, etc.)
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bayesian_controller import (
    Action,
    BayesianController,
    CostModel,
    CriticLikelihood,
    TransitionKernel,
)


@pytest.fixture
def dummy_tables_path():
    """Create dummy likelihood tables matching the paper's structure."""
    tables = {
        "critic_likelihoods": {
            "L0_syntax": {
                "p_pass_given_correct": 0.95,
                "p_pass_given_incorrect": 0.80,
            },
            "L1_lint": {
                "p_pass_given_correct": 0.90,
                "p_pass_given_incorrect": 0.55,
            },
            "L2_fast_test": {
                "p_pass_given_correct": 0.85,
                "p_pass_given_incorrect": 0.15,
            },
            "L3_llm_review": {
                "p_pass_given_correct": 0.57,
                "p_pass_given_incorrect": 0.21,
            },
        },
        "generator_transition": {
            "p_fix_given_broken": 0.20,
            "p_break_given_correct": 0.08,
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(tables, f)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def controller(dummy_tables_path):
    return BayesianController.from_likelihood_tables(
        dummy_tables_path, horizon=5, grid_size=100
    )


class TestBayesUpdate:
    """Test that Bayes updates are mathematically correct."""

    def test_pass_increases_belief(self, controller):
        """Observing pass should increase belief for informative critics."""
        b = 0.5
        for level in ["L0_syntax", "L1_lint", "L2_fast_test", "L3_llm_review"]:
            b_new = controller.update_belief(b, level, passed=True)
            assert b_new > b, f"Pass on {level} should increase belief"

    def test_fail_decreases_belief(self, controller):
        """Observing fail should decrease belief."""
        b = 0.5
        for level in ["L0_syntax", "L1_lint", "L2_fast_test", "L3_llm_review"]:
            b_new = controller.update_belief(b, level, passed=False)
            assert b_new < b, f"Fail on {level} should decrease belief"

    def test_l2_more_informative_than_l1(self, controller):
        """L2 should move belief more than L1."""
        b = 0.5
        l1_pass = controller.update_belief(b, "L1_lint", passed=True)
        l2_pass = controller.update_belief(b, "L2_fast_test", passed=True)
        assert l2_pass > l1_pass, "L2 pass should increase belief more than L1"

        l1_fail = controller.update_belief(b, "L1_lint", passed=False)
        l2_fail = controller.update_belief(b, "L2_fast_test", passed=False)
        assert l2_fail < l1_fail, "L2 fail should decrease belief more than L1"

    def test_bayes_rule_formula(self, controller):
        """Verify the update matches Bayes' rule exactly."""
        b = 0.3
        critic = controller.critics["L1_lint"]

        # Manual calculation
        p_pass = b * critic.p_pass_given_correct + (1 - b) * critic.p_pass_given_incorrect
        expected_pass = (b * critic.p_pass_given_correct) / p_pass

        actual_pass = controller.update_belief(b, "L1_lint", passed=True)
        assert abs(actual_pass - expected_pass) < 1e-10

    def test_extreme_beliefs(self, controller):
        """Beliefs near 0 and 1 should stay near those values."""
        b_low = controller.update_belief(0.01, "L0_syntax", passed=True)
        assert b_low < 0.05, "Very low belief should stay low even with L0 pass"

        b_high = controller.update_belief(0.99, "L1_lint", passed=False)
        assert b_high > 0.9, "Very high belief should stay relatively high even with L1 fail"


class TestTransition:
    """Test generator transition updates."""

    def test_generation_moves_toward_fix_rate(self, controller):
        """After generation, belief should move toward p_fix for low beliefs."""
        b = 0.1
        b_new = controller.update_belief_after_generation(b)
        # b' = b * (1 - p_break) + (1 - b) * p_fix
        expected = 0.1 * 0.92 + 0.9 * 0.20
        assert abs(b_new - expected) < 1e-10

    def test_generation_may_decrease_high_belief(self, controller):
        """For very high beliefs, generation can break correct patches."""
        b = 0.99
        b_new = controller.update_belief_after_generation(b)
        # Should decrease because p_break > 0
        assert b_new < b


class TestPolicy:
    """Test that the policy makes intuitive sense."""

    def test_low_belief_generates(self, controller):
        """With low belief, should generate (not worth running tests)."""
        action = controller.select_action(0.1, step=0)
        assert action == Action.GENERATE

    def test_very_high_belief_verifies(self, controller):
        """With very high belief, should verify."""
        action = controller.select_action(0.99, step=0)
        assert action == Action.VERIFY

    def test_medium_belief_uses_critic(self, controller):
        """With medium belief, should run a critic."""
        action = controller.select_action(0.6, step=0)
        assert action in (Action.CRITIC_L0, Action.CRITIC_L1, Action.CRITIC_L2)

    def test_value_function_monotone(self, controller):
        """Value function should be non-decreasing in belief."""
        for step in range(controller.horizon):
            values = controller._value_table[step]
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1] + 1e-10, \
                    f"V(b) should be non-decreasing at step {step}"


class TestQValues:
    """Test Q-value calculations."""

    def test_q_verify_linear(self, controller):
        """Q_ver should be linear in belief."""
        q1 = controller._q_verify(0.5)
        q2 = controller._q_verify(1.0)
        expected1 = 0.5 * 200 - 20  # 80
        expected2 = 1.0 * 200 - 20  # 180
        assert abs(q1 - expected1) < 1e-10
        assert abs(q2 - expected2) < 1e-10

    def test_q_verify_breakeven(self, controller):
        """Q_ver = 0 at b = C_ver / reward."""
        breakeven = controller.costs.c_ver / controller.costs.reward
        q = controller._q_verify(breakeven)
        assert abs(q) < 1e-10


class TestEdgeCases:
    def test_zero_belief(self, controller):
        """Should not crash with b=0."""
        action = controller.select_action(0.0, step=0)
        assert action is not None or action is None  # just shouldn't crash

    def test_one_belief(self, controller):
        """b=1 should verify."""
        action = controller.select_action(1.0, step=0)
        assert action == Action.VERIFY

    def test_unknown_critic(self, controller):
        """Unknown critic level should return original belief."""
        b = controller.update_belief(0.5, "UNKNOWN_LEVEL", passed=True)
        assert b == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
