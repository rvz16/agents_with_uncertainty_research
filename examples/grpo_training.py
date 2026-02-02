#!/usr/bin/env python3
"""GRPO Training example for SAGE-Agent.

This example demonstrates the certainty-weighted GRPO training procedure
from Section 6.2 of the paper.

Usage:
    pip install sage-agent[training]
    python examples/grpo_training.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from sage_agent import (
    ActionType,
    CertaintyWeightedReward,
    SAGEGRPOConfig,
    SAGEGRPOTrainer,
    GRPOStepResult,
)


# =============================================================================
# Mock Policy and Reference Models
# =============================================================================

class MockPolicyModel:
    """Mock policy model for demonstration."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.params = [torch.nn.Parameter(torch.zeros(1))]

    def sample(
        self,
        prompts: Sequence[str],
        num_samples: int,
    ) -> List[List[str]]:
        """Sample responses for each prompt."""
        return [[self.responses[i % len(self.responses)] for i in range(num_samples)]
                for _ in prompts]

    def log_probs(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """Return log probabilities for responses."""
        batch_size = len(prompts)
        num_samples = len(responses[0]) if responses else 0
        # Mock log probs - in practice this would be computed from the model
        return torch.randn(batch_size, num_samples, requires_grad=True)


class MockReferenceModel:
    """Mock reference model for KL constraint."""

    def log_probs(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """Return reference log probabilities."""
        batch_size = len(prompts)
        num_samples = len(responses[0]) if responses else 0
        return torch.zeros(batch_size, num_samples)


# =============================================================================
# Certainty-Weighted Reward Demo
# =============================================================================

def demo_certainty_weighting():
    """Demonstrate certainty-weighted reward computation."""
    print("=" * 60)
    print("Certainty-Weighted Reward (Section 6.2)")
    print("=" * 60)

    reward_computer = CertaintyWeightedReward(min_certainty=0.01)

    # Scenario 1: Confident tool call (should get high weight)
    print("\n📍 Scenario 1: Confident Tool Call")
    print("   max_c π_c(t) = 0.9 (high confidence)")
    print("   Action: TOOL_CALL")
    certainty = reward_computer.compute_certainty(
        ActionType.TOOL_CALL,
        max_candidate_probability=0.9,
    )
    weighted = reward_computer.weighted_reward(
        base_reward=1.0,
        action_type=ActionType.TOOL_CALL,
        max_candidate_probability=0.9,
    )
    print(f"   Cert(a_t) = {certainty:.2f}")
    print(f"   R(a_t) = Cert(a_t) × r_base = {weighted:.2f}")
    print("   ✅ High reward - tool call when confident is good!")

    # Scenario 2: Uncertain tool call (should get low weight)
    print("\n📍 Scenario 2: Uncertain Tool Call")
    print("   max_c π_c(t) = 0.3 (low confidence)")
    print("   Action: TOOL_CALL")
    certainty = reward_computer.compute_certainty(
        ActionType.TOOL_CALL,
        max_candidate_probability=0.3,
    )
    weighted = reward_computer.weighted_reward(
        base_reward=1.0,
        action_type=ActionType.TOOL_CALL,
        max_candidate_probability=0.3,
    )
    print(f"   Cert(a_t) = {certainty:.2f}")
    print(f"   R(a_t) = Cert(a_t) × r_base = {weighted:.2f}")
    print("   ⚠️ Low reward - should have asked a question instead!")

    # Scenario 3: Asking when uncertain (should get high weight)
    print("\n📍 Scenario 3: Question When Uncertain")
    print("   max_c π_c(t) = 0.3 (low confidence)")
    print("   Action: CLARIFICATION")
    certainty = reward_computer.compute_certainty(
        ActionType.CLARIFICATION,
        max_candidate_probability=0.3,
    )
    weighted = reward_computer.weighted_reward(
        base_reward=1.0,
        action_type=ActionType.CLARIFICATION,
        max_candidate_probability=0.3,
    )
    print(f"   Cert(a_t) = 1 - max_prob = {certainty:.2f}")
    print(f"   R(a_t) = Cert(a_t) × r_base = {weighted:.2f}")
    print("   ✅ High reward - asking when uncertain is good!")

    # Scenario 4: Asking when confident (wasteful)
    print("\n📍 Scenario 4: Question When Confident")
    print("   max_c π_c(t) = 0.9 (high confidence)")
    print("   Action: CLARIFICATION")
    certainty = reward_computer.compute_certainty(
        ActionType.CLARIFICATION,
        max_candidate_probability=0.9,
    )
    weighted = reward_computer.weighted_reward(
        base_reward=1.0,
        action_type=ActionType.CLARIFICATION,
        max_candidate_probability=0.9,
    )
    print(f"   Cert(a_t) = 1 - max_prob = {certainty:.2f}")
    print(f"   R(a_t) = Cert(a_t) × r_base = {weighted:.2f}")
    print("   ⚠️ Low reward - unnecessary question wastes user time!")


# =============================================================================
# GRPO Training Demo
# =============================================================================

def demo_grpo_training():
    """Demonstrate SAGE GRPO training loop."""
    print("\n" + "=" * 60)
    print("SAGE GRPO Training Loop")
    print("=" * 60)

    # Mock models
    policy = MockPolicyModel(responses=[
        "book_flight(origin=NYC, dest=LAX)",
        "What city are you flying from?",
        "book_flight(origin=<UNK>, dest=LAX)",
    ])
    reference = MockReferenceModel()

    # Reward function that returns (base_reward, action_type, max_prob)
    def sage_reward_fn(prompt: str, response: str) -> Tuple[float, ActionType, float]:
        """Compute reward with action type and probability."""
        # Mock logic - in practice this would evaluate the response
        if "book_flight" in response and "<UNK>" not in response:
            return (1.0, ActionType.TOOL_CALL, 0.85)  # Confident call
        elif "?" in response:
            return (0.5, ActionType.CLARIFICATION, 0.4)  # Question when uncertain
        else:
            return (0.0, ActionType.TOOL_CALL, 0.3)  # Uncertain call

    # Configuration
    config = SAGEGRPOConfig(
        num_samples=4,
        kl_coef=0.02,
        max_grad_norm=1.0,
        min_certainty=0.01,
        use_certainty_weighting=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(policy.params, lr=1e-4)

    # Trainer
    trainer = SAGEGRPOTrainer(
        policy=policy,
        reference=reference,
        reward_fn=sage_reward_fn,
        config=config,
        optimizer=optimizer,
    )

    # Training prompts
    prompts = [
        "Book a flight to Los Angeles",
        "I need to travel next week",
        "Find me a flight from Boston",
    ]

    print("\n📈 Training for 3 epochs...")
    for epoch in range(3):
        results = trainer.train_epoch(prompts)
        avg_loss = sum(r.loss for r in results) / len(results)
        avg_reward = sum(r.mean_reward for r in results) / len(results)
        avg_kl = sum(r.mean_kl for r in results) / len(results)
        print(f"   Epoch {epoch + 1}: loss={avg_loss:.4f}, reward={avg_reward:.4f}, kl={avg_kl:.4f}")

    print("\n✅ Training complete!")
    print("\nKey insight from Section 6.2:")
    print("   The certainty-weighted reward encourages the policy to:")
    print("   • Execute tool calls when confident (high max_prob)")
    print("   • Ask clarifying questions when uncertain (low max_prob)")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all demos."""
    demo_certainty_weighting()
    demo_grpo_training()

    print("\n" + "=" * 60)
    print("Summary: SAGE GRPO Training")
    print("=" * 60)
    print("""
The SAGE GRPO trainer extends standard GRPO with certainty-weighted rewards:

    R(a_t) = Cert(a_t) × r_base(a_t)

Where:
    • Cert(a_t) = max_c π_c(t)       for tool calls
    • Cert(a_t) = 1 - max_c π_c(t)   for clarification questions

This ensures the policy learns to calibrate its actions with uncertainty,
asking questions when needed rather than making uncertain tool calls.

Paper reference: Section 6.2 "Certainty-Weighted GRPO"
""")


if __name__ == "__main__":
    main()
