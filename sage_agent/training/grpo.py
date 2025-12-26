from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Protocol, Sequence, Tuple

import torch


class PolicyModel(Protocol):
    def sample(self, prompts: Sequence[str], num_samples: int) -> List[List[str]]:
        ...

    def log_probs(self, prompts: Sequence[str], responses: Sequence[Sequence[str]]) -> torch.Tensor:
        """Return log-probabilities shaped [batch, num_samples]."""
        ...


class ReferenceModel(Protocol):
    def log_probs(self, prompts: Sequence[str], responses: Sequence[Sequence[str]]) -> torch.Tensor:
        ...


RewardFn = Callable[[str, str], float]


@dataclass
class GRPOConfig:
    num_samples: int = 4
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    device: str = "cpu"


@dataclass
class GRPOStepResult:
    loss: float
    mean_reward: float
    mean_kl: float


class GRPOTrainer:
    def __init__(
        self,
        policy: PolicyModel,
        reference: ReferenceModel,
        reward_fn: RewardFn,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

    def train_epoch(self, prompts: Sequence[str]) -> List[GRPOStepResult]:
        results: List[GRPOStepResult] = []
        for prompt in prompts:
            result = self._train_step([prompt])
            results.append(result)
        return results

    def _train_step(self, prompts: Sequence[str]) -> GRPOStepResult:
        responses = self.policy.sample(prompts, self.config.num_samples)
        rewards = self._compute_rewards(prompts, responses)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)

        policy_log_probs = self.policy.log_probs(prompts, responses)
        ref_log_probs = self.reference.log_probs(prompts, responses)
        kl = policy_log_probs - ref_log_probs

        advantages = _group_normalize(rewards_tensor)
        loss = -(advantages * policy_log_probs).mean() + self.config.kl_coef * kl.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]["params"], self.config.max_grad_norm
        )
        self.optimizer.step()

        return GRPOStepResult(
            loss=float(loss.detach().cpu().item()),
            mean_reward=float(rewards_tensor.mean().cpu().item()),
            mean_kl=float(kl.mean().detach().cpu().item()),
        )

    def _compute_rewards(
        self, prompts: Sequence[str], responses: Sequence[Sequence[str]]
    ) -> List[List[float]]:
        reward_matrix: List[List[float]] = []
        for prompt, response_group in zip(prompts, responses):
            row = [self.reward_fn(prompt, response) for response in response_group]
            reward_matrix.append(row)
        return reward_matrix


def _group_normalize(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + eps
    return (rewards - mean) / std
