"""Small deterministic adapter used to validate the portable pipeline."""

from __future__ import annotations

import random

from trajectory_uq_toolkit.signals import summarize_token_logprobs


def list_episodes(config: dict) -> list[str]:
    return [f"toy_{index:04d}" for index in range(int(config.get("num_episodes", 40)))]


def run_episode(episode_id: str, seed: int, config: dict) -> dict:
    rng = random.Random(seed)
    numeric_id = int(episode_id.rsplit("_", 1)[1])
    success = int((numeric_id * 7 + 3) % 10 < 4)
    min_generations = int(config.get("min_generations", 2))
    max_generations = int(config.get("max_generations", 5))
    generation_count = rng.randint(min_generations, max_generations)
    generations = []
    tool_results = []
    for index in range(generation_count):
        center = -0.35 if success else -1.25
        token_count = rng.randint(5, 12) + (0 if success else 4)
        chosen = [rng.gauss(center, 0.18) for _ in range(token_count)]
        top = [
            [logprob, logprob - rng.uniform(0.2, 1.2), logprob - rng.uniform(0.8, 2.0)]
            for logprob in chosen
        ]
        signals = summarize_token_logprobs(chosen, top)
        action_valid = rng.random() < (0.95 if success else 0.65)
        format_valid = rng.random() < (0.98 if success else 0.78)
        tool_results.append(action_valid)
        generations.append(
            {
                "index": index,
                "signals": signals,
                "critics": {
                    "format_valid": format_valid,
                    "action_valid": action_valid,
                    "no_repeated_fallback": rng.random() < (0.97 if success else 0.7),
                },
            }
        )
    return {
        "episode_id": episode_id,
        "environment": str(config.get("environment", "toy-debug")),
        "success": success,
        "generations": generations,
        "critics": {
            "all_formats_valid": all(row["critics"]["format_valid"] for row in generations),
            "all_actions_valid": all(row["critics"]["action_valid"] for row in generations),
            "no_repeated_fallback": all(
                row["critics"]["no_repeated_fallback"] for row in generations
            ),
            "llm_judge_pass": rng.random() < (0.9 if success else 0.12),
        },
        "features": {"tool_success_rate": sum(tool_results) / len(tool_results)},
        "metadata": {"task_family": "toy"},
    }
