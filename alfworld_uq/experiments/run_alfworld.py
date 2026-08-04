from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.react_agent import AgentError, RandomAdmissibleAgent, ReActAgent
from environments.alfworld_env import ALFWorldTextEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_agent(args: argparse.Namespace) -> Any:
    if args.policy == "random":
        return RandomAdmissibleAgent(seed=args.seed)

    base_url = os.getenv("LLM_BASE_URI")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
    missing = [
        name
        for name, value in (("LLM_BASE_URI", base_url), ("LLM_API_KEY", api_key))
        if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)} in {args.env_file}. "
            "Use --policy random only for an offline pipeline smoke-test."
        )
    extra_body = None
    if (
        args.require_api_parameters
        and not args.no_logprobs
        and "openrouter.ai" in base_url
    ):
        provider: dict[str, Any] = {"require_parameters": True}
        if args.provider_order:
            provider["order"] = [
                value.strip()
                for value in args.provider_order.split(",")
                if value.strip()
            ]
            provider["allow_fallbacks"] = args.allow_provider_fallbacks
        extra_body = {"provider": provider}

    return ReActAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.api_timeout,
        max_retries=args.api_retries,
        max_tokens=args.max_generation_tokens,
        request_logprobs=not args.no_logprobs,
        repeat_action_limit=args.repeat_action_limit,
        seed=args.seed,
        extra_body=extra_body,
        max_empty_response_retries=args.empty_response_retries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect ReAct trajectories from text-only ALFWorld."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument(
        "--gamefile",
        type=Path,
        help="Run one exact ALFWorld gamefile (used for deterministic repair).",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("ALFWORLD_DATA", "~/.cache/alfworld")).expanduser(),
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid_seen", "valid_unseen"],
        default="valid_seen",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["llm", "random"], default="llm")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--max-generation-tokens", type=int, default=1024)
    parser.add_argument("--empty-response-retries", type=int, default=1)
    parser.add_argument("--repeat-action-limit", type=int, default=2)
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument(
        "--require-api-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On OpenRouter, route only to providers advertising requested parameters.",
    )
    parser.add_argument(
        "--provider-order",
        default=os.getenv("OPENROUTER_PROVIDER_ORDER", ""),
        help="Comma-separated OpenRouter provider order.",
    )
    parser.add_argument(
        "--allow-provider-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args()
    parser = build_parser()
    if preliminary.config:
        payload = json.loads(preliminary.config.read_text(encoding="utf-8"))
        for key in ("output_dir", "data_root", "env_file"):
            if key in payload:
                payload[key] = Path(payload[key]).expanduser()
        parser.set_defaults(**payload)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = args.output_dir / "trajectories.jsonl"
    episodes_path = args.output_dir / "episodes.jsonl"
    if not args.overwrite and (trajectories_path.exists() or episodes_path.exists()):
        raise SystemExit(
            f"{args.output_dir} already contains a run; pass --overwrite to replace it."
        )
    if args.overwrite:
        trajectories_path.unlink(missing_ok=True)
        episodes_path.unlink(missing_ok=True)

    agent = _build_agent(args)
    model = (
        os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
        if args.policy == "llm"
        else "offline-random"
    )
    public_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_episodes": args.num_episodes,
        "episode_offset": args.episode_offset,
        "gamefile": str(args.gamefile) if args.gamefile else None,
        "max_steps": args.max_steps,
        "split": args.split,
        "seed": args.seed,
        "policy": args.policy,
        "model": model,
        "request_logprobs": not args.no_logprobs,
        "empty_response_retries": args.empty_response_retries,
        "require_api_parameters": args.require_api_parameters,
        "provider_order": args.provider_order,
        "allow_provider_fallbacks": args.allow_provider_fallbacks,
        "data_root": str(args.data_root),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(public_config, indent=2), encoding="utf-8"
    )

    env = ALFWorldTextEnv(
        data_root=args.data_root,
        split=args.split,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        episode_offset=args.episode_offset,
        seed=args.seed,
        gamefile=args.gamefile,
    )
    success_count = 0
    try:
        for episode_index in range(args.num_episodes):
            started = time.monotonic()
            initial = env.reset()
            observation = initial.observation
            admissible = initial.admissible_actions
            history: list[dict[str, str]] = []
            records: list[dict[str, Any]] = []
            total_tokens = 0
            stop_reason = "max_steps"
            final_success = False

            for step_number in range(1, args.max_steps + 1):
                try:
                    generation = agent.act(initial.task, history, admissible)
                except AgentError as exc:
                    records.append(
                        {
                            "episode_id": initial.episode_id,
                            "task_type": initial.task_type,
                            "task": initial.task,
                            "step": step_number,
                            "thought": "",
                            "action": "",
                            "observation": observation,
                            "admissible_actions": admissible,
                            "token_logprobs": [],
                            "perplexity": None,
                            "seqprob": None,
                            "verb": None,
                            "progress": None,
                            "done": True,
                            "final_success": False,
                            "error": str(exc),
                            "uq": {},
                        }
                    )
                    stop_reason = "api_error"
                    break

                result = env.step(generation.action)
                combined_uq = generation.uq.get("combined", {})
                record = {
                    "episode_id": initial.episode_id,
                    "task_type": initial.task_type,
                    "task": initial.task,
                    "step": step_number,
                    "thought": generation.thought,
                    "action": generation.action,
                    "proposed_action": generation.proposed_action,
                    "observation": result.observation,
                    "admissible_actions": admissible,
                    "token_logprobs": generation.token_logprobs,
                    "perplexity": combined_uq.get("perplexity"),
                    "seqprob": combined_uq.get("sequence_probability"),
                    "verb": combined_uq.get("verbalized_confidence"),
                    "progress": result.progress,
                    "done": result.done,
                    "final_success": False,
                    "format_valid": generation.format_valid,
                    "action_valid": generation.action_valid,
                    "fallback_reason": generation.fallback_reason,
                    "raw_response": generation.raw_text,
                    "logprobs_available": generation.logprobs_available,
                    "provider": generation.provider,
                    "uq": generation.uq,
                    "usage": {
                        "prompt_tokens": generation.prompt_tokens,
                        "completion_tokens": generation.completion_tokens,
                        "total_tokens": generation.total_tokens,
                        "request_attempts": generation.request_attempts,
                        "empty_response_retries": generation.empty_response_retries,
                        "generation_token_limit": generation.generation_token_limit,
                    },
                }
                records.append(record)
                total_tokens += generation.total_tokens
                history.append(
                    {
                        "thought": generation.thought,
                        "action": generation.action,
                        "observation": result.observation,
                    }
                )
                observation = result.observation
                admissible = result.admissible_actions
                if result.done:
                    final_success = result.won
                    if result.won:
                        stop_reason = "success"
                    elif step_number >= args.max_steps:
                        stop_reason = "max_steps"
                    else:
                        stop_reason = "environment_done"
                    break

            for record in records:
                record["final_success"] = final_success
            _write_jsonl(trajectories_path, records)
            duration = time.monotonic() - started
            summary = {
                "episode_id": initial.episode_id,
                "task_type": initial.task_type,
                "task": initial.task,
                "gamefile": initial.gamefile,
                "final_success": final_success,
                "num_steps": len(records),
                "stop_reason": stop_reason,
                "total_tokens": total_tokens,
                "duration_seconds": duration,
            }
            _write_jsonl(episodes_path, [summary])
            success_count += int(final_success)
            print(
                f"[{episode_index + 1}/{args.num_episodes}] "
                f"{initial.task_type}: {stop_reason}, steps={len(records)}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        env.close()

    print(
        json.dumps(
            {
                "episodes": args.num_episodes,
                "successes": success_count,
                "success_rate": success_count / args.num_episodes,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
