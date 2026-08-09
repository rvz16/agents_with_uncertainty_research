from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
SYSTEM_PROMPT = """You are an independent evaluator of a text-only household agent.
Infer whether the task was completed from the visible task and interaction transcript.
Do not assume success merely because the transcript ended. Return one JSON object only:
{"verdict":"PASS|FAIL","confidence":0.0,"reason":"one short sentence"}
PASS means the transcript provides enough evidence that every requested condition is
satisfied. Otherwise return FAIL. Confidence is your probability that the verdict is
correct, between 0 and 1."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def build_judge_prompt(rows: list[dict[str, Any]]) -> str:
    """Build a label-free transcript from explicitly allowed trajectory fields."""
    if not rows:
        raise ValueError("trajectory must contain at least one step")
    task = str(rows[0].get("task", "")).strip()
    transcript = []
    for row in sorted(rows, key=lambda item: int(item["step"])):
        transcript.extend(
            [
                f"Step {int(row['step'])}",
                f"Thought: {str(row.get('thought', '')).strip()}",
                f"Action: {str(row.get('action', '')).strip()}",
                f"Observation: {str(row.get('observation', '')).strip()}",
            ]
        )
    return f"Task: {task}\n\nInteraction transcript:\n" + "\n".join(transcript)


def parse_judge_response(text: str) -> tuple[bool, float, str]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    verdict = str(payload["verdict"]).strip().upper()
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError(f"unsupported verdict: {verdict!r}")
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    reason = str(payload.get("reason", "")).strip()[:500]
    return verdict == "PASS", confidence, reason


def _usage(response: Any, name: str) -> int:
    return int(getattr(getattr(response, "usage", None), name, 0) or 0)


def _judge_one(
    client: OpenAI,
    episode_id: str,
    rows: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    prompt = build_judge_prompt(rows)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            passed, confidence, reason = parse_judge_response(text)
            return {
                "episode_id": episode_id,
                "model": model,
                "judge_pass": passed,
                "confidence": confidence,
                "reason": reason,
                "raw_response": text,
                "prompt_hash": prompt_hash,
                "num_steps": len(rows),
                "prompt_tokens": _usage(response, "prompt_tokens"),
                "completion_tokens": _usage(response, "completion_tokens"),
                "total_tokens": _usage(response, "total_tokens"),
                "attempts": attempt,
                "status": "ok",
            }
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return {
        "episode_id": episode_id,
        "model": model,
        "prompt_hash": prompt_hash,
        "num_steps": len(rows),
        "attempts": retries + 1,
        "status": "error",
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score saved ALFWorld trajectories with an LLM judge."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    load_dotenv(args.env_file, override=False)
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPEN_ROUTER_API_KEY")
        or os.getenv("OPEN_ROUTER")
    )
    if not api_key:
        raise SystemExit(f"OpenRouter API key not found in {args.env_file}")

    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(args.trajectories):
        trajectories[str(row["episode_id"])].append(row)
    episode_ids = sorted(trajectories)
    if args.limit > 0:
        episode_ids = episode_ids[: args.limit]

    if args.overwrite:
        args.output.unlink(missing_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            str(row["episode_id"])
            for row in _read_jsonl(args.output)
            if row.get("status") == "ok" and row.get("model") == args.model
        }
    pending = [episode_id for episode_id in episode_ids if episode_id not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=120.0, max_retries=0)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _judge_one,
                client,
                episode_id,
                trajectories[episode_id],
                model=args.model,
                max_tokens=args.max_tokens,
                retries=args.retries,
            ): episode_id
            for episode_id in pending
        }
        with args.output.open("a", encoding="utf-8") as handle:
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                failures += int(result["status"] != "ok")
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                if done % 10 == 0 or done == len(futures):
                    print(
                        f"judged {done}/{len(futures)} pending episodes "
                        f"({failures} failures)",
                        flush=True,
                    )
    if failures:
        raise SystemExit(f"{failures} judge calls failed; rerun without --overwrite")


if __name__ == "__main__":
    main()
