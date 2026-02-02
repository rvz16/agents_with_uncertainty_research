#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="When2Call evaluation using llm-tts uncertainty as logprob proxy."
    )
    parser.add_argument(
        "--split",
        choices=("llm_judge", "mcq"),
        default="llm_judge",
        help="When2Call split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit (0 = no limit).",
    )
    parser.add_argument(
        "--model",
        default="xiaomi/mimo-v2-flash:free",
        help="Model name for llm-tts service.",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8001/v1",
        help="llm-tts service URL.",
    )
    parser.add_argument(
        "--tts-strategy",
        default="self_consistency",
        help="llm-tts strategy name.",
    )
    parser.add_argument(
        "--tts-budget",
        type=int,
        default=8,
        help="llm-tts budget.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for llm-tts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout for llm-tts requests (seconds).",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-example results (recommended with --limit).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a demo-style summary for the uncertainty-proxy choice.",
    )
    return parser.parse_args()


def _load_when2call(split: str) -> Sequence[Mapping[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset("nvidia/When2Call", "test")
    return dataset[split]


def _build_tools_text(tools: Sequence[str]) -> str:
    if not tools:
        return "<none>"
    return "\n".join(tools)


def _build_options(row: Mapping[str, object]) -> Dict[str, str]:
    answers = row.get("answers", {}) or {}
    options = {
        "direct": answers.get("direct", ""),
        "tool_call": answers.get("tool_call", ""),
        "request_for_info": answers.get("request_for_info", ""),
        "cannot_answer": answers.get("cannot_answer", ""),
    }
    return {k: v for k, v in options.items() if isinstance(v, str)}


def _prompt_choose_with_options(
    question: str, tools_text: str, options: Mapping[str, str]
) -> str:
    options_block = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return (
        "You are evaluating tool-calling responses.\n"
        "Choose the best answer type for the user question.\n"
        "Reply with one label only: direct, tool_call, request_for_info, cannot_answer.\n"
        f"Question: {question}\n"
        f"Tools:\n{tools_text}\n"
        f"Options:\n{options_block}\n"
        "Label:"
    )


def _prompt_choose_without_options(question: str, tools_text: str) -> str:
    return (
        "You are evaluating tool-calling responses.\n"
        "Choose the best answer type for the user question.\n"
        "Reply with one label only: direct, tool_call, request_for_info, cannot_answer.\n"
        f"Question: {question}\n"
        f"Tools:\n{tools_text}\n"
        "Label:"
    )


def _prompt_forced_label(question: str, tools_text: str, label: str) -> str:
    return (
        "You are evaluating tool-calling responses.\n"
        "Only respond with the label provided below.\n"
        f"Question: {question}\n"
        f"Tools:\n{tools_text}\n"
        f"Label: {label}\n"
        "Response:"
    )


def _normalize_label(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = text.strip().lower()
    for label in ("direct", "tool_call", "request_for_info", "cannot_answer"):
        if label == cleaned:
            return label
    return None


@dataclass
class EvalCounts:
    total: int = 0
    correct: int = 0

    def add(self, correct: bool) -> None:
        self.total += 1
        self.correct += 1 if correct else 0

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / float(self.total)


def _choose_by_uncertainty(
    llm, question: str, tools_text: str, options: Mapping[str, str]
) -> Tuple[Optional[str], Dict[str, float]]:
    scores: Dict[str, float] = {}
    for label in options.keys():
        prompt = _prompt_forced_label(question, tools_text, label)
        _ = llm.complete(prompt)
        uncertainty = llm.last_uncertainty
        if uncertainty is None:
            uncertainty = 1.0
        scores[label] = float(uncertainty)
    best = min(scores.items(), key=lambda kv: kv[1])[0] if scores else None
    return best, scores


def main() -> None:
    args = _parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from tts_llm_client import TTSLLMClient

    rows = list(_load_when2call(args.split))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    llm = TTSLLMClient(
        base_url=args.service_url,
        model=args.model,
        tts_strategy=args.tts_strategy,
        tts_budget=args.tts_budget,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    counts_option = EvalCounts()
    counts_direct = EvalCounts()
    counts_uncertainty = EvalCounts()
    per_source: Dict[str, EvalCounts] = {}

    for row in rows:
        question = row.get("question", "")
        tools_text = _build_tools_text(row.get("tools") or [])
        options = _build_options(row)
        correct = row.get("correct_answer")
        if not options or not isinstance(correct, str):
            continue

        option_prompt = _prompt_choose_with_options(question, tools_text, options)
        option_label = _normalize_label(llm.complete(option_prompt))
        option_correct = option_label == correct
        counts_option.add(option_correct)

        direct_prompt = _prompt_choose_without_options(question, tools_text)
        direct_label = _normalize_label(llm.complete(direct_prompt))
        direct_correct = direct_label == correct
        counts_direct.add(direct_correct)

        uncertainty_label, uncertainty_scores = _choose_by_uncertainty(
            llm, question, tools_text, options
        )
        uncertainty_correct = uncertainty_label == correct
        counts_uncertainty.add(uncertainty_correct)

        source = row.get("source", "unknown")
        per_source.setdefault(source, EvalCounts()).add(uncertainty_correct)

        if args.pretty:
            chosen_uncertainty = (
                uncertainty_scores.get(uncertainty_label)
                if uncertainty_label is not None
                else None
            )
            confidence = (
                max(0.0, min(1.0, 1.0 - chosen_uncertainty))
                if chosen_uncertainty is not None
                else None
            )
            status = "ACCEPTED" if uncertainty_correct else "REJECTED"
            print("=" * 60)
            print("When2Call Uncertainty-Aware Selection")
            print("=" * 60)
            print(f"Question: {question}")
            print("-" * 40)
            print(f"Status: {status}")
            print(f"Answer: {uncertainty_label}")
            if confidence is not None:
                print(f"Confidence: {confidence:.2f}")
            if chosen_uncertainty is not None:
                print(f"Uncertainty: {chosen_uncertainty:.2f}")
            print("Attempts: 1")
            print(f"Final budget: {args.tts_budget}")
        if args.print_each:
            print("uuid:", row.get("uuid"))
            print("question:", question)
            print("correct:", correct)
            print("option_label:", option_label)
            print("direct_label:", direct_label)
            print("uncertainty_label:", uncertainty_label, "scores:", uncertainty_scores)
            print("-" * 60)

    print("rows_evaluated:", counts_uncertainty.total)
    print("option_prompt_accuracy:", counts_option.accuracy())
    print("direct_prompt_accuracy:", counts_direct.accuracy())
    print("uncertainty_proxy_accuracy:", counts_uncertainty.accuracy())
    print("per_source_accuracy (uncertainty proxy):")
    for source, counts in sorted(per_source.items()):
        print("  ", source, "=", counts.accuracy())


if __name__ == "__main__":
    main()
