#!/usr/bin/env python3
"""
Enhanced When2Call evaluation for SAGE-Agent with improved uncertainty integration.

This evaluation script implements all the improvements from the SAGE-Agent paper:
1. Proper structured + LLM uncertainty combination
2. Adaptive thresholds for critical operations
3. LLM-backed constraint extraction
4. Uncertainty propagation across steps
5. Calibration metrics (ECE, MCE, Brier score)

Usage:
    python run_when2call_eval.py --limit 10 --print-each
    python run_when2call_eval.py --model "qwen/qwen-2.5-7b-instruct" --tts-budget 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAGE-Agent on When2Call with llm-tts uncertainty."
    )
    # Default to 2 levels up from different_agents/evaluations/
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--sage-root",
        type=Path,
        default=default_root,
        help="Path to agents_with_uncertainty_research repo.",
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
        "--filter-correct-answer",
        default="tool_call",
        help="Only keep rows with this correct_answer value.",
    )
    parser.add_argument(
        "--use-orig-tools",
        action="store_true",
        help="Use orig_tools instead of tools.",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen-2.5-vl-7b-instruct:free",
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
        default=1024,
        help="Max tokens for llm-tts.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=6,
        help="SAGE max_questions (n_s).",
    )
    parser.add_argument(
        "--redundancy-weight",
        type=float,
        default=0.5,
        help="SAGE redundancy weight (lambda).",
    )
    parser.add_argument(
        "--tau-exec",
        type=float,
        default=0.85,
        help="SAGE tau_execute threshold.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="SAGE alpha threshold.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="LangGraph recursion limit.",
    )
    parser.add_argument(
        "--print-each",
        action="store_true",
        help="Print per-example results (recommended with --limit).",
    )
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Use enhanced v2 graph with error recovery and separate belief update.",
    )
    parser.add_argument(
        "--use-v3",
        action="store_true",
        help="Use v3 graph with constrained decoding, SGR, resampling, SAUP, and smart reflexion.",
    )
    parser.add_argument(
        "--v3-config",
        choices=["conservative", "balanced", "aggressive", "lite"],
        default="balanced",
        help="v3 configuration profile (default: balanced)",
    )
    parser.add_argument(
        "--paper-style-accuracy",
        action="store_true",
        help="Report paper-style accuracy on correct_answer labels (tool_call/request_for_info/cannot_answer).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["agent", "mcq", "logprob"],
        default="agent",
        help="Evaluation mode for When2Call: agent (full), mcq, or logprob.",
    )
    return parser.parse_args()


def _load_when2call(split: str) -> Sequence[Mapping[str, object]]:
    from datasets import load_dataset

    dataset = load_dataset("nvidia/When2Call", "test")
    return dataset[split]


def _parse_tool_schema(
    tool_json: Mapping[str, object], ParameterDomain, ToolSchema
) -> ToolSchema:
    name = tool_json.get("name", "")
    params = tool_json.get("parameters", {}) or {}
    required = params.get("required", []) or []
    properties = params.get("properties", {}) or {}
    domains: Dict[str, object] = {}

    for param_name, prop in properties.items():
        domain = _domain_from_property(prop, ParameterDomain)
        domains[param_name] = domain

    for param_name in required:
        if param_name not in domains:
            domains[param_name] = ParameterDomain.continuous()

    return ToolSchema(name=name, parameters=domains, required=frozenset(required))


def _domain_from_property(prop: Mapping[str, object], ParameterDomain):
    enum = prop.get("enum")
    if isinstance(enum, list) and enum:
        return ParameterDomain.from_values(enum)
    if prop.get("type") == "boolean":
        return ParameterDomain.from_values([True, False])
    return ParameterDomain.continuous()


def _parse_tool_call(raw: str, ToolCall) -> ToolCall:
    payload = json.loads(raw)
    return ToolCall(tool_name=payload.get("name", ""), arguments=payload.get("arguments", {}))


@dataclass
class GroundTruthQuestionAsker:
    truth: "ToolCall"
    count: int = 0

    def ask(self, question) -> str:
        self.count += 1
        if not getattr(question, "aspects", None):
            return ""
        values: List[str] = []
        for aspect in question.aspects:
            if aspect.tool_name != self.truth.tool_name:
                continue
            value = self.truth.arguments.get(aspect.param_name)
            if value is None:
                continue
            values.append(str(value))
        return "; ".join(values)


def _build_tool_registry(tool_schemas: Sequence["ToolSchema"]):
    def _dummy_tool(_args):
        return {"ok": True}

    return {tool.name: _dummy_tool for tool in tool_schemas}


def _paper_label(result_state: Mapping[str, object]) -> str:
    result = result_state.get("result")
    status = result_state.get("status")
    if result:
        return "tool_call"
    if status == "escalated":
        return "request_for_info"
    return "cannot_answer"


def _get_correct_label(row: Mapping[str, object]) -> Optional[str]:
    label = row.get("correct_answer")
    if isinstance(label, str):
        return label
    alt = row.get("label")
    if isinstance(alt, str):
        return alt
    return None


def _when2call_label_prompt(question: str) -> str:
    return (
        "Decide which action is appropriate for the user request.\n"
        "Choose exactly one option and answer with a single letter.\n\n"
        "A) tool_call\n"
        "B) request_for_info\n"
        "C) cannot_answer\n\n"
        f"User request: {question}\n"
        "Answer:"
    )


def _get_mcq_options(row: Mapping[str, object]) -> List[str]:
    for key in ("options", "choices"):
        payload = row.get(key)
        if isinstance(payload, list) and payload:
            if all(isinstance(item, str) for item in payload):
                return list(payload)
            if all(isinstance(item, dict) for item in payload):
                texts = [item.get("text") for item in payload if isinstance(item.get("text"), str)]
                if texts:
                    return texts
    return []


def _get_mcq_correct_index(row: Mapping[str, object], options: List[str]) -> Optional[int]:
    label = row.get("label")
    if isinstance(label, int):
        return label if 0 <= label < len(options) else None
    if isinstance(label, str):
        if label.isdigit():
            idx = int(label)
            return idx if 0 <= idx < len(options) else None
        if label.upper() in ("A", "B", "C", "D", "E", "F"):
            idx = ord(label.upper()) - ord("A")
            return idx if 0 <= idx < len(options) else None
    correct_answer = row.get("correct_answer")
    if isinstance(correct_answer, int):
        return correct_answer if 0 <= correct_answer < len(options) else None
    if isinstance(correct_answer, str):
        if correct_answer in options:
            return options.index(correct_answer)
        if correct_answer.upper() in ("A", "B", "C", "D", "E", "F"):
            idx = ord(correct_answer.upper()) - ord("A")
            return idx if 0 <= idx < len(options) else None
    return None


def _build_mcq_prompt(question: str, options: List[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = ["Choose the best answer and reply with a single letter.\n"]
    for idx, opt in enumerate(options):
        label = letters[idx]
        lines.append(f"{label}) {opt}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _parse_label_choice(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    first = cleaned[0].upper()
    mapping = {"A": "tool_call", "B": "request_for_info", "C": "cannot_answer"}
    if first in mapping:
        return mapping[first]
    lowered = cleaned.lower()
    if "tool_call" in lowered:
        return "tool_call"
    if "request_for_info" in lowered:
        return "request_for_info"
    if "cannot_answer" in lowered or "cant_answer" in lowered:
        return "cannot_answer"
    return ""


def _select_logprob_label(logprob_payload: Mapping[str, object]) -> str:
    mapping = {"A": "tool_call", "B": "request_for_info", "C": "cannot_answer"}
    logprobs = None
    try:
        logprobs = logprob_payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception:
        return ""
    best = None
    best_score = float("-inf")
    for entry in logprobs:
        token = (entry.get("token") or "").strip()
        score = entry.get("logprob", float("-inf"))
        letter = token[:1].upper()
        if letter in mapping and score > best_score:
            best_score = score
            best = mapping[letter]
    return best or ""


def main() -> None:
    args = _parse_args()
    if not args.sage_root.exists():
        raise FileNotFoundError(f"Missing sage repo: {args.sage_root}")

    sys.path.insert(0, str(args.sage_root))
    sys.path.insert(0, str(args.sage_root / "different_agents" / "shared"))
    sys.path.insert(0, str(args.sage_root / "different_agents" / "pure_sage"))
    sys.path.insert(0, str(args.sage_root / "different_agents" / "v3"))
    sys.path.insert(0, str(args.sage_root / "different_agents" / "misc"))

    from tts_llm_client import TTSLLMClient

    # Import the appropriate graph version based on --use-v2/--use-v3 flags
    if args.use_v3:
        from langgraph_sage_agent_v3 import (
            GraphDeps,
            build_graph,
            create_initial_state,
        )
        from v3_configs import get_config
        AGENT_CONFIG = get_config(args.v3_config)
        print(f"Using v3 graph ({args.v3_config} profile) with constrained decoding, SGR, resampling, SAUP, and smart reflexion")
    elif args.use_v2:
        from langgraph_sage_agent_v2 import (
            GraphDeps,
            build_graph,
            create_initial_state,
            CONFIG as AGENT_CONFIG,
        )
        print("Using enhanced v2 graph with error recovery and separate belief update")
    else:
        from langgraph_sage_agent import (
            GraphDeps,
            build_graph,
            CONFIG as AGENT_CONFIG,
        )
        create_initial_state = None  # v1 doesn't have this helper
        print("Using standard v1 graph")
    
    from sage_agent import (
        LLMBackedCandidateGenerator,
        LLMBackedQuestionGenerator,
        ParameterDomain,
        SageAgentConfig,
        SimpleConstraintExtractor,
        HybridConstraintExtractor,
        ToolCall,
        ToolRegistryExecutor,
        ToolSchema,
        evaluate_metrics,
        evaluate_extended_metrics,
        compute_uncertainty_aware_accuracy,
        create_sage_propagator,
    )
    from sage_agent.core.types import ExecutionResult

    rows = list(_load_when2call(args.split))
    if args.filter_correct_answer:
        rows = [r for r in rows if r.get("correct_answer") == args.filter_correct_answer]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    predictions: List[ToolCall] = []
    ground_truths: List[ToolCall] = []
    question_counts: List[int] = []
    confidence_scores: List[float] = []  # For calibration metrics
    uncertainty_scores: List[float] = []  # For uncertainty-aware accuracy
    paper_correct: List[bool] = []
    label_counts: Dict[str, int] = {}

    if args.eval_mode in {"mcq", "logprob"}:
        from openai import OpenAI

        api_key = os.getenv("SAGE_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not set. Set SAGE_OPENROUTER_API_KEY or OPENROUTER_API_KEY.")
        base_url = os.getenv("SAGE_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("SAGE_OPENROUTER_MODEL", args.model)
        client = OpenAI(api_key=api_key, base_url=base_url)

        correct = 0
        total = 0
        print("\n" + "=" * 60)
        print(f"When2Call {args.eval_mode.upper()} evaluation")
        print(f"Model: {model}")
        print(f"Split: {args.split}")
        print(f"Rows: {len(rows)}")
        print("=" * 60)
        for row in rows:
            question = row.get("question")
            if not question:
                continue
            options = _get_mcq_options(row)
            if args.eval_mode == "mcq" and options:
                prompt = _build_mcq_prompt(question, options)
                correct_idx = _get_mcq_correct_index(row, options)
            else:
                prompt = _when2call_label_prompt(question)
                label = _get_correct_label(row)
                correct_idx = None
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1,
            }
            if args.eval_mode == "logprob":
                params["logprobs"] = True
                params["top_logprobs"] = 3
            response = client.chat.completions.create(**params)
            prediction = ""
            if args.eval_mode == "logprob":
                prediction = _select_logprob_label(response.model_dump())
            if not prediction:
                content = response.choices[0].message.content or ""
                prediction = _parse_label_choice(content)
            if options and correct_idx is not None:
                pred_idx = None
                if prediction and prediction[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    pred_idx = ord(prediction[0].upper()) - ord("A")
                if pred_idx is not None and pred_idx == correct_idx:
                    correct += 1
            else:
                if prediction == label:
                    correct += 1
            total += 1
            if args.print_each:
                truth_display = ""
                if options and correct_idx is not None:
                    truth_display = f"{chr(ord('A') + correct_idx)}"
                else:
                    truth_display = label or ""
                print(f"[{total}] pred={prediction or '∅'} truth={truth_display} question={question[:120]}")
            elif total % 10 == 0:
                print(f"Processed {total}/{len(rows)}")

        acc = correct / total if total else 0.0
        print("\n" + "=" * 60)
        print(f"When2Call {args.eval_mode.upper()} Accuracy: {acc:.4f} ({correct}/{total})")
        print("=" * 60)
        return

    llm = TTSLLMClient(
        base_url=args.service_url,
        model=args.model,
        tts_strategy=args.tts_strategy,
        tts_budget=args.tts_budget,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Use hybrid constraint extractor for better domain refinement
    constraint_extractor = HybridConstraintExtractor(llm=llm, ambiguity_threshold=0.5)

    for row in rows:
        tool_payloads = row.get("orig_tools") if args.use_orig_tools else row.get("tools")
        if not tool_payloads:
            continue

        tool_schemas = []
        for tool_str in tool_payloads:
            tool_json = json.loads(tool_str)
            tool_schemas.append(_parse_tool_schema(tool_json, ParameterDomain, ToolSchema))

        tool_call_raw = row.get("answers", {}).get("tool_call")
        if not tool_call_raw:
            continue
        truth = _parse_tool_call(tool_call_raw, ToolCall)

        question_asker = GroundTruthQuestionAsker(truth=truth)
        tool_schemas_dict = {tool.name: tool for tool in tool_schemas}
        
        # Create uncertainty propagator for this example
        structured_weight = AGENT_CONFIG.get("structured_uncertainty_weight", AGENT_CONFIG.get("structured_weight", 0.7))
        if os.getenv("SAGE_DISABLE_PROPAGATION") == "1":
            uncertainty_propagator = None
        else:
            uncertainty_propagator = create_sage_propagator(
                structured_weight=structured_weight,
                llm_weight=1.0 - structured_weight,
            )
        
        # Build tool executor with proper return type for v2/v3
        if args.use_v2 or args.use_v3:
            tool_registry = {
                tool.name: lambda _args: ExecutionResult(success=True, output={"ok": True})
                for tool in tool_schemas
            }
        else:
            tool_registry = _build_tool_registry(tool_schemas)
        
        # Create dependencies (works for v1, v2, and v3)
        deps_kwargs = {
            "tool_schemas": tool_schemas_dict,
            "candidate_generator": LLMBackedCandidateGenerator(llm),
            "question_generator": LLMBackedQuestionGenerator(llm),
            "question_asker": question_asker,
            "tool_executor": ToolRegistryExecutor(tool_registry),
            "config": SageAgentConfig(
                max_questions=args.max_questions,
                redundancy_weight=args.redundancy_weight,
                tau_execute=args.tau_exec,
                alpha=args.alpha,
            ),
            "constraint_extractor": constraint_extractor,
            "uncertainty_propagator": uncertainty_propagator,
        }

        # Add v3-specific components
        if args.use_v3:
            from sage_agent.core.advanced_reasoning import UncertaintyDecomposer
            deps_kwargs["uncertainty_decomposer"] = UncertaintyDecomposer(num_samples=5)

        deps = GraphDeps(**deps_kwargs)
        graph = build_graph(deps).compile()

        # Create initial state (v2/v3 have a helper function, v1 uses inline dict)
        if (args.use_v2 or args.use_v3) and create_initial_state is not None:
            initial_state = create_initial_state(
                user_input=row.get("question", ""),
                tool_schemas=tool_schemas_dict,
            )
        else:
            initial_domains = {tool.name: dict(tool.parameters) for tool in tool_schemas}
            initial_state = {
                "user_input": row.get("question", ""),
                "observations": [],
                "candidates": [],
                "probabilities": [],
                "best_candidate_index": 0,
                "questions": [],
                "best_question": None,
                "best_score": 0.0,
                "aspect_counts": {},
                "domains": initial_domains,
                "steps": 0,
                "attempts": 0,
                "uncertainty": 1.0,
                "llm_uncertainty": 0.5,
                "combined_uncertainty": 1.0,
                "status": "pending",
                "result": None,
                "error": None,
            }

        result_state = graph.invoke(
            initial_state, {"recursion_limit": args.recursion_limit}
        )
        predictions.append(result_state.get("result") or ToolCall("", {}))
        ground_truths.append(truth)
        question_counts.append(question_asker.count)
        
        # Track confidence (1 - uncertainty) for calibration metrics
        combined_unc = result_state.get("combined_uncertainty", result_state.get("uncertainty", 0.5))
        confidence_scores.append(1.0 - combined_unc)
        uncertainty_scores.append(combined_unc)
        
        if args.paper_style_accuracy:
            correct_label = _get_correct_label(row)
            if isinstance(correct_label, str):
                label_counts[correct_label] = label_counts.get(correct_label, 0) + 1
                predicted_label = _paper_label(result_state)
                paper_correct.append(predicted_label == correct_label)

        if args.print_each:
            pred = predictions[-1]
            status = result_state.get("status")
            error = result_state.get("error")
            struct_unc = result_state.get("uncertainty", "N/A")
            llm_unc = result_state.get("llm_uncertainty", "N/A")
            print("uuid:", row.get("uuid"))
            print("question:", row.get("question"))
            print("pred:", pred.tool_name, pred.arguments)
            print("truth:", truth.tool_name, truth.arguments)
            print("questions:", question_asker.count, "status:", status, "error:", error)
            print(f"uncertainty: struct={struct_unc}, llm={llm_unc}, combined={combined_unc:.3f}")
            if args.paper_style_accuracy:
                print("paper_label:", _paper_label(result_state), "correct_answer:", _get_correct_label(row))
            if uncertainty_propagator.num_steps > 0:
                print(f"propagated: {uncertainty_propagator.accumulated_uncertainty:.3f} ({uncertainty_propagator.num_steps} steps)")

            # v3-specific metrics
            if args.use_v3:
                epistemic = result_state.get("epistemic_uncertainty", "N/A")
                aleatoric = result_state.get("aleatoric_uncertainty", "N/A")
                num_samples = result_state.get("num_samples", 1)
                agreement = result_state.get("sample_agreement", "N/A")
                trajectory_unc = result_state.get("trajectory_uncertainty", "N/A")
                print(f"v3 uncertainty: epistemic={epistemic}, aleatoric={aleatoric}")
                print(f"v3 resampling: {num_samples} samples, agreement={agreement}")
                print(f"v3 trajectory: {trajectory_unc}")
                if result_state.get("should_reflect"):
                    print(f"v3 reflexion: triggered ({result_state.get('reflection_trigger')})")
                if result_state.get("warning"):
                    print(f"⚠️  v3 soft escalation: {result_state['warning']}")
                    if result_state.get("confidence_score"):
                        print(f"   confidence: {result_state['confidence_score']:.2f}")

            print("-" * 60)

    # Compute standard metrics
    metrics = evaluate_metrics(predictions, ground_truths, question_counts)
    
    # Compute extended metrics with calibration
    extended_metrics = evaluate_extended_metrics(
        predictions, ground_truths, question_counts, confidence_scores
    )
    
    # Compute uncertainty-aware accuracy
    confident_acc, abstention_rate, selective_coverage = compute_uncertainty_aware_accuracy(
        predictions, ground_truths, uncertainty_scores, threshold=0.5
    )
    
    print("\n" + "=" * 60)
    print("SAGE-Agent Evaluation Results (When2Call)")
    print("=" * 60)
    print(f"Rows evaluated: {len(predictions)}")
    print()
    print("Standard Metrics:")
    print(f"  Coverage rate:        {metrics.coverage_rate:.4f}")
    print(f"  Tool match rate:      {metrics.tool_match_rate:.4f}")
    print(f"  Parameter match rate: {metrics.parameter_match_rate:.4f}")
    print(f"  Avg questions:        {metrics.avg_questions:.2f}")
    print()
    print("Uncertainty-Aware Metrics:")
    print(f"  Confident accuracy:   {confident_acc:.4f} (accuracy on low-uncertainty predictions)")
    print(f"  Abstention rate:      {abstention_rate:.4f} (fraction rejected due to high uncertainty)")
    print(f"  Selective coverage:   {selective_coverage:.4f} (correct predictions / total)")

    if args.paper_style_accuracy and paper_correct:
        paper_acc = sum(1 for ok in paper_correct if ok) / float(len(paper_correct))
        print()
        print("Paper-Style Accuracy:")
        print(f"  Label accuracy:       {paper_acc:.4f} (tool_call/request_for_info/cannot_answer)")
        if label_counts:
            print(f"  Label distribution:   {label_counts}")
    
    if extended_metrics.calibration:
        cal = extended_metrics.calibration
        print()
        print("Calibration Metrics:")
        print(f"  ECE (Expected Cal. Error):  {cal.ece:.4f} (lower is better)")
        print(f"  MCE (Max Cal. Error):       {cal.mce:.4f}")
        print(f"  Brier Score:                {cal.brier_score:.4f} (lower is better)")
        print()
        print("  Reliability Diagram (bin_accuracy | bin_confidence | count):")
        for i, (acc, conf, cnt) in enumerate(zip(cal.bin_accuracies, cal.bin_confidences, cal.bin_counts)):
            if cnt > 0:
                print(f"    Bin {i}: {acc:.3f} | {conf:.3f} | {cnt}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
