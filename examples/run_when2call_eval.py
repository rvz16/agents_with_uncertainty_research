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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAGE-Agent on When2Call with llm-tts uncertainty."
    )
    parser.add_argument(
        "--sage-root",
        type=Path,
        default=Path(
            "/Users/victor/Documents/vs_files/research/article_implementation/"
            "agents_with_uncertainty_research"
        ),
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


def main() -> None:
    args = _parse_args()
    if not args.sage_root.exists():
        raise FileNotFoundError(f"Missing sage repo: {args.sage_root}")

    sys.path.insert(0, str(args.sage_root))

    from examples.tts_llm_client import TTSLLMClient
    
    # Import the appropriate graph version based on --use-v2 flag
    if args.use_v2:
        from examples.langgraph_sage_agent_v2 import (
            GraphDeps, 
            build_graph, 
            create_initial_state,
            CONFIG as AGENT_CONFIG,
        )
        print("Using enhanced v2 graph with error recovery and separate belief update")
    else:
        from examples.langgraph_sage_agent import (
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
        uncertainty_propagator = create_sage_propagator(
            structured_weight=structured_weight,
            llm_weight=1.0 - structured_weight,
        )
        
        # Build tool executor with proper return type for v2
        if args.use_v2:
            tool_registry = {
                tool.name: lambda _args: ExecutionResult(success=True, output={"ok": True})
                for tool in tool_schemas
            }
        else:
            tool_registry = _build_tool_registry(tool_schemas)
        
        # Create dependencies (works for both v1 and v2)
        deps = GraphDeps(
            tool_schemas=tool_schemas_dict,
            candidate_generator=LLMBackedCandidateGenerator(llm),
            question_generator=LLMBackedQuestionGenerator(llm),
            question_asker=question_asker,
            tool_executor=ToolRegistryExecutor(tool_registry),
            config=SageAgentConfig(
                max_questions=args.max_questions,
                redundancy_weight=args.redundancy_weight,
                tau_execute=args.tau_exec,
                alpha=args.alpha,
            ),
            constraint_extractor=constraint_extractor,
            uncertainty_propagator=uncertainty_propagator,
        )
        graph = build_graph(deps).compile()

        # Create initial state (v2 has a helper function, v1 uses inline dict)
        if args.use_v2 and create_initial_state is not None:
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
            if uncertainty_propagator.num_steps > 0:
                print(f"propagated: {uncertainty_propagator.accumulated_uncertainty:.3f} ({uncertainty_propagator.num_steps} steps)")
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
