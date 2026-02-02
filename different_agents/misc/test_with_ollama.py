#!/usr/bin/env python3
"""
Quick integration test using Ollama (without TTS service).
Tests the full SAGE-Agent flow with the improvements.

Run: python examples/test_with_ollama.py
      python examples/test_with_ollama.py --use-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Test SAGE-Agent with Ollama")
    parser.add_argument("--use-v2", action="store_true", help="Use enhanced v2 graph")
    parser.add_argument("--model", default="qwen3:4b-instruct-2507-q8_0", help="Ollama model")
    return parser.parse_args()

from sage_agent import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    ParameterDomain,
    SageAgentConfig,
    ToolCall,
    ToolRegistryExecutor,
    ToolSchema,
    create_sage_propagator,
)
from sage_agent.core.constraints import HybridConstraintExtractor, SimpleConstraintExtractor
from sage_agent.core.types import ExecutionResult
from examples.ollama_client import OllamaClient


class AutoQuestionAsker:
    """Auto-answers questions based on ground truth for testing."""
    def __init__(self, truth: ToolCall):
        self.truth = truth
        self.count = 0
    
    def ask(self, question) -> str:
        self.count += 1
        if not getattr(question, "aspects", None):
            return ""
        values = []
        for aspect in question.aspects:
            if aspect.tool_name != self.truth.tool_name:
                continue
            value = self.truth.arguments.get(aspect.param_name)
            if value is not None:
                values.append(str(value))
        return "; ".join(values) if values else "I'm not sure"


def test_sage_agent_with_ollama(use_v2: bool = False, model: str = "qwen3:4b-instruct-2507-q8_0"):
    print("\n" + "=" * 60)
    print(f"SAGE-Agent Integration Test with Ollama {'(v2)' if use_v2 else '(v1)'}")
    print("=" * 60)
    
    # Import appropriate graph version
    if use_v2:
        from examples.langgraph_sage_agent_v2 import (
            GraphDeps, build_graph, create_initial_state, CONFIG
        )
        print("Using enhanced v2 graph")
    else:
        from examples.langgraph_sage_agent import GraphDeps, build_graph, CONFIG
        create_initial_state = None
        print("Using standard v1 graph")
    
    # Define tool schema
    tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "dest": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "date": ParameterDomain.from_values(["2024-01-15", "2024-01-16", "2024-01-17"]),
        },
        required=frozenset({"origin", "dest", "date"}),
    )
    
    # Ground truth for auto-answering
    ground_truth = ToolCall(
        tool_name="book_flight",
        arguments={"origin": "NYC", "dest": "LAX", "date": "2024-01-15"}
    )
    
    # Create Ollama client
    print(f"\nConnecting to Ollama with model: {model}...")
    llm = OllamaClient(model=model, verbose=False)
    
    # Test LLM connection
    test_response = llm.complete("Say 'OK' if you can read this.")
    print(f"LLM test response: {test_response[:50]}...")
    
    # Create uncertainty propagator
    structured_weight = CONFIG.get("structured_uncertainty_weight", CONFIG.get("structured_weight", 0.7))
    uncertainty_propagator = create_sage_propagator(
        structured_weight=structured_weight,
        llm_weight=1.0 - structured_weight,
    )
    
    # Create auto question asker
    question_asker = AutoQuestionAsker(ground_truth)
    
    # Tool executor - v2 needs ExecutionResult, v1 uses dict
    if use_v2:
        tool_registry = {
            "book_flight": lambda args: ExecutionResult(success=True, output={"booked": args})
        }
    else:
        tool_registry = {
            "book_flight": lambda args: {"status": "booked", "args": args}
        }
    
    # Build graph dependencies
    deps = GraphDeps(
        tool_schemas={tool.name: tool},
        candidate_generator=LLMBackedCandidateGenerator(llm),
        question_generator=LLMBackedQuestionGenerator(llm),
        question_asker=question_asker,
        tool_executor=ToolRegistryExecutor(tool_registry),
        config=SageAgentConfig(max_questions=4, tau_execute=0.85, alpha=0.1),
        constraint_extractor=SimpleConstraintExtractor(),  # Use simple for Ollama (faster)
        uncertainty_propagator=uncertainty_propagator,
    )
    
    # Build and compile graph
    print("\nBuilding SAGE-Agent graph...")
    graph = build_graph(deps).compile()
    
    # Test scenarios
    test_cases = [
        "Book me a flight from NYC to LAX",  # Mostly specified
        "I need to fly to LAX",  # Missing origin and date
    ]
    
    for i, user_input in enumerate(test_cases):
        print(f"\n{'─' * 60}")
        print(f"Test Case {i + 1}: {user_input}")
        print("─" * 60)
        
        # Reset propagator and question asker for each test
        uncertainty_propagator.reset()
        question_asker.count = 0
        
        # Create initial state (v2 has helper, v1 uses dict)
        if use_v2 and create_initial_state is not None:
            initial_state = create_initial_state(
                user_input=user_input,
                tool_schemas={tool.name: tool},
            )
        else:
            initial_state = {
                "user_input": user_input,
                "observations": [],
                "candidates": [],
                "probabilities": [],
                "best_candidate_index": 0,
                "questions": [],
                "best_question": None,
                "best_score": 0.0,
                "aspect_counts": {},
                "domains": {tool.name: dict(tool.parameters)},
                "steps": 0,
                "attempts": 0,
                "uncertainty": 1.0,
                "llm_uncertainty": 0.5,
                "combined_uncertainty": 1.0,
                "status": "pending",
                "result": None,
                "error": None,
            }
        
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 50})
            
            print(f"Status: {result['status']}")
            print(f"Questions asked: {question_asker.count}")
            
            if result.get("result"):
                pred = result["result"]
                print(f"Prediction: {pred.tool_name}({pred.arguments})")
                print(f"Ground truth: {ground_truth.tool_name}({ground_truth.arguments})")
                
                # Check correctness
                tool_match = pred.tool_name == ground_truth.tool_name
                param_match = all(
                    pred.arguments.get(k) == v 
                    for k, v in ground_truth.arguments.items()
                )
                print(f"Tool match: {'✓' if tool_match else '✗'}")
                print(f"Param match: {'✓' if param_match else '✗'}")
            
            print(f"Structured uncertainty: {result.get('uncertainty', 'N/A')}")
            print(f"Combined uncertainty: {result.get('combined_uncertainty', 'N/A')}")
            print(f"Propagated uncertainty: {uncertainty_propagator.accumulated_uncertainty:.4f}")
            print(f"Propagation steps: {uncertainty_propagator.num_steps}")
            
            if result.get("error"):
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    test_sage_agent_with_ollama(use_v2=args.use_v2, model=args.model)

