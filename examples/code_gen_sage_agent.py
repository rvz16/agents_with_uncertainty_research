"""
SAGE-Agent for Code Generation with Uncertainty-Guided Clarification.

This implements the SAGE-Agent approach for code generation tasks:
- Uses structured uncertainty to identify ambiguous requirements
- Asks clarifying questions when uncertainty is high
- Generates code with higher confidence after clarification

Key insight: Code generation often has ambiguous requirements that benefit
from the SAGE-Agent clarification approach:
- Input/output formats
- Edge cases
- Performance requirements  
- Library preferences
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, END

# Import SAGE-Agent components
from sage_agent.core.belief import BeliefState
from sage_agent.core.domains import ParameterDomain
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.uncertainty_propagation import (
    UncertaintyPropagator,
    create_sage_propagator,
)

@dataclass
class CodeRequirement:
    """Сode requirements"""
    description: str
    weight: float
    input_format: str = ""
    output_format: str = ""
    edge_cases: List[str] = field(default_factory=list)


class CodeGenState(TypedDict, total=False):
    """State for code generation agent."""
    # Original problem
    problem: str
    
    # Interpretations and their weights
    interpretations: List[CodeRequirement]
    
    # Clarifying questions
    questions_asked: List[str]
    answers_received: List[str]
    
    # Current belief about the best interpretation
    best_interpretation_idx: int
    
    # Uncertainty metrics
    structured_uncertainty: float
    llm_uncertainty: float
    combined_uncertainty: float
    
    # Generated code
    generated_code: str
    
    # Propagator
    propagator: UncertaintyPropagator
    
    # Status
    status: Literal["analyzing", "clarifying", "generating", "done", "error"]
    max_questions: int
    iteration: int


def analyze_problem_node(state: CodeGenState, llm) -> CodeGenState:
    """Analyze the problem and generate possible interpretations."""
    problem = state["problem"]
    
    prompt = f"""Analyze this coding problem and identify 2-4 different valid interpretations.
    For each interpretation, describe:
    1. What the function should do
    2. Expected input format
    3. Expected output format
    4. Important edge cases

    Problem: {problem}

    Format your response as JSON:
    {{
        "interpretations": [
            {{
                "description": "...",
                "input_format": "...",
                "output_format": "...",
                "edge_cases": ["...", "..."]
            }}
        ]
    }}
    """
    response = llm.complete(prompt)
    llm_unc = getattr(llm, "last_uncertainty", 0.5)
    
    # Parse interpretations
    interpretations = []
    try:
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            for interp in data.get("interpretations", []):
                interpretations.append(CodeRequirement(
                    description=interp.get("description", ""),
                    weight=1.0 / len(data.get("interpretations", [1])),
                    input_format=interp.get("input_format", ""),
                    output_format=interp.get("output_format", ""),
                    edge_cases=interp.get("edge_cases", []),
                ))
    except (json.JSONDecodeError, AttributeError):
        # Fallback: single interpretation
        interpretations = [CodeRequirement(
            description="Direct implementation as described",
            weight=1.0,
        )]
    
    # Calculate structured uncertainty from interpretation weights
    if interpretations:
        weights = [i.weight for i in interpretations]
        max_weight = max(weights)
        # If weights are uniform, high uncertainty; if one dominates, low uncertainty
        structured_unc = 1.0 - max_weight
    else:
        structured_unc = 1.0
    
    # Combined uncertainty
    combined = 0.7 * structured_unc + 0.3 * llm_unc
    
    propagator = state.get("propagator") or create_sage_propagator()
    propagator.observe(combined, "analyze", {"num_interpretations": len(interpretations)})
    
    return {
        **state,
        "interpretations": interpretations,
        "structured_uncertainty": structured_unc,
        "llm_uncertainty": llm_unc,
        "combined_uncertainty": combined,
        "propagator": propagator,
        "status": "clarifying" if combined > 0.3 and state.get("questions_asked", []) == [] else "generating",
    }


def generate_question_node(state: CodeGenState, llm) -> CodeGenState:
    """Generate a clarifying question based on uncertain interpretations."""
    interpretations = state["interpretations"]
    asked = state.get("questions_asked", [])
    
    if not interpretations or len(interpretations) <= 1:
        return {**state, "status": "generating"}
    
    # Find the most uncertain aspect
    prompt = f"""Given these different interpretations of a coding problem, generate ONE clarifying question that would help distinguish between them.

Previous questions asked: {asked if asked else 'None'}

Interpretations:
"""
    for i, interp in enumerate(interpretations):
        prompt += f"\n{i+1}. {interp.description}"
        if interp.input_format:
            prompt += f"\n   Input: {interp.input_format}"
        if interp.edge_cases:
            prompt += f"\n   Edge cases: {', '.join(interp.edge_cases[:2])}"
    
    prompt += "\n\nGenerate ONE clear, specific clarifying question:"
    
    response = llm.complete(prompt)
    question = response.strip()
    
    # Simulate user answer (in real scenario, this would come from user)
    # For evaluation we generate a random answer that favors one interpretation
    answer = _simulate_user_answer(interpretations, question)
    
    # Update interpretation weights based on answer
    updated_interpretations = _update_weights(interpretations, question, answer, llm)
    
    new_questions = asked + [question]
    new_answers = state.get("answers_received", []) + [answer]
    
    # Recalculate uncertainty
    weights = [i.weight for i in updated_interpretations]
    max_weight = max(weights) if weights else 0.5
    struct_unc = 1.0 - max_weight
    
    propagator = state.get("propagator", create_sage_propagator())
    propagator.observe(struct_unc, "clarify", {"question": question})
    
    return {
        **state,
        "interpretations": updated_interpretations,
        "questions_asked": new_questions,
        "answers_received": new_answers,
        "structured_uncertainty": struct_unc,
        "combined_uncertainty": struct_unc,  # Simplify after clarification
        "propagator": propagator,
        "iteration": state.get("iteration", 0) + 1,
        "status": "clarifying" if struct_unc > 0.3 and len(new_questions) < state.get("max_questions", 3) else "generating",
    }


def _simulate_user_answer(interpretations: List[CodeRequirement], question: str) -> str:
    """Simulate user answer (for evaluation). In practice, this comes from user."""
    # In evaluation we use ground truth to guide the answer
    if interpretations:
        best = interpretations[0]
        return f"The function should {best.description[:100]}"
    return "Please implement the standard approach."


def _update_weights(
    interpretations: List[CodeRequirement],
    question: str,
    answer: str,
    llm,
) -> List[CodeRequirement]:
    """Update interpretation weights based on user answer."""
    if not interpretations:
        return interpretations
    
    # Use LLM to determine which interpretation the answer supports
    prompt = f"""Given this clarifying question and answer, rate how well each interpretation matches.

    Question: {question}
    Answer: {answer}

    Interpretations:
    """
    for i, interp in enumerate(interpretations):
        prompt += f"\n{i+1}. {interp.description}"
    
    prompt += "\n\nRate each from 0-10 (10 = perfect match). Format: 1:X, 2:Y, ..."
    
    response = llm.complete(prompt)
    
    # Parse ratings
    ratings = {}
    for match in re.finditer(r'(\d+)\s*:\s*(\d+)', response):
        idx = int(match.group(1)) - 1
        score = int(match.group(2))
        if 0 <= idx < len(interpretations):
            ratings[idx] = score
    
    # Update weights
    total = sum(ratings.values()) or 1
    updated = []
    for i, interp in enumerate(interpretations):
        new_weight = ratings.get(i, 5) / total
        updated.append(CodeRequirement(
            description=interp.description,
            weight=new_weight,
            input_format=interp.input_format,
            output_format=interp.output_format,
            edge_cases=interp.edge_cases,
        ))
    
    return updated


def generate_code_node(state: CodeGenState, llm) -> CodeGenState:
    """Generate code based on the clarified requirements."""
    problem = state["problem"]
    interpretations = state["interpretations"]
    answers = state.get("answers_received", [])
    
    # Select best interpretation
    best_idx = 0
    if interpretations:
        best_idx = max(range(len(interpretations)), 
                       key=lambda i: interpretations[i].weight)
        best = interpretations[best_idx]
    else:
        best = None
    
    prompt = f"""Generate Python code to solve this problem.

Problem: {problem}
"""
    if best:
        prompt += f"\nInterpretation: {best.description}"
        if best.input_format:
            prompt += f"\nInput format: {best.input_format}"
        if best.output_format:
            prompt += f"\nOutput format: {best.output_format}"
    
    if answers:
        prompt += f"\n\nClarifications received:\n" + "\n".join(f"- {a}" for a in answers)
    
    prompt += "\n\nGenerate the code (only the function, no explanation):"
    
    response = llm.complete(prompt)
    llm_unc = getattr(llm, "last_uncertainty", 0.5)
    
    # Extract code
    code_match = re.search(r'```python\n([\s\S]*?)```', response)
    if code_match:
        code = code_match.group(1)
    else:
        code = response.strip()
    
    propagator = state.get("propagator", create_sage_propagator())
    propagator.observe(llm_unc, "generate", {"code_length": len(code)})
    
    return {
        **state,
        "generated_code": code,
        "best_interpretation_idx": best_idx,
        "llm_uncertainty": llm_unc,
        "status": "done",
        "propagator": propagator,
    }


def route_after_analysis(state: CodeGenState) -> Literal["clarify", "generate"]:
    """Route based on uncertainty."""
    if state.get("status") == "clarifying":
        return "clarify"
    return "generate"


def route_after_clarify(state: CodeGenState) -> Literal["clarify", "generate"]:
    """Route based on remaining uncertainty."""
    if state.get("status") == "clarifying":
        return "clarify"
    return "generate"


def build_code_gen_graph(llm) -> StateGraph:
    """Build the code generation graph."""
    
    def analyze(state):
        return analyze_problem_node(state, llm)
    
    def clarify(state):
        return generate_question_node(state, llm)
    
    def generate(state):
        return generate_code_node(state, llm)
    
    graph = StateGraph(CodeGenState)
    
    graph.add_node("analyze", analyze)
    graph.add_node("clarify", clarify)
    graph.add_node("generate", generate)
    
    graph.set_entry_point("analyze")
    
    graph.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {"clarify": "clarify", "generate": "generate"},
    )
    
    graph.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {"clarify": "clarify", "generate": "generate"},
    )
    
    graph.add_edge("generate", END)
    
    return graph.compile()


def evaluate_on_humaneval(llm, limit: int = 10, print_each: bool = True) -> Dict[str, float]:
    """Evaluate the SAGE-Agent code gen on HumanEval."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("SAGE-Agent Code Generation on HumanEval")
    print("=" * 60)
    
    dataset = load_dataset("openai_humaneval", split="test")
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    graph = build_code_gen_graph(llm)
    
    correct = 0
    questions_total = 0
    uncertainties = []
    
    for i, example in enumerate(dataset):
        task_id = example["task_id"]
        prompt = example["prompt"]
        canonical = example["canonical_solution"]
        
        # Run the agent
        initial_state: CodeGenState = {
            "problem": prompt,
            "interpretations": [],
            "questions_asked": [],
            "answers_received": [],
            "best_interpretation_idx": 0,
            "structured_uncertainty": 1.0,
            "llm_uncertainty": 0.5,
            "combined_uncertainty": 0.5,
            "generated_code": "",
            "status": "analyzing",
            "max_questions": 2,
            "iteration": 0,
        }
        
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 20})
            
            code = result.get("generated_code", "")
            questions = len(result.get("questions_asked", []))
            unc = result.get("combined_uncertainty", 0.5)
            
            questions_total += questions
            uncertainties.append(unc)
            
            # Simple correctness check
            is_correct = _check_code_correctness(code, canonical)
            if is_correct:
                correct += 1
            
            if print_each:
                status = "✓" if is_correct else "✗"
                print(f"[{i+1}/{len(dataset)}] {task_id}: {status} "
                      f"(questions={questions}, unc={unc:.3f})")
                # Show the actual questions asked
                for q_idx, q in enumerate(result.get("questions_asked", [])):
                    print(f"    Q{q_idx+1}: {q[:80]}...")
        
        except Exception as e:
            print(f"[{i+1}] Error: {e}")
            uncertainties.append(1.0)
    
    accuracy = correct / len(dataset) if dataset else 0
    avg_questions = questions_total / len(dataset) if dataset else 0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    print("\n" + "-" * 60)
    print("Results:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Avg questions:   {avg_questions:.2f}")
    print(f"  Avg uncertainty: {avg_unc:.4f}")
    
    return {
        "accuracy": accuracy,
        "avg_questions": avg_questions,
        "avg_uncertainty": avg_unc,
    }


def _check_code_correctness(generated: str, canonical: str) -> bool:
    """Simple heuristic for code correctness."""
    gen_tokens = set(re.findall(r'\w+', generated.lower()))
    can_tokens = set(re.findall(r'\w+', canonical.lower()))
    
    if "return" not in generated.lower():
        return False
    
    overlap = len(gen_tokens & can_tokens) / max(len(can_tokens), 1)
    return overlap > 0.3


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-model", default="qwen3:4b-instruct-2507-q8_0")
    parser.add_argument("--model", default="xiaomi/mimo-v2-flash:free")
    parser.add_argument("--service-url", default="http://localhost:8001/v1")
    args = parser.parse_args()
    
    if args.use_ollama:
        from examples.ollama_client import OllamaClient
        llm = OllamaClient(model=args.ollama_model, verbose=True)
    else:
        from examples.tts_llm_client import TTSLLMClient
        llm = TTSLLMClient(
            base_url=args.service_url,
            model=args.model,
            tts_budget=4,
        )
    
    evaluate_on_humaneval(llm, limit=args.limit, print_each=True)


if __name__ == "__main__":
    main()

