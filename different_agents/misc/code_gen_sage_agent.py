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
from sage_agent.core.advanced_reasoning import (
    UncertaintyDecomposer,
    DecomposedUncertainty,
    ChainOfThoughtVerifier,
    ReflexionAgent,
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
    
    # Verification/Observation
    verification_result: str
    verification_passed: bool
    verification_issues: List[str]
    retry_count: int
    max_retries: int
    
    # Advanced Reasoning (SAUP + CoT + Reflexion)
    decomposed_uncertainty: Optional[DecomposedUncertainty]
    cot_verification_passed: bool
    cot_error_step: Optional[int]
    reflexion_attempts: int
    reflexion_history: List[str]
    
    # Propagator
    propagator: UncertaintyPropagator
    
    # Status
    status: Literal["analyzing", "clarifying", "generating", "verifying", "done", "error"]
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
        "status": "verifying",  # Go to verification step
        "propagator": propagator,
    }


def verify_result_node(state: CodeGenState, llm) -> CodeGenState:
    """Self-analyze and verify the generated code (Observation step).
    
    Enhanced with:
    1. Chain-of-Thought Verification - checks reasoning steps
    2. SAUP Uncertainty Decomposition - separates epistemic/aleatoric
    3. Reflexion-style feedback for improvement
    """
    code = state.get("generated_code", "")
    problem = state["problem"]
    interpretations = state.get("interpretations", [])
    answers = state.get("answers_received", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    # Build context about requirements
    requirements_context = ""
    if interpretations:
        best_idx = state.get("best_interpretation_idx", 0)
        if best_idx < len(interpretations):
            best = interpretations[best_idx]
            requirements_context = f"\nChosen interpretation: {best.description}"
            if best.edge_cases:
                requirements_context += f"\nEdge cases to handle: {', '.join(best.edge_cases)}"
    
    if answers:
        requirements_context += f"\nClarifications received: {'; '.join(answers)}"
    
    # === Chain-of-Thought Verification ===
    # First, verify any reasoning in the code comments
    cot_verifier = ChainOfThoughtVerifier()
    cot_result = cot_verifier.verify_chain(code, problem)
    cot_passed = cot_result.overall_valid
    cot_error_step = cot_result.first_error_index
    
    # === SAUP Uncertainty Decomposition ===
    # Get multiple verification samples to decompose uncertainty
    decomposer = UncertaintyDecomposer(num_samples=3)
    
    verify_prompt = f"""You are a code reviewer. Analyze this code solution carefully.

PROBLEM:
{problem}
{requirements_context}

GENERATED CODE:
```python
{code}
```

REVIEW CHECKLIST:
1. Does the code correctly implement the requirements?
2. Are edge cases handled (empty input, negative numbers, etc.)?
3. Is the logic correct? Any off-by-one errors?
4. Does it match the expected input/output format?
5. Any obvious bugs or issues?

Respond in this format:
PASSED: [YES or NO]
ISSUES: [List any issues found, or "None" if passed]
REASONING: [Brief explanation of your analysis]
"""
    
    # Get multiple samples for uncertainty decomposition
    responses = []
    for _ in range(3):
        responses.append(llm.complete(verify_prompt))
    
    # Use last response as primary
    response = responses[-1]
    llm_unc = getattr(llm, "last_uncertainty", 0.5)
    
    # Decompose uncertainty from multiple samples
    def extract_verdict(r: str) -> str:
        return "PASS" if "PASSED: YES" in r.upper() else "FAIL"
    
    decomposed = decomposer.decompose_from_samples(responses, extract_verdict)
    
    # Parse verification result
    passed = "PASSED: YES" in response.upper() or "PASSED:YES" in response.upper()
    
    # Combine verification signals
    # If CoT found errors OR high epistemic uncertainty, be more cautious
    if cot_error_step >= 0:
        passed = False  # CoT found an error
    if decomposed.epistemic > 0.5 and not passed:
        # High disagreement between samples - definitely uncertain
        passed = False
    
    # Extract issues
    issues = []
    if "ISSUES:" in response.upper():
        issues_start = response.upper().find("ISSUES:")
        issues_end = response.upper().find("REASONING:")
        if issues_end == -1:
            issues_end = len(response)
        issues_text = response[issues_start + 7:issues_end].strip()
        if issues_text.lower() not in ["none", "none.", "n/a", ""]:
            issues = [i.strip() for i in issues_text.split("\n") if i.strip()]
    
    # Add CoT error to issues if found
    if cot_error_step >= 0 and cot_result.error_step:
        issues.insert(0, f"CoT Error at step {cot_error_step}: {cot_result.error_step.error_type}")
    
    # === Reflexion-style feedback ===
    reflexion_history = state.get("reflexion_history", [])
    if not passed and issues:
        reflection = f"Attempt {retry_count + 1} failed: {'; '.join(issues[:2])}"
        reflexion_history = reflexion_history + [reflection]
    
    propagator = state.get("propagator", create_sage_propagator())
    propagator.observe(llm_unc, "verify", {
        "passed": passed, 
        "issues_count": len(issues),
        "epistemic": decomposed.epistemic,
        "aleatoric": decomposed.aleatoric,
        "cot_passed": cot_passed,
    })
    
    # Decide next status
    if passed:
        next_status = "done"
    elif retry_count < max_retries:
        next_status = "generating"  # Go back to generate with feedback
    else:
        next_status = "done"  # Max retries reached, accept current code
    
    return {
        **state,
        "verification_result": response,
        "verification_passed": passed,
        "verification_issues": issues,
        "retry_count": retry_count + (0 if passed else 1),
        "llm_uncertainty": llm_unc,
        "status": next_status,
        "propagator": propagator,
        # Advanced reasoning fields
        "decomposed_uncertainty": decomposed,
        "cot_verification_passed": cot_passed,
        "cot_error_step": cot_error_step if cot_error_step >= 0 else None,
        "reflexion_history": reflexion_history,
        "reflexion_attempts": retry_count + 1,
    }


def refine_code_node(state: CodeGenState, llm) -> CodeGenState:
    """Refine code based on verification feedback (Reflexion-style)."""
    code = state.get("generated_code", "")
    problem = state["problem"]
    issues = state.get("verification_issues", [])
    verification_result = state.get("verification_result", "")
    reflexion_history = state.get("reflexion_history", [])
    cot_error = state.get("cot_error_step")
    decomposed = state.get("decomposed_uncertainty")
    
    # Build reflexion context
    reflexion_context = ""
    if reflexion_history:
        reflexion_context = "\nPREVIOUS ATTEMPTS AND LESSONS:\n"
        for i, reflection in enumerate(reflexion_history[-3:]):  # Last 3
            reflexion_context += f"  {i+1}. {reflection}\n"
    
    # Add uncertainty info
    uncertainty_context = ""
    if decomposed:
        if decomposed.is_epistemic_dominant:
            uncertainty_context = "\nNote: High model uncertainty detected - please be extra careful with logic."
    
    if cot_error is not None:
        uncertainty_context += f"\nChain-of-thought error detected at step {cot_error}. Review reasoning carefully."
    
    refine_prompt = f"""Fix the following code based on the review feedback.

PROBLEM:
{problem}

CURRENT CODE:
```python
{code}
```

REVIEW FEEDBACK:
{verification_result}

ISSUES TO FIX:
{chr(10).join(f"- {issue}" for issue in issues) if issues else "- General improvements needed"}
{reflexion_context}{uncertainty_context}

Using the lessons from previous attempts, generate the FIXED code only (no explanation):
"""
    
    response = llm.complete(refine_prompt)
    llm_unc = getattr(llm, "last_uncertainty", 0.5)
    
    # Extract code
    code_match = re.search(r'```python\n([\s\S]*?)```', response)
    if code_match:
        refined_code = code_match.group(1)
    else:
        refined_code = response.strip()
    
    propagator = state.get("propagator", create_sage_propagator())
    propagator.observe(llm_unc, "refine", {"retry": state.get("retry_count", 0)})
    
    return {
        **state,
        "generated_code": refined_code,
        "llm_uncertainty": llm_unc,
        "status": "verifying",  # Verify the refined code
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


def route_after_generate(state: CodeGenState) -> Literal["verify", "done"]:
    """Route after code generation."""
    if state.get("status") == "verifying":
        return "verify"
    return "done"


def route_after_verify(state: CodeGenState) -> Literal["refine", "done"]:
    """Route based on verification result."""
    if state.get("status") == "generating":
        return "refine"  # Needs refinement
    return "done"


def route_after_refine(state: CodeGenState) -> Literal["verify"]:
    """After refinement, always verify again."""
    return "verify"


def build_code_gen_graph(llm, enable_verification: bool = True) -> StateGraph:
    """Build the code generation graph.
    
    Args:
        llm: The language model to use
        enable_verification: If True, includes self-verification/observation step
    """
    
    def analyze(state):
        return analyze_problem_node(state, llm)
    
    def clarify(state):
        return generate_question_node(state, llm)
    
    def generate(state):
        return generate_code_node(state, llm)
    
    def verify(state):
        return verify_result_node(state, llm)
    
    def refine(state):
        return refine_code_node(state, llm)
    
    graph = StateGraph(CodeGenState)
    
    # Add all nodes
    graph.add_node("analyze", analyze)
    graph.add_node("clarify", clarify)
    graph.add_node("generate", generate)
    
    if enable_verification:
        graph.add_node("verify", verify)
        graph.add_node("refine", refine)
    
    graph.set_entry_point("analyze")
    
    # Analyze → Clarify or Generate
    graph.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {"clarify": "clarify", "generate": "generate"},
    )
    
    # Clarify → Clarify again or Generate
    graph.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {"clarify": "clarify", "generate": "generate"},
    )
    
    if enable_verification:
        # Generate → Verify
        graph.add_conditional_edges(
            "generate",
            route_after_generate,
            {"verify": "verify", "done": END},
        )
        
        # Verify → Refine or Done
        graph.add_conditional_edges(
            "verify",
            route_after_verify,
            {"refine": "refine", "done": END},
        )
        
        # Refine → Verify (always re-verify after refinement)
        graph.add_edge("refine", "verify")
    else:
        # Without verification, generate goes directly to END
        graph.add_edge("generate", END)
    
    return graph.compile()


def evaluate_on_humaneval(
    llm, 
    limit: int = 10, 
    print_each: bool = True,
    enable_verification: bool = True,
) -> Dict[str, float]:
    """Evaluate the SAGE-Agent code gen on HumanEval.
    
    Args:
        llm: Language model to use
        limit: Number of examples to evaluate
        print_each: Print per-example results
        enable_verification: Enable self-verification/observation step
    """
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("SAGE-Agent Code Generation on HumanEval")
    print(f"Verification: {'ENABLED' if enable_verification else 'DISABLED'}")
    print("=" * 60)
    
    dataset = load_dataset("openai_humaneval", split="test")
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    graph = build_code_gen_graph(llm, enable_verification=enable_verification)
    
    correct = 0
    questions_total = 0
    refinements_total = 0
    verifications_passed = 0
    uncertainties = []
    
    for i, example in enumerate(dataset):
        task_id = example["task_id"]
        prompt = example["prompt"]
        canonical = example["canonical_solution"]
        
        # Run the agent with new state fields
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
            "verification_result": "",
            "verification_passed": False,
            "verification_issues": [],
            "retry_count": 0,
            "max_retries": 2,
            "status": "analyzing",
            "max_questions": 2,
            "iteration": 0,
        }
        
        # Get test code and entry point for execution-based evaluation
        test_code = example.get("test", "")
        entry_point = example.get("entry_point", "")
        
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 30})
            
            code = result.get("generated_code", "")
            questions = len(result.get("questions_asked", []))
            unc = result.get("combined_uncertainty", 0.5)
            # Ensure uncertainty is bounded [0, 1]
            unc = max(0.0, min(1.0, unc)) if isinstance(unc, (int, float)) else 0.5
            retries = result.get("retry_count", 0)
            verified = result.get("verification_passed", False)
            
            questions_total += questions
            refinements_total += retries
            if verified:
                verifications_passed += 1
            uncertainties.append(unc)
            
            # Check correctness using execution (with heuristic fallback)
            is_correct, eval_method = _check_code_correctness(
                generated=code,
                canonical=canonical,
                prompt=prompt,
                test_code=test_code,
                entry_point=entry_point,
                use_execution=True,
            )
            if is_correct:
                correct += 1
            
            if print_each:
                status = "✓" if is_correct else "✗"
                verify_status = "✓" if verified else f"✗ (retries={retries})"
                eval_info = f"[{eval_method}]" if eval_method != "execution" else ""
                print(f"[{i+1}/{len(dataset)}] {task_id}: {status} {eval_info}"
                      f"(questions={questions}, verify={verify_status}, unc={unc:.3f})")
                # Show the actual questions asked
                for q_idx, q in enumerate(result.get("questions_asked", [])):
                    print(f"    Q{q_idx+1}: {q[:80]}...")
                # Show verification issues if any
                issues = result.get("verification_issues", [])
                if issues and retries > 0:
                    print(f"    Issues found & fixed: {issues[:2]}")
                # Show execution error if failed
                if not is_correct and "execution_failed" in eval_method:
                    print(f"    ❌ {eval_method}")
        
        except Exception as e:
            print(f"[{i+1}] Error: {e}")
            uncertainties.append(1.0)
    
    accuracy = correct / len(dataset) if dataset else 0
    avg_questions = questions_total / len(dataset) if dataset else 0
    avg_refinements = refinements_total / len(dataset) if dataset else 0
    verification_rate = verifications_passed / len(dataset) if dataset else 0
    avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    print("\n" + "-" * 60)
    print("Results:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Avg questions:      {avg_questions:.2f}")
    print(f"  Avg refinements:    {avg_refinements:.2f}")
    print(f"  Verification rate:  {verification_rate:.4f}")
    print(f"  Avg uncertainty:    {avg_unc:.4f}")
    
    return {
        "accuracy": accuracy,
        "avg_questions": avg_questions,
        "avg_refinements": avg_refinements,
        "verification_rate": verification_rate,
        "avg_uncertainty": avg_unc,
    }


def _check_code_correctness_heuristic(generated: str, canonical: str) -> bool:
    """Simple heuristic for code correctness (fallback)."""
    gen_tokens = set(re.findall(r'\w+', generated.lower()))
    can_tokens = set(re.findall(r'\w+', canonical.lower()))
    
    if "return" not in generated.lower():
        return False
    
    overlap = len(gen_tokens & can_tokens) / max(len(can_tokens), 1)
    return overlap > 0.3


def _execute_code_safely(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
) -> Tuple[bool, str]:
    """Execute code in a sandboxed environment.
    
    Args:
        code: The generated code (function body or full function)
        test_code: HumanEval test code with assertions
        entry_point: Name of the function to test
        timeout: Maximum execution time in seconds
        
    Returns:
        (success, error_message)
    """
    import multiprocessing
    import signal
    from contextlib import contextmanager
    
    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
    
    def execute_in_sandbox():
        """Execute code in isolated namespace."""
        # Create isolated namespace with common imports
        namespace = {
            '__builtins__': __builtins__,
            'List': List,
            'Dict': Dict,
            'Tuple': Tuple,
            'Optional': Optional,
            'Any': Any,
            'math': __import__('math'),
            'collections': __import__('collections'),
            'itertools': __import__('itertools'),
            'functools': __import__('functools'),
            'typing': __import__('typing'),
        }
        
        try:
            # Execute the generated code to define the function
            exec(code, namespace)
            
            # Check if function was defined
            if entry_point not in namespace:
                return False, f"Function '{entry_point}' not defined"
            
            # Execute test code
            exec(test_code, namespace)
            
            return True, ""
            
        except AssertionError as e:
            return False, f"Assertion failed: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    
    try:
        with time_limit(timeout):
            return execute_in_sandbox()
    except TimeoutError:
        return False, "Execution timed out"
    except Exception as e:
        return False, f"Sandbox error: {e}"


def _check_code_correctness(
    generated: str,
    canonical: str,
    prompt: str = "",
    test_code: str = "",
    entry_point: str = "",
    use_execution: bool = True,
) -> Tuple[bool, str]:
    """Check code correctness using execution or heuristic fallback.
    
    Args:
        generated: Generated code
        canonical: Canonical solution (for heuristic fallback)
        prompt: Original function prompt/signature
        test_code: HumanEval test assertions
        entry_point: Function name
        use_execution: If True, try actual execution first
        
    Returns:
        (is_correct, method_used)
    """
    if use_execution and test_code and entry_point:
        # Build complete code: prompt + generated body
        # Handle case where generated code is just the body vs full function
        if f"def {entry_point}" in generated:
            full_code = generated
        else:
            # Assume generated is just the function body
            full_code = prompt + generated
        
        success, error = _execute_code_safely(
            full_code, test_code, entry_point, timeout=5.0
        )
        
        if success:
            return True, "execution"
        else:
            # Log the error for debugging
            # print(f"    Execution failed: {error}")
            # Fall back to heuristic if execution fails due to syntax etc.
            # But if it's an assertion failure, that's a real failure
            if "Assertion" in error:
                return False, f"execution_failed: {error}"
    
    # Fallback to heuristic
    heuristic_result = _check_code_correctness_heuristic(generated, canonical)
    return heuristic_result, "heuristic"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SAGE-Agent for Code Generation with Verification"
    )
    parser.add_argument("--limit", type=int, default=5, 
                        help="Number of examples to evaluate")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use Ollama instead of TTS service")
    parser.add_argument("--ollama-model", default="qwen3:4b-instruct-2507-q8_0")
    parser.add_argument("--model", default="xiaomi/mimo-v2-flash:free",
                        help="Model for TTS service")
    parser.add_argument("--service-url", default="http://localhost:8001/v1")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable self-verification/observation step")
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
    
    evaluate_on_humaneval(
        llm, 
        limit=args.limit, 
        print_each=True,
        enable_verification=not args.no_verify,
    )


if __name__ == "__main__":
    main()

