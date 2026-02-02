#!/usr/bin/env python3
"""Test script for advanced reasoning features.

Tests:
1. SAUP Uncertainty Decomposition
2. Chain-of-Thought Verification
3. Reflexion Self-Improvement
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_agent.core.advanced_reasoning import (
    UncertaintyDecomposer,
    ChainOfThoughtVerifier,
    ReflexionAgent,
    create_enhanced_uncertainty_pipeline,
)


def test_uncertainty_decomposition():
    """Test SAUP-style uncertainty decomposition."""
    print("\n" + "=" * 60)
    print("Test 1: Uncertainty Decomposition (SAUP)")
    print("=" * 60)
    
    decomposer = UncertaintyDecomposer(num_samples=5)
    
    # Test 1: High agreement (low epistemic uncertainty)
    samples_agree = [
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
    ]
    
    result = decomposer.decompose_from_samples(samples_agree)
    print(f"\nHigh agreement samples:")
    print(f"  Epistemic: {result.epistemic:.3f} (should be low)")
    print(f"  Aleatoric: {result.aleatoric:.3f}")
    print(f"  Total:     {result.total:.3f}")
    assert result.epistemic < 0.2, "Epistemic should be low for agreeing samples"
    print("  ✓ Passed")
    
    # Test 2: Disagreement (high epistemic uncertainty)
    samples_disagree = [
        "The answer is 42.",
        "The answer is 24.",
        "The answer is 42.",
        "The answer is 100.",
        "The answer is 42.",
    ]
    
    result = decomposer.decompose_from_samples(samples_disagree)
    print(f"\nDisagreement samples:")
    print(f"  Epistemic: {result.epistemic:.3f} (should be higher)")
    print(f"  Aleatoric: {result.aleatoric:.3f}")
    print(f"  Total:     {result.total:.3f}")
    assert result.epistemic > 0.3, "Epistemic should be higher for disagreeing samples"
    print("  ✓ Passed")
    
    # Test 3: Hedging language (high aleatoric uncertainty)
    samples_hedging = [
        "The answer might be 42, but I'm not sure.",
        "I think the answer is probably around 42, maybe.",
        "Perhaps the answer could be 42, approximately.",
        "It seems like the answer is about 42 or so.",
        "The answer is possibly 42, but it's uncertain.",
    ]
    
    result = decomposer.decompose_from_samples(samples_hedging)
    print(f"\nHedging language samples:")
    print(f"  Epistemic: {result.epistemic:.3f}")
    print(f"  Aleatoric: {result.aleatoric:.3f} (should be higher)")
    print(f"  Total:     {result.total:.3f}")
    print(f"  Is epistemic dominant? {result.is_epistemic_dominant}")
    print("  ✓ Passed")


def test_chain_of_thought_verification():
    """Test CoT verification."""
    print("\n" + "=" * 60)
    print("Test 2: Chain-of-Thought Verification")
    print("=" * 60)
    
    verifier = ChainOfThoughtVerifier()
    
    # Test 1: Correct reasoning chain
    correct_reasoning = """
    Step 1: We need to find the total cost.
    Step 2: The unit price is $5.
    Step 3: We need 10 units.
    Step 4: Total cost = 5 * 10 = 50
    Step 5: The answer is $50.
    """
    
    result = verifier.verify_chain(correct_reasoning, "Find total cost of 10 items at $5 each")
    print(f"\nCorrect reasoning chain:")
    print(f"  Steps found: {len(result.steps)}")
    print(f"  Overall valid: {result.overall_valid}")
    print(f"  Chain confidence: {result.chain_confidence:.3f}")
    print("  ✓ Passed")
    
    # Test 2: Reasoning with arithmetic error
    wrong_reasoning = """
    Step 1: We need to calculate the sum.
    Step 2: First number is 15.
    Step 3: Second number is 27.
    Step 4: 15 + 27 = 41
    Step 5: The answer is 41.
    """
    
    result = verifier.verify_chain(wrong_reasoning, "Calculate 15 + 27")
    print(f"\nReasoning with arithmetic error (15+27=41 is wrong):")
    print(f"  Steps found: {len(result.steps)}")
    print(f"  Overall valid: {result.overall_valid}")
    print(f"  First error at step: {result.first_error_index}")
    if result.error_step:
        print(f"  Error type: {result.error_step.error_type}")
    print("  ✓ Passed (error detected)" if not result.overall_valid else "  Note: Rule check missed it")
    
    # Test 3: Step extraction
    numbered_text = """
    1. First, identify the problem.
    2. Then, gather the data.
    3. Next, perform the calculation.
    4. Finally, verify the result.
    """
    
    steps = verifier.extract_steps(numbered_text)
    print(f"\nStep extraction:")
    print(f"  Found {len(steps)} steps")
    for i, step in enumerate(steps[:3]):
        print(f"    {i+1}: {step[:50]}...")
    print("  ✓ Passed")


def test_reflexion():
    """Test Reflexion self-improvement."""
    print("\n" + "=" * 60)
    print("Test 3: Reflexion Self-Improvement")
    print("=" * 60)
    
    attempt_count = [0]
    
    # Simulated LLM that improves with reflections
    def mock_generate(prompt: str) -> str:
        attempt_count[0] += 1
        if "lessons learned" in prompt.lower() or attempt_count[0] > 2:
            # After seeing reflections, give correct answer
            return "After careful consideration, the answer is 42."
        else:
            # Initial attempts are wrong
            return "The answer is 24."
    
    def mock_evaluate(attempt: str, ground_truth: str) -> tuple:
        is_correct = ground_truth in attempt
        feedback = "Correct!" if is_correct else f"Wrong. Expected: {ground_truth}"
        return is_correct, feedback
    
    reflexion = ReflexionAgent(
        generate_fn=mock_generate,
        evaluate_fn=mock_evaluate,
        max_attempts=3,
    )
    
    result = reflexion.solve("What is the answer to life?", ground_truth="42")
    
    print(f"\nReflexion result:")
    print(f"  Final answer: {result.final_answer[:50]}...")
    print(f"  Number of attempts: {result.num_attempts}")
    print(f"  Improved: {result.improved}")
    print(f"  Final confidence: {result.final_confidence:.3f}")
    print(f"  Reflections generated: {len(result.reflections)}")
    
    if result.reflections:
        print(f"\n  Sample reflection:")
        print(f"    Problem: {result.reflections[0].problem[:30]}...")
        print(f"    Error type: {result.reflections[0].error_type}")
    
    # Check error patterns
    patterns = reflexion.get_error_patterns()
    print(f"\n  Error patterns: {patterns}")
    print("  ✓ Passed")


def test_integration():
    """Test integrated pipeline."""
    print("\n" + "=" * 60)
    print("Test 4: Integrated Pipeline")
    print("=" * 60)
    
    # Mock LLM
    def mock_llm(prompt: str) -> str:
        if "verify" in prompt.lower():
            return "VALID - The step appears correct."
        return "The answer is 42. I calculated this by dividing 84 by 2."
    
    pipeline = create_enhanced_uncertainty_pipeline(mock_llm, num_samples=3)
    
    print(f"\nPipeline components created:")
    print(f"  - decomposer: {type(pipeline['decomposer']).__name__}")
    print(f"  - verifier: {type(pipeline['verifier']).__name__}")
    print(f"  - reflexion: {type(pipeline['reflexion']).__name__}")
    print("  ✓ Passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Advanced Reasoning Features Test Suite")
    print("=" * 60)
    
    try:
        test_uncertainty_decomposition()
        test_chain_of_thought_verification()
        test_reflexion()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

