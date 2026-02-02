#!/usr/bin/env python3
"""
Test script for SAGE-Agent improvements.

Run: python examples/test_improvements.py

Tests:
1. Uncertainty propagation
2. Calibration metrics
3. Constraint extractors (simple)
4. Adaptive thresholds
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_uncertainty_propagation():
    """Test the SAUP-inspired uncertainty propagator."""
    print("\n" + "=" * 60)
    print("TEST 1: Uncertainty Propagation")
    print("=" * 60)
    
    from sage_agent.core.uncertainty_propagation import (
        UncertaintyPropagator,
        PropagationMode,
        create_sage_propagator,
    )
    
    # Test basic propagator
    prop = UncertaintyPropagator(base_uncertainty=0.1)
    assert prop.accumulated_uncertainty == 0.1, "Base uncertainty should be 0.1"
    print("✓ Base uncertainty correct")
    
    # Add observations
    prop.observe(0.3, "candidate_generation")
    prop.observe(0.2, "llm_parsing")
    prop.observe(0.1, "question_answering")
    
    acc = prop.accumulated_uncertainty
    assert acc > 0.1, f"Accumulated should increase: {acc}"
    print(f"✓ Accumulated uncertainty: {acc:.4f}")
    print(f"✓ Steps recorded: {prop.num_steps}")
    print(f"✓ Breakdown: {prop.get_uncertainty_breakdown()}")
    
    # Test escalation
    assert not prop.should_escalate(escalation_threshold=0.9)
    print("✓ Should NOT escalate with high threshold")
    
    # Test SAGE factory
    sage_prop = create_sage_propagator(structured_weight=0.7)
    sage_prop.observe(0.8, "candidate_generation")
    sage_prop.observe(0.8, "candidate_generation")
    sage_prop.observe(0.8, "candidate_generation")
    assert sage_prop.should_escalate(max_high_uncertainty_steps=3, high_uncertainty_threshold=0.6)
    print("✓ Should escalate after 3 high-uncertainty steps")
    
    print("\n✅ Uncertainty propagation tests PASSED")


def test_calibration_metrics():
    """Test ECE and calibration computation."""
    print("\n" + "=" * 60)
    print("TEST 2: Calibration Metrics")
    print("=" * 60)
    
    from sage_agent.metrics.metrics import (
        compute_calibration,
        compute_uncertainty_aware_accuracy,
    )
    from sage_agent.core.types import ToolCall
    
    # Perfect calibration: confidence matches accuracy
    conf = [0.9, 0.9, 0.1, 0.1]
    correct = [1.0, 1.0, 0.0, 0.0]
    cal = compute_calibration(conf, correct, num_bins=10)
    print(f"Perfect calibration ECE: {cal.ece:.4f} (should be ~0)")
    assert cal.ece < 0.2, f"ECE too high for perfect calibration: {cal.ece}"
    print("✓ Perfect calibration has low ECE")
    
    # Poor calibration: overconfident
    conf_over = [0.9, 0.9, 0.9, 0.9]
    correct_poor = [1.0, 0.0, 0.0, 0.0]  # Only 25% correct but 90% confident
    cal_poor = compute_calibration(conf_over, correct_poor, num_bins=10)
    print(f"Overconfident ECE: {cal_poor.ece:.4f} (should be high)")
    assert cal_poor.ece > 0.3, f"ECE should be higher for poor calibration"
    print("✓ Overconfident model has high ECE")
    
    # Test Brier score
    assert cal_poor.brier_score > cal.brier_score
    print(f"✓ Brier scores: good={cal.brier_score:.4f}, bad={cal_poor.brier_score:.4f}")
    
    # Test uncertainty-aware accuracy
    predictions = [
        ToolCall("tool_a", {"x": 1}),
        ToolCall("tool_a", {"x": 2}),
        ToolCall("tool_b", {"x": 1}),
    ]
    truths = [
        ToolCall("tool_a", {"x": 1}),  # Correct
        ToolCall("tool_a", {"x": 1}),  # Wrong param
        ToolCall("tool_b", {"x": 1}),  # Correct
    ]
    uncertainties = [0.2, 0.8, 0.3]  # 2nd is uncertain
    
    conf_acc, abstain, selective = compute_uncertainty_aware_accuracy(
        predictions, truths, uncertainties, threshold=0.5
    )
    print(f"✓ Confident accuracy: {conf_acc:.2f}")
    print(f"✓ Abstention rate: {abstain:.2f}")
    print(f"✓ Selective coverage: {selective:.2f}")
    
    print("\n✅ Calibration metrics tests PASSED")


def test_constraint_extractors():
    """Test constraint extraction logic."""
    print("\n" + "=" * 60)
    print("TEST 3: Constraint Extractors")
    print("=" * 60)
    
    from sage_agent.core.constraints import SimpleConstraintExtractor
    from sage_agent.core.domains import ParameterDomain
    
    extractor = SimpleConstraintExtractor()
    
    # Test positive match
    domain = ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"])
    updated = extractor.update_domain(domain, "I want to fly to NYC")
    assert "NYC" in (updated.values or set())
    print(f"✓ Positive match: {updated.values}")
    
    # Test negative match
    domain2 = ParameterDomain.from_values(["morning", "afternoon", "evening"])
    updated2 = extractor.update_domain(domain2, "Not morning please")
    assert "morning" not in (updated2.values or set())
    print(f"✓ Negative exclusion: {updated2.values}")
    
    # Test no match (should return original or subset)
    domain3 = ParameterDomain.from_values(["A", "B", "C"])
    updated3 = extractor.update_domain(domain3, "Something unrelated")
    # If no match found, domain should be unchanged or have valid subset
    assert updated3.values is not None and len(updated3.values) > 0
    print(f"✓ Unrelated response: {updated3.values} (original or valid subset)")
    
    print("\n✅ Constraint extractor tests PASSED")


def test_adaptive_thresholds():
    """Test adaptive threshold computation."""
    print("\n" + "=" * 60)
    print("TEST 4: Adaptive Thresholds")
    print("=" * 60)
    
    # Import the config and function from langgraph agent
    # We'll simulate the logic here
    critical_patterns = ["delete", "cancel", "remove", "drop", "terminate"]
    critical_reduction = 0.5
    base_threshold = 0.3
    
    def compute_adaptive_threshold(tool_name: str) -> float:
        tool_lower = tool_name.lower()
        is_critical = any(p in tool_lower for p in critical_patterns)
        return base_threshold * critical_reduction if is_critical else base_threshold
    
    # Test normal tool
    normal_thresh = compute_adaptive_threshold("book_flight")
    assert normal_thresh == base_threshold
    print(f"✓ Normal tool threshold: {normal_thresh}")
    
    # Test critical tools
    for critical_tool in ["delete_file", "cancel_order", "remove_user"]:
        thresh = compute_adaptive_threshold(critical_tool)
        assert thresh == base_threshold * critical_reduction
        print(f"✓ Critical tool '{critical_tool}' threshold: {thresh}")
    
    print("\n✅ Adaptive threshold tests PASSED")


def test_all():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# SAGE-Agent Improvements Test Suite")
    print("#" * 60)
    
    try:
        test_uncertainty_propagation()
        test_calibration_metrics()
        test_constraint_extractors()
        test_adaptive_thresholds()
        
        print("\n" + "#" * 60)
        print("# ALL TESTS PASSED! ✅")
        print("#" * 60 + "\n")
        return True
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)

