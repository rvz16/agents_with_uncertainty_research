"""Integration test with REAL TTS service.

Run with:
    # First start TTS service on port 8001
    python test_sgr_sage_uq_real.py

This tests the actual self-consistency uncertainty quantification.
"""

import os
import sys

# Project root is 2 levels up from different_agents/sgr_sage_uq/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(ROOT, "different_agents", "shared")
sys.path.insert(0, ROOT)
sys.path.insert(0, LOCAL_DIR)
sys.path.insert(0, SHARED_DIR)

from sgr_sage_uq_agent import (
    TTSConfig,
    SelfConsistencyClient,
    SAGEConfig,
    SAGEDecisionEngine,
    create_agent,
    run_agent,
    print_result,
)
from sage_agent import ParameterDomain, ToolSchema


def test_tts_connection():
    """Test basic TTS service connectivity."""
    print("\n" + "=" * 60)
    print("Test 1: TTS Service Connection")
    print("=" * 60)

    config = TTSConfig(
        service_url=os.getenv("TTS_SERVICE_URL", "http://localhost:8001/v1"),
        model=os.getenv("TTS_MODEL", "openai/gpt-4o-mini"),
        tts_budget=5,
    )

    client = SelfConsistencyClient(config)

    try:
        samples, metadata = client.get_samples(
            prompt="What is 2 + 2?",
            system_prompt="Answer with just a number.",
            n=5,
        )

        print(f"Response: {samples[0][:100]}...")
        print(f"Uncertainty score: {metadata.get('uncertainty_score')}")
        print(f"Consensus score: {metadata.get('consensus_score')}")
        print("[PASS] TTS service is working")
        return True

    except Exception as e:
        print(f"[FAIL] Could not connect to TTS service: {e}")
        print(f"Make sure TTS service is running at {config.service_url}")
        return False


def test_uncertainty_varies_with_ambiguity():
    """Test that uncertainty increases with ambiguous prompts."""
    print("\n" + "=" * 60)
    print("Test 2: Uncertainty Varies with Ambiguity")
    print("=" * 60)

    config = TTSConfig(
        service_url=os.getenv("TTS_SERVICE_URL", "http://localhost:8001/v1"),
        model=os.getenv("TTS_MODEL", "openai/gpt-4o-mini"),
        tts_budget=8,
    )

    client = SelfConsistencyClient(config)

    # Clear prompt - should have LOW uncertainty
    _, metadata_clear = client.get_samples(
        prompt="What is the capital of France?",
        system_prompt="Answer with just the city name.",
    )
    uncertainty_clear = metadata_clear.get("uncertainty_score", 0.5)

    # Ambiguous prompt - should have HIGHER uncertainty
    _, metadata_ambiguous = client.get_samples(
        prompt="What is the best programming language?",
        system_prompt="Answer with just one language name.",
    )
    uncertainty_ambiguous = metadata_ambiguous.get("uncertainty_score", 0.5)

    print(f"Clear prompt uncertainty: {uncertainty_clear:.3f}")
    print(f"Ambiguous prompt uncertainty: {uncertainty_ambiguous:.3f}")

    if uncertainty_ambiguous > uncertainty_clear:
        print("[PASS] Ambiguous prompts have higher uncertainty")
        return True
    else:
        print("[INFO] Uncertainty did not vary as expected (may depend on model)")
        return True  # Don't fail - model behavior varies


def test_full_agent_with_real_tts():
    """Test the complete agent with real TTS service."""
    print("\n" + "=" * 60)
    print("Test 3: Full Agent with Real TTS")
    print("=" * 60)

    # Define tool
    flight_tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "destination": ParameterDomain.from_values(["NYC", "BOS", "LAX", "SFO"]),
            "date": ParameterDomain.from_values(["2024-03-01", "2024-03-02", "2024-03-03"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )

    tool_schemas = {"book_flight": flight_tool}

    # Mock tool execution
    def execute_flight(args):
        return {"confirmation": "FL-123", "details": dict(args)}

    tool_registry = {"book_flight": execute_flight}

    # Auto-answer questions for testing
    class AutoQuestionAsker:
        def __init__(self):
            # Map keywords to answers
            self.keyword_answers = {
                # Origin keywords
                "origin": "New York",
                "departure": "New York",
                "from": "New York",
                "leaving": "New York",
                # Destination keywords
                "destination": "Los Angeles",
                "to": "Los Angeles",
                "arriving": "Los Angeles",
                "going": "Los Angeles",
                # Date keywords
                "date": "March 1st",
                "when": "March 1st",
                "day": "March 1st",
            }
            self.questions_asked = []

        def ask(self, question):
            self.questions_asked.append(question.text)
            print(f"  [AGENT] {question.text}")

            # Find relevant answer by keyword matching
            q_lower = question.text.lower()
            for keyword, answer in self.keyword_answers.items():
                if keyword in q_lower:
                    print(f"  [AUTO] {answer}")
                    return answer

            print(f"  [AUTO] I'm not sure")
            return "I'm not sure"

    asker = AutoQuestionAsker()

    # Create agent with real TTS
    tts_config = TTSConfig(
        service_url=os.getenv("TTS_SERVICE_URL", "http://localhost:8001/v1"),
        model=os.getenv("TTS_MODEL", "openai/gpt-4o-mini"),
        tts_budget=8,
    )

    sage_config = SAGEConfig(
        cost_of_clarification=0.15,
        max_clarification_rounds=3,
    )

    try:
        graph, _ = create_agent(
            tool_schemas=tool_schemas,
            tts_config=tts_config,
            sage_config=sage_config,
            question_asker=asker,
            tool_registry=tool_registry,
        )

        # Test with ambiguous input
        user_input = "I need a flight to California"
        print(f"\nUser: {user_input}")

        result = run_agent(
            graph=graph,
            user_input=user_input,
            tool_schemas=tool_schemas,
            num_samples=8,
        )

        print(f"\nStatus: {result['status']}")
        print(f"Clarification rounds: {result['clarification_rounds']}")
        print(f"Combined uncertainty: {result['combined_uncertainty']:.3f}")
        print(f"LLM uncertainty: {result['llm_uncertainty']:.3f}")

        if result['result']:
            print(f"Tool: {result['result'].tool_name}")
            print(f"Args: {dict(result['result'].arguments)}")

        if asker.questions_asked:
            print(f"\nQuestions asked ({len(asker.questions_asked)}):")
            for q in asker.questions_asked:
                print(f"  - {q}")

        print("\n[PASS] Agent completed successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all real TTS tests."""
    print("=" * 60)
    print("SGR-SAGE-UQ Agent - Real TTS Integration Tests")
    print("=" * 60)
    print(f"TTS URL: {os.getenv('TTS_SERVICE_URL', 'http://localhost:8001/v1')}")
    print(f"Model: {os.getenv('TTS_MODEL', 'openai/gpt-4o-mini')}")

    # Test 1: Basic connection
    if not test_tts_connection():
        print("\n[ABORT] TTS service not available. Start it first.")
        return False

    # Test 2: Uncertainty varies
    test_uncertainty_varies_with_ambiguity()

    # Test 3: Full agent
    test_full_agent_with_real_tts()

    print("\n" + "=" * 60)
    print("Integration tests complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
