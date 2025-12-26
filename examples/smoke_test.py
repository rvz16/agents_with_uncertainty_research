import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sage_agent import (
    ClarifyBenchSimulator,
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    ParameterDomain,
    Question,
    SageAgent,
    SageAgentConfig,
    SimulationScenario,
    ToolCall,
    ToolRegistryExecutor,
    ToolSchema,
)
from examples.ollama_client import OllamaClient


class InteractiveUser:
    def answer(self, question: Question, scenario_id=None):
        print(f"\nAgent question: {question.text}")
        return input("Your answer: ").strip()


def book_flight(args):
    return {"ok": True, "args": args}


tool = ToolSchema(
    name="book_flight",
    parameters={
        "origin": ParameterDomain.from_values(["NYC", "BOS"]),
        "dest": ParameterDomain.from_values(["SFO", "LAX"]),
        "date": ParameterDomain.from_values(["March 3", "March 4"]),
    },
    required=frozenset({"origin", "dest", "date"}),
)

MODEL_NAME = "qwen3:4b-instruct-2507-q8_0"
llm = OllamaClient(MODEL_NAME, verbose=True)

agent = SageAgent(
    tool_schemas=[tool],
    candidate_generator=LLMBackedCandidateGenerator(llm),
    question_generator=LLMBackedQuestionGenerator(llm),
    question_asker=None,
    tool_executor=ToolRegistryExecutor({"book_flight": book_flight}),
    config=SageAgentConfig(max_questions=4, tau_execute=10.0),
)

scenario = SimulationScenario(
    scenario_id="smoke",
    requests=["Book me a flight from NYC."],
    ground_truth=[ToolCall("book_flight", {"origin": "NYC", "dest": "SFO", "date": "March 3"})],
)

sim = ClarifyBenchSimulator(agent, InteractiveUser())
result = sim.run(scenario)
print(result.metrics)
