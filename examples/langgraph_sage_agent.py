"""LangGraph wrapper around SAGE-Agent decision loop.

This uses LangGraph nodes to mirror the SAGE flow:
- generate candidates
- generate questions
- score EVPI + cost
- ask/update
- execute
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Literal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langgraph.graph import StateGraph, END

from sage_agent import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    ParameterDomain,
    Question,
    SageAgentConfig,
    ToolCall,
    ToolCallCandidate,
    ToolRegistryExecutor,
    ToolSchema,
    UNK,
)
from sage_agent.core.belief import BeliefState
from sage_agent.core.constraints import SimpleConstraintExtractor
from sage_agent.core.evpi import compute_evpi
from sage_agent.core.types import ToolExecutor
from examples.ollama_client import OllamaClient
from examples.tts_llm_client import TTSLLMClient


class AgentState(TypedDict):
    user_input: str
    observations: List[str]
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    best_candidate_index: int
    questions: List[Question]
    best_question: Optional[Question]
    best_score: float
    aspect_counts: Dict[str, int]
    domains: Dict[str, Dict[str, ParameterDomain]]
    steps: int
    attempts: int
    uncertainty: float
    status: Literal["pending", "asking", "executed", "done", "escalated"]
    result: Optional[ToolCall]
    error: Optional[str]


@dataclass
class GraphDeps:
    tool_schemas: Dict[str, ToolSchema]
    candidate_generator: LLMBackedCandidateGenerator
    question_generator: LLMBackedQuestionGenerator
    question_asker: "QuestionAsker"
    tool_executor: ToolExecutor
    config: SageAgentConfig
    constraint_extractor: SimpleConstraintExtractor


class QuestionAsker:
    def ask(self, question: Question) -> str:
        raise NotImplementedError


class InteractiveQuestionAsker(QuestionAsker):
    def ask(self, question: Question) -> str:
        print(f"\nAgent question: {question.text}")
        return input("Your answer: ").strip()

USE_TTS = False # should be True by default
TTS_CONFIG = {
    "service_url": "http://localhost:8001/v1",
    "model": "openai/gpt-4o-mini",
    "tts_budget": 8,
}

CONFIG = {
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
}


def build_graph(deps: GraphDeps) -> StateGraph:
    graph = StateGraph(AgentState)

    def generate_candidates_node(state: AgentState) -> AgentState:
        candidates = deps.candidate_generator.generate_candidates(
            state["user_input"], state["observations"], deps.tool_schemas
        )
        if not candidates:
            return {**state, "error": "No candidates", "status": "done"}

        belief = BeliefState(domains=state["domains"], epsilon=deps.config.epsilon)
        weights = [
            belief.candidate_weight(c, deps.tool_schemas[c.tool_name]) for c in candidates
        ]
        probs = belief.normalize(weights)
        best_idx = probs.index(max(probs))
        uncertainty = 1.0 - (max(probs) if probs else 0.0)
        llm = getattr(deps.candidate_generator, "llm", None)
        llm_uncertainty = getattr(llm, "last_uncertainty", None)
        if llm_uncertainty is not None:
            uncertainty = llm_uncertainty
        return {
            **state,
            "candidates": candidates,
            "probabilities": probs,
            "best_candidate_index": best_idx,
            "uncertainty": uncertainty,
        }

    def generate_questions_node(state: AgentState) -> AgentState:
        questions = deps.question_generator.generate_questions(
            state["user_input"],
            state["candidates"],
            state["observations"],
            deps.tool_schemas,
        )
        best_score = float("-inf")
        best_question = None
        for question in questions:
            evpi = compute_evpi(state["candidates"], state["probabilities"], question.aspects)
            cost = _redundancy_cost(question, state["aspect_counts"], deps.config.redundancy_weight)
            score = evpi - cost
            if score > best_score:
                best_score = score
                best_question = question
        return {**state, "questions": questions, "best_question": best_question, "best_score": best_score}

    def ask_question_node(state: AgentState) -> AgentState:
        question = state["best_question"]
        if question is None:
            return {**state, "status": "executed"}
        response = deps.question_asker.ask(question)
        new_observations = list(state["observations"]) + [response]
        new_domains = _update_domains(
            deps,
            state["domains"],
            question,
            response,
        )
        new_counts = dict(state["aspect_counts"])
        for aspect in question.aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            new_counts[key] = new_counts.get(key, 0) + 1
        return {
            **state,
            "observations": new_observations,
            "domains": new_domains,
            "aspect_counts": new_counts,
            "steps": state["steps"] + 1,
            "attempts": state["attempts"] + 1,
        }

    def execute_node(state: AgentState) -> AgentState:
        if state["error"]:
            return {**state, "status": "done"}
        candidate = state["candidates"][state["best_candidate_index"]]
        tool_schema = deps.tool_schemas[candidate.tool_name]
        try:
            tool_schema.validate_call(candidate.arguments)
        except ValueError as exc:
            return {**state, "error": str(exc), "status": "done"}
        tool_call = ToolCall(tool_name=candidate.tool_name, arguments=candidate.arguments)
        result = deps.tool_executor.execute(tool_call)
        if not result.success:
            return {**state, "error": result.error or "Execution failed", "status": "done"}
        return {**state, "result": tool_call, "status": "done"}

    def route_after_questions(state: AgentState) -> Literal["ask", "execute", "escalate"]:
        if state["error"]:
            return "execute"
        candidate = state["candidates"][state["best_candidate_index"]]
        if _has_required_unknowns(candidate, deps.tool_schemas[candidate.tool_name]):
            return "ask"
        if state["uncertainty"] <= CONFIG["uncertainty_threshold"]:
            return "execute"
        if state["attempts"] >= CONFIG["max_attempts"]:
            return "escalate"
        max_prob = max(state["probabilities"]) if state["probabilities"] else 0.0
        if max_prob >= deps.config.tau_execute:
            return "execute"
        if state["steps"] >= deps.config.max_questions:
            return "execute"
        if not state["questions"]:
            return "execute"
        if state["best_question"] is None:
            return "execute"
        if state["best_score"] < deps.config.alpha * max_prob:
            return "execute"
        return "ask"

    def escalate_node(state: AgentState) -> AgentState:
        return {**state, "status": "escalated"}

    graph.add_node("generate_candidates", generate_candidates_node)
    graph.add_node("generate_questions", generate_questions_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("execute", execute_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("generate_candidates")
    graph.add_edge("generate_candidates", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        route_after_questions,
        {"ask": "ask_question", "execute": "execute", "escalate": "escalate"},
    )
    graph.add_edge("ask_question", "generate_candidates")
    graph.add_edge("execute", END)
    graph.add_edge("escalate", END)

    return graph


def _redundancy_cost(question: Question, counts: Dict[str, int], weight: float) -> float:
    cost = 0.0
    for aspect in question.aspects:
        key = f"{aspect.tool_name}:{aspect.param_name}"
        cost += float(counts.get(key, 0))
    return cost * weight


def _has_required_unknowns(candidate: ToolCallCandidate, tool_schema: ToolSchema) -> bool:
    for param in tool_schema.required:
        if candidate.arguments.get(param, UNK) == UNK:
            return True
    return False


def _update_domains(
    deps: GraphDeps,
    domains: Dict[str, Dict[str, ParameterDomain]],
    question: Question,
    response: str,
) -> Dict[str, Dict[str, ParameterDomain]]:
    updated = {tool: dict(params) for tool, params in domains.items()}
    for aspect in question.aspects:
        tool = deps.tool_schemas[aspect.tool_name]
        current = updated[tool.name][aspect.param_name]
        refined = deps.constraint_extractor.update_domain(current, response)
        updated[tool.name][aspect.param_name] = refined
        if tool.domain_refiner is not None:
            updated[tool.name] = dict(tool.domain_refiner.refine(tool, updated[tool.name], {}))
    return updated


def main() -> None:
    tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS"]),
            "dest": ParameterDomain.from_values(["SFO", "LAX"]),
            "date": ParameterDomain.from_values(["March 3", "March 4"]),
        },
        required=frozenset({"origin", "dest", "date"}),
    )

    if USE_TTS:
        llm = TTSLLMClient(
            base_url=TTS_CONFIG["service_url"],
            model=TTS_CONFIG["model"],
            tts_budget=TTS_CONFIG["tts_budget"],
        )
    else:
        model = "qwen3:4b-instruct-2507-q8_0"
        llm = OllamaClient(model, verbose=True)

    deps = GraphDeps(
        tool_schemas={tool.name: tool},
        candidate_generator=LLMBackedCandidateGenerator(llm),
        question_generator=LLMBackedQuestionGenerator(llm),
        question_asker=InteractiveQuestionAsker(),
        tool_executor=ToolRegistryExecutor({"book_flight": lambda args: {"ok": True, "args": args}}),
        config=SageAgentConfig(max_questions=4, tau_execute=10.0, alpha=0.1),
        constraint_extractor=SimpleConstraintExtractor(),
    )

    graph = build_graph(deps).compile()

    initial_domains = {tool.name: dict(tool.parameters)}
    initial_state: AgentState = {
        "user_input": "Book me a flight from NYC.",
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
        "status": "pending",
        "result": None,
        "error": None,
    }

    result = graph.invoke(initial_state)
    print("\nFinal status:", result["status"])
    print("Result tool call:", result["result"])
    print("Error:", result["error"])


if __name__ == "__main__":
    main()
