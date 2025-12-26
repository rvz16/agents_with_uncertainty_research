from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

from .belief import BeliefState
from .constraints import SimpleConstraintExtractor
from .evpi import compute_evpi
from .types import (
    Aspect,
    CandidateGenerator,
    ConstraintExtractor,
    ErrorQuestionGenerator,
    ExecutionResult,
    Question,
    QuestionAsker,
    QuestionGenerator,
    ToolCall,
    ToolCallCandidate,
    ToolExecutor,
    ToolSchema,
    UNK,
)


@dataclass(frozen=True)
class SageAgentConfig:
    max_questions: int = 6
    redundancy_weight: float = 0.5
    tau_execute: float = 0.85
    alpha: float = 0.1
    epsilon: float = 1e-4


@dataclass
class AgentStep:
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    question: Optional[Question]
    score: Optional[float]
    response: Optional[str]


@dataclass
class AgentResult:
    tool_call: Optional[ToolCall]
    execution_result: ExecutionResult
    steps: List[AgentStep] = field(default_factory=list)


class SageAgent:
    def __init__(
        self,
        tool_schemas: Iterable[ToolSchema],
        candidate_generator: CandidateGenerator,
        question_generator: QuestionGenerator,
        question_asker: QuestionAsker,
        tool_executor: ToolExecutor,
        constraint_extractor: Optional[ConstraintExtractor] = None,
        error_question_generator: Optional[ErrorQuestionGenerator] = None,
        config: Optional[SageAgentConfig] = None,
    ) -> None:
        self.tool_schemas = {tool.name: tool for tool in tool_schemas}
        self.candidate_generator = candidate_generator
        self.question_generator = question_generator
        self.question_asker = question_asker
        self.tool_executor = tool_executor
        self.constraint_extractor = constraint_extractor or SimpleConstraintExtractor()
        self.error_question_generator = error_question_generator
        self.config = config or SageAgentConfig()
        self._base_domains = {
            tool.name: {k: v for k, v in tool.parameters.items()} for tool in tool_schemas
        }
        self._belief = self._new_belief()

    def _new_belief(self) -> BeliefState:
        return BeliefState(
            domains=copy.deepcopy(self._base_domains),
            epsilon=self.config.epsilon,
        )

    def run(self, user_input: str) -> AgentResult:
        self._belief = self._new_belief()
        observations: List[str] = []
        steps: List[AgentStep] = []
        aspect_counts: Dict[Aspect, int] = defaultdict(int)
        t = 0

        while True:
            candidates = self.candidate_generator.generate_candidates(
                user_input, observations, self.tool_schemas
            )
            if not candidates:
                raise ValueError("Candidate generator returned no candidates")
            for candidate in candidates:
                if candidate.tool_name not in self.tool_schemas:
                    raise ValueError(
                        f"Unknown tool in candidate: {candidate.tool_name}"
                    )

            weights = [
                self._belief.candidate_weight(candidate, self.tool_schemas[candidate.tool_name])
                for candidate in candidates
            ]
            probabilities = self._belief.normalize(weights)
            max_prob = max(probabilities)
            best_idx = probabilities.index(max_prob)
            best_candidate = candidates[best_idx]

            if max_prob >= self.config.tau_execute or t >= self.config.max_questions:
                return self._execute_candidate(best_candidate, steps)

            questions = self.question_generator.generate_questions(
                user_input, candidates, observations, self.tool_schemas
            )
            if not questions:
                return self._execute_candidate(best_candidate, steps)

            scored_questions = []
            for question in questions:
                evpi = compute_evpi(candidates, probabilities, question.aspects)
                cost = self._redundancy_cost(question, aspect_counts)
                scored_questions.append((evpi - cost, question))

            scored_questions.sort(key=lambda item: item[0], reverse=True)
            best_score, best_question = scored_questions[0]
            if best_score < self.config.alpha * max_prob:
                if not self._has_required_unknowns(best_candidate):
                    return self._execute_candidate(best_candidate, steps)

            response = self.question_asker.ask(best_question)
            observations.append(response)
            for aspect in best_question.aspects:
                aspect_counts[aspect] += 1
                self._update_domain(aspect, response)

            steps.append(
                AgentStep(
                    candidates=candidates,
                    probabilities=probabilities,
                    question=best_question,
                    score=best_score,
                    response=response,
                )
            )
            t += 1

    def _update_domain(self, aspect: Aspect, response: str) -> None:
        tool = self.tool_schemas[aspect.tool_name]
        current = self._belief.domains[tool.name][aspect.param_name]
        updated = self.constraint_extractor.update_domain(current, response)
        self._belief.domains[tool.name][aspect.param_name] = updated
        if tool.domain_refiner is not None:
            refined = tool.domain_refiner.refine(
                tool,
                self._belief.domains[tool.name],
                self._current_assignments(tool.name),
            )
            self._belief.domains[tool.name] = dict(refined)

    def _current_assignments(self, tool_name: str) -> Mapping[str, object]:
        assignments: Dict[str, object] = {}
        for param, domain in self._belief.domains[tool_name].items():
            if domain.is_finite() and domain.size() == 1:
                assignments[param] = next(iter(domain.values or []))
            else:
                assignments[param] = UNK
        return assignments

    def _redundancy_cost(self, question: Question, aspect_counts: Mapping[Aspect, int]) -> float:
        cost = 0.0
        for aspect in question.aspects:
            cost += float(aspect_counts.get(aspect, 0))
        return cost * self.config.redundancy_weight

    def _execute_candidate(
        self, candidate: ToolCallCandidate, steps: List[AgentStep]
    ) -> AgentResult:
        tool_schema = self.tool_schemas[candidate.tool_name]
        tool_call = ToolCall(tool_name=candidate.tool_name, arguments=candidate.arguments)
        try:
            tool_schema.validate_call(tool_call.arguments)
        except ValueError as exc:
            result = ExecutionResult(success=False, error=str(exc))
            return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

        result = self.tool_executor.execute(tool_call)
        if result.success or self.error_question_generator is None:
            return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

        error_question = self.error_question_generator.generate_error_question(
            result.error or "Execution failed", tool_call, self.tool_schemas
        )
        if error_question is None:
            return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

        response = self.question_asker.ask(error_question)
        steps.append(
            AgentStep(
                candidates=[candidate],
                probabilities=[1.0],
                question=error_question,
                score=None,
                response=response,
            )
        )
        for aspect in error_question.aspects:
            self._update_domain(aspect, response)
        return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

    def _has_required_unknowns(self, candidate: ToolCallCandidate) -> bool:
        tool_schema = self.tool_schemas[candidate.tool_name]
        for param in tool_schema.required:
            if candidate.arguments.get(param, UNK) == UNK:
                return True
        return False
