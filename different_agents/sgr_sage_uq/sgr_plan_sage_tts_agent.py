"""SGR Plan → Arguments → Execute agent with TTS self-consistency + SAGE check.

Flow per step:
1) SGR plan generation (structured JSON).
2) Arguments selection via TTS self-consistency (pick lowest uncertainty).
3) SAGE decision pass (clarify vs act) with optional questions.
4) Execute tool, then move to next plan step.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

import sys

# Project root is 2 levels up from different_agents/sgr_sage_uq/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(REPO_ROOT, "different_agents", "shared")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, LOCAL_ROOT)
sys.path.insert(0, SHARED_DIR)

from langchain_core.messages import HumanMessage, SystemMessage

from llm_tts.integration import ChatTTS

from sage_agent import (
    GeneratePlanTool,
    LLMBackedQuestionGenerator,
    ParameterDomain,
    Question,
    ReasoningTool,
    ToolCall,
    ToolRegistryExecutor,
    ToolSchema,
)
from sage_agent.core.constraints import HybridConstraintExtractor
from sage_agent.core.types import Aspect, ConstraintExtractor, ToolExecutor

from sgr_sage_uq_agent import ActionSchema, SAGEConfig, SAGEDecisionEngine


class ToolSelectionSchema(BaseModel):
    tool_name: str = Field(description="Tool to call (must match available tools)")
    arguments: Dict[str, Any] = Field(description="Tool arguments (use <UNK> if unknown)")


@dataclass
class TTSConfig:
    service_url: str = "http://localhost:8001/v1"
    model: str = "openai/gpt-4o-mini"
    tts_strategy: str = "self_consistency"
    tts_budget: int = 8
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 120.0


@dataclass
class TTSChooser:
    config: TTSConfig
    system_prompt: str
    _llm: ChatTTS = field(init=False)

    def __post_init__(self) -> None:
        self._llm = ChatTTS(
            base_url=self.config.service_url,
            model=self.config.model,
            tts_strategy=self.config.tts_strategy,
            tts_budget=self.config.tts_budget,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )

    def best_of(self, prompt: str, attempts: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Return best response (lowest uncertainty) + all attempts."""
        attempts = max(1, attempts)
        results: List[Tuple[str, float]] = []
        for _ in range(attempts):
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]
            response = self._llm.invoke(messages)
            tts_meta = response.response_metadata.get("tts_metadata", {})
            uncertainty = float(tts_meta.get("uncertainty_score", 1.0))
            results.append((response.content, uncertainty))
        best = min(results, key=lambda item: item[1])
        return best[0], best[1], results


class InteractiveQuestionAsker:
    def ask(self, question: Question) -> str:
        print(f"\n[AGENT] {question.text}")
        return input("[USER] ").strip()


@dataclass
class SGRPlanSageTTSAgent:
    tool_schemas: Dict[str, ToolSchema]
    tool_executor: ToolExecutor
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    sage_config: SAGEConfig = field(default_factory=SAGEConfig)
    num_tts_attempts: int = 3
    question_asker: Optional[InteractiveQuestionAsker] = None
    constraint_extractor: Optional[ConstraintExtractor] = None

    def __post_init__(self) -> None:
        if self.question_asker is None:
            self.question_asker = InteractiveQuestionAsker()
        if self.constraint_extractor is None:
            llm = self._make_tts_llm()
            self.constraint_extractor = HybridConstraintExtractor(llm=llm, ambiguity_threshold=0.5)
        self._question_generator = LLMBackedQuestionGenerator(self._make_tts_llm())
        self._sage_engine = SAGEDecisionEngine(
            config=self.sage_config,
            tool_schemas=self.tool_schemas,
        )

    def _make_tts_llm(self):
        from tts_llm_client import TTSLLMClient

        return TTSLLMClient(
            base_url=self.tts_config.service_url,
            model=self.tts_config.model,
            tts_strategy=self.tts_config.tts_strategy,
            tts_budget=self.tts_config.tts_budget,
            temperature=self.tts_config.temperature,
            max_tokens=self.tts_config.max_tokens,
            timeout=self.tts_config.timeout,
        )

    def _make_plan(self, user_input: str) -> GeneratePlanTool:
        chooser = TTSChooser(
            config=self.tts_config,
            system_prompt=(
                "Return JSON only. "
                "Use the GeneratePlanTool schema with fields: "
                "reasoning, research_goal, planned_steps."
            ),
        )
        prompt = f"User request:\n{user_input}\n\nCreate a concise plan."
        content, _, _ = chooser.best_of(prompt, attempts=self.num_tts_attempts)
        return GeneratePlanTool.model_validate_json(self._extract_json(content))

    def _make_reasoning(
        self,
        user_input: str,
        plan: GeneratePlanTool,
        step_index: int,
        observations: Iterable[str],
        results: List[Dict[str, Any]],
    ) -> ReasoningTool:
        chooser = TTSChooser(
            config=self.tts_config,
            system_prompt=(
                "Return JSON only. "
                "Use the ReasoningTool schema."
            ),
        )
        planned_steps = plan.planned_steps
        remaining = planned_steps[step_index:]
        prompt = (
            "Fill ReasoningTool for the current step.\n"
            f"User request: {user_input}\n"
            f"Plan steps: {planned_steps}\n"
            f"Remaining steps: {remaining}\n"
            f"Observations: {list(observations)}\n"
            f"Results so far: {results}\n"
            "Return JSON only."
        )
        content, _, _ = chooser.best_of(prompt, attempts=self.num_tts_attempts)
        return ReasoningTool.model_validate_json(self._extract_json(content))

    def _select_candidates(
        self,
        step_text: str,
        observations: Iterable[str],
    ) -> Tuple[List[ActionSchema], Optional[float]]:
        tool_list = ", ".join(sorted(self.tool_schemas.keys()))
        chooser = TTSChooser(
            config=self.tts_config,
            system_prompt=(
                "Return JSON only. "
                "Schema: {tool_name: str, arguments: object}. "
                "tool_name must be one of: "
                f"[{tool_list}]. "
                "Use <UNK> for missing required parameters."
            ),
        )
        prompt = (
            "Pick the best tool call for the current plan step.\n"
            f"Step: {step_text}\n"
            f"Observations: {list(observations)}\n"
            "Return JSON only."
        )
        best_content, best_uncertainty, all_attempts = chooser.best_of(
            prompt, attempts=self.num_tts_attempts
        )

        actions: List[ActionSchema] = []
        for content, _unc in all_attempts:
            try:
                parsed = ToolSelectionSchema.model_validate_json(self._extract_json(content))
            except ValidationError:
                continue
            if parsed.tool_name not in self.tool_schemas:
                continue
            action = ActionSchema(
                tool_name=parsed.tool_name,
                arguments=parsed.arguments,
                parameter_uncertainties={},
            )
            actions.append(action)

        # Ensure best candidate is included
        try:
            parsed_best = ToolSelectionSchema.model_validate_json(self._extract_json(best_content))
            if parsed_best.tool_name in self.tool_schemas:
                actions.append(
                    ActionSchema(
                        tool_name=parsed_best.tool_name,
                        arguments=parsed_best.arguments,
                        parameter_uncertainties={},
                    )
                )
        except ValidationError:
            pass

        # Deduplicate by hash
        dedup: Dict[str, ActionSchema] = {}
        for action in actions:
            dedup[action.compute_hash()] = action
        return list(dedup.values()), best_uncertainty

    def _make_domains(self) -> Dict[str, Dict[str, ParameterDomain]]:
        return {name: dict(schema.parameters) for name, schema in self.tool_schemas.items()}

    def _update_domains(
        self,
        domains: Dict[str, Dict[str, ParameterDomain]],
        question: Question,
        response: str,
    ) -> Dict[str, Dict[str, ParameterDomain]]:
        updated = {tool: dict(params) for tool, params in domains.items()}
        for aspect in question.aspects:
            tool = self.tool_schemas.get(aspect.tool_name)
            if tool is None:
                continue
            if aspect.param_name not in updated.get(tool.name, {}):
                continue
            current = updated[tool.name][aspect.param_name]
            refined = self.constraint_extractor.update_domain(current, response)
            updated[tool.name][aspect.param_name] = refined
            if tool.domain_refiner is not None:
                updated[tool.name] = dict(tool.domain_refiner.refine(tool, updated[tool.name], {}))
        return updated

    def run(self, user_input: str) -> Dict[str, Any]:
        plan = self._make_plan(user_input)
        results = []

        for idx, step in enumerate(plan.planned_steps):
            observations: List[str] = []
            aspect_counts: Dict[str, int] = {}
            clarification_rounds = 0
            domains = self._make_domains()
            sgr_reasoning = self._make_reasoning(
                user_input=user_input,
                plan=plan,
                step_index=idx,
                observations=observations,
                results=results,
            )

            while True:
                hypotheses, llm_unc = self._select_candidates(step, observations)
                if not hypotheses:
                    results.append(
                        {
                            "step": step,
                            "sgr_reasoning": sgr_reasoning.model_dump(),
                            "status": "failed",
                            "error": "no candidates",
                        }
                    )
                    break

                questions = self._question_generator.generate_questions(
                    step,
                    [h.to_tool_call_candidate() for h in hypotheses],
                    observations,
                    self.tool_schemas,
                )

                metrics, best_action, clarify_request = self._sage_engine.decide(
                    hypotheses=hypotheses,
                    domains=domains,
                    questions=questions,
                    aspect_counts=aspect_counts,
                    clarification_rounds=clarification_rounds,
                    llm_uncertainty=llm_unc,
                )

                if metrics.chosen_action == "CLARIFY" and clarify_request is not None:
                    question = Question(
                        text=clarify_request.question,
                        aspects=tuple(
                            Aspect(tool_name=t, param_name=p)
                            for t, p in clarify_request.aspects
                        ),
                    )
                    response = self.question_asker.ask(question)
                    observations.append(response)
                    for aspect in question.aspects:
                        key = f"{aspect.tool_name}:{aspect.param_name}"
                        aspect_counts[key] = aspect_counts.get(key, 0) + 1
                    domains = self._update_domains(domains, question, response)
                    clarification_rounds += 1
                    if clarification_rounds >= self.sage_config.max_clarification_rounds:
                        results.append(
                            {
                                "step": step,
                                "sgr_reasoning": sgr_reasoning.model_dump(),
                                "status": "escalated",
                            }
                        )
                        break
                    continue

                if metrics.chosen_action == "ESCALATE":
                    results.append(
                        {
                            "step": step,
                            "sgr_reasoning": sgr_reasoning.model_dump(),
                            "status": "escalated",
                        }
                    )
                    break

                # ACT
                tool_call = ToolCall(
                    tool_name=best_action.tool_name,
                    arguments=best_action.arguments,
                )
                tool_schema = self.tool_schemas.get(tool_call.tool_name)
                if tool_schema is None:
                    results.append(
                        {
                            "step": step,
                            "sgr_reasoning": sgr_reasoning.model_dump(),
                            "status": "failed",
                            "error": f"unknown tool: {tool_call.tool_name}",
                        }
                    )
                    break
                try:
                    tool_schema.validate_call(tool_call.arguments)
                except ValueError as exc:
                    results.append(
                        {
                            "step": step,
                            "sgr_reasoning": sgr_reasoning.model_dump(),
                            "status": "failed",
                            "tool": tool_call.tool_name,
                            "arguments": dict(tool_call.arguments),
                            "error": str(exc),
                        }
                    )
                    break
                exec_result = self.tool_executor.execute(tool_call)
                results.append(
                    {
                        "step": step,
                        "sgr_reasoning": sgr_reasoning.model_dump(),
                        "status": "executed" if exec_result.success else "failed",
                        "tool": tool_call.tool_name,
                        "arguments": dict(tool_call.arguments),
                        "output": exec_result.output,
                        "error": exec_result.error,
                    }
                )
                break

        return {
            "plan": plan.model_dump(),
            "results": results,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return text[start : end + 1]
        return text


def main() -> None:
    book_flight_tool = ToolSchema(
        name="book_flight",
        parameters={
            "origin": ParameterDomain.from_values(["NYC", "BOS", "LAX"]),
            "destination": ParameterDomain.from_values(["SFO", "LAX", "JFK"]),
            "date": ParameterDomain.from_values(["2024-03-01", "2024-03-02"]),
            "class": ParameterDomain.from_values(["economy", "business"]),
        },
        required=frozenset({"origin", "destination", "date"}),
    )

    tool_registry = {
        "book_flight": lambda args: {"ok": True, "args": dict(args)},
    }

    agent = SGRPlanSageTTSAgent(
        tool_schemas={book_flight_tool.name: book_flight_tool},
        tool_executor=ToolRegistryExecutor(tool_registry),
        tts_config=TTSConfig(
            service_url=os.getenv("TTS_SERVICE_URL", "http://localhost:8001/v1"),
            model=os.getenv("TTS_MODEL", "openai/gpt-4o-mini"),
            tts_budget=int(os.getenv("TTS_BUDGET", "8")),
            temperature=float(os.getenv("TTS_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("TTS_MAX_TOKENS", "1024")),
            timeout=float(os.getenv("TTS_TIMEOUT", "120.0")),
        ),
    )

    user_input = "Book a flight from New York to Los Angeles next week."
    result = agent.run(user_input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
