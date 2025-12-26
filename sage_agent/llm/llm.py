from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Protocol

from .prompts import build_candidate_prompt, build_question_prompt
from ..core.types import Aspect, Question, ToolCallCandidate, ToolSchema


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class PromptCandidateGenerator:
    llm: LLMClient

    def generate(self, user_input: str, observations: Iterable[str], tool_schemas: Iterable[ToolSchema]) -> List[ToolCallCandidate]:
        prompt = build_candidate_prompt(user_input, tool_schemas, observations)
        response = self.llm.complete(prompt)
        payload = _parse_json_list(response)
        candidates: List[ToolCallCandidate] = []
        for item in payload:
            tool_name = item.get("tool")
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict) or not isinstance(tool_name, str):
                continue
            candidates.append(ToolCallCandidate(tool_name=tool_name, arguments=arguments))
        return candidates


@dataclass
class PromptQuestionGenerator:
    llm: LLMClient

    def generate(
        self,
        user_input: str,
        observations: Iterable[str],
        tool_schemas: Iterable[ToolSchema],
        candidates: Iterable[ToolCallCandidate],
    ) -> List[Question]:
        candidates_payload = [
            {"tool": c.tool_name, "arguments": dict(c.arguments)} for c in candidates
        ]
        prompt = build_question_prompt(user_input, tool_schemas, observations, candidates_payload)
        response = self.llm.complete(prompt)
        payload = _parse_json_list(response)
        questions: List[Question] = []
        for item in payload:
            text = item.get("question")
            aspects_payload = item.get("aspects", [])
            if not isinstance(text, str) or not isinstance(aspects_payload, list):
                continue
            aspects: List[Aspect] = []
            for aspect in aspects_payload:
                tool_name = aspect.get("tool")
                param = aspect.get("param")
                if isinstance(tool_name, str) and isinstance(param, str):
                    aspects.append(Aspect(tool_name=tool_name, param_name=param))
            if aspects:
                questions.append(Question(text=text, aspects=tuple(aspects)))
        return questions


def _parse_json_list(text: str) -> List[Mapping[str, object]]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, list):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, list):
                return payload
        except json.JSONDecodeError:
            return []
    return []
