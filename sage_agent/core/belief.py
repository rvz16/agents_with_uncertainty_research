from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

from .domains import ParameterDomain
from .types import ToolCallCandidate, ToolSchema, UNK


@dataclass
class BeliefState:
    domains: Dict[str, Dict[str, ParameterDomain]]
    epsilon: float = 1e-4

    def candidate_weight(
        self, candidate: ToolCallCandidate, tool_schema: ToolSchema
    ) -> float:
        tool_domains = self.domains[tool_schema.name]
        weight = 1.0
        for param_name, domain in tool_schema.parameters.items():
            value = candidate.arguments.get(param_name, UNK)
            if value != UNK:
                if not domain.contains(value):
                    return 0.0
                weight *= 1.0
                continue
            size = domain.size()
            if size is None:
                weight *= self.epsilon
            elif size == 0:
                return 0.0
            else:
                weight *= 1.0 / float(size)
        return weight

    def normalize(self, weights: Iterable[float]) -> list[float]:
        weights_list = list(weights)
        total = sum(weights_list)
        if total <= 0:
            if not weights_list:
                return []
            uniform = 1.0 / float(len(weights_list))
            return [uniform for _ in weights_list]
        return [w / total for w in weights_list]
