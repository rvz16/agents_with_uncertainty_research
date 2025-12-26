from __future__ import annotations

import re
from typing import Iterable, Optional

from .domains import ParameterDomain
from .types import ConstraintExtractor


class SimpleConstraintExtractor(ConstraintExtractor):
    def __init__(self, negative_patterns: Optional[Iterable[str]] = None) -> None:
        self._negative_patterns = tuple(negative_patterns or ("not {value}", "no {value}"))

    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        if domain.is_finite():
            return self._update_finite(domain, response)
        return domain

    def _update_finite(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        response_lower = response.lower()
        values = list(domain.values or [])
        matched = {v for v in values if str(v).lower() in response_lower}
        excluded = set()
        for v in values:
            v_lower = str(v).lower()
            for pattern in self._negative_patterns:
                if pattern.format(value=v_lower) in response_lower:
                    excluded.add(v)
                    break
        if matched:
            new_values = matched.difference(excluded)
            if not new_values:
                # If all matched values were excluded, fall back to excluding only.
                return domain.exclude_values(excluded)
            return domain.intersect_values(new_values)
        if excluded:
            return domain.exclude_values(excluded)
        return domain
