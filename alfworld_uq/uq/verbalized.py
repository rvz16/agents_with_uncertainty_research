from __future__ import annotations

import re


def parse_verbalized_confidence(text: str) -> float | None:
    """Parse an optional confidence emitted by a compatible custom prompt."""
    match = re.search(
        r"(?i)\b(?:confidence|probability of success)\s*:\s*([01](?:\.\d+)?)",
        text,
    )
    if not match:
        return None
    return min(1.0, max(0.0, float(match.group(1))))
