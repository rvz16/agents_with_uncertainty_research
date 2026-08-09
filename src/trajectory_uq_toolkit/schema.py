from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = "trajectory-uq/v1"


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _validate_numeric_map(values: Any, path: str) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{path} must be an object")
    return {
        str(name): _finite_number(value, f"{path}.{name}")
        for name, value in values.items()
        if value is not None
    }


def _validate_critic_map(values: Any, path: str) -> dict[str, bool]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{path} must be an object")
    result = {}
    for name, value in values.items():
        if not isinstance(value, bool):
            raise ValueError(f"{path}.{name} must be bool")
        result[str(name)] = value
    return result


def validate_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one compact, environment-independent episode."""
    if not isinstance(record, Mapping):
        raise ValueError("episode must be an object")
    episode_id = str(record.get("episode_id", "")).strip()
    environment = str(record.get("environment", "")).strip()
    if not episode_id:
        raise ValueError("episode_id is required")
    if not environment:
        raise ValueError("environment is required")
    success = record.get("success")
    if success not in (0, 1, False, True):
        raise ValueError("success must be binary")

    raw_generations = record.get("generations")
    if not isinstance(raw_generations, list) or not raw_generations:
        raise ValueError("generations must be a non-empty list")
    generations = []
    for position, generation in enumerate(raw_generations):
        if not isinstance(generation, Mapping):
            raise ValueError(f"generations[{position}] must be an object")
        index = generation.get("index", position)
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"generations[{position}].index must be non-negative int")
        normalized = {
            "index": index,
            "signals": _validate_numeric_map(
                generation.get("signals"), f"generations[{position}].signals"
            ),
            "critics": _validate_critic_map(
                generation.get("critics"), f"generations[{position}].critics"
            ),
        }
        if generation.get("metadata") is not None:
            if not isinstance(generation["metadata"], Mapping):
                raise ValueError(f"generations[{position}].metadata must be an object")
            normalized["metadata"] = dict(generation["metadata"])
        generations.append(normalized)
    generations.sort(key=lambda row: row["index"])
    if len({row["index"] for row in generations}) != len(generations):
        raise ValueError("generation indices must be unique")

    normalized_episode = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "environment": environment,
        "success": int(bool(success)),
        "generations": generations,
        "critics": _validate_critic_map(record.get("critics"), "critics"),
        "features": _validate_numeric_map(record.get("features"), "features"),
    }
    if record.get("metadata") is not None:
        if not isinstance(record["metadata"], Mapping):
            raise ValueError("metadata must be an object")
        normalized_episode["metadata"] = dict(record["metadata"])
    return normalized_episode
