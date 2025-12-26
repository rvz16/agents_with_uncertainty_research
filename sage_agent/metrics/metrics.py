from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from ..core.types import ToolCall


@dataclass(frozen=True)
class MetricResult:
    coverage_rate: float
    tool_match_rate: float
    parameter_match_rate: float
    avg_questions: float


def evaluate_metrics(
    predictions: Sequence[ToolCall],
    ground_truths: Sequence[ToolCall],
    question_counts: Sequence[int],
) -> MetricResult:
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    if len(predictions) != len(question_counts):
        raise ValueError("Question counts must align with predictions")
    if not predictions:
        raise ValueError("At least one prediction is required for metric computation")

    total_tool_match = 0
    total_param_match = 0.0
    total_coverage = 0
    total_questions = 0.0

    for idx, (pred, truth, questions) in enumerate(
        zip(predictions, ground_truths, question_counts)
    ):
        if questions < 0:
            raise ValueError(f"Question count must be non-negative at index {idx}")
        tool_match = pred.tool_name == truth.tool_name
        total_tool_match += 1 if tool_match else 0
        total_questions += float(questions)

        if tool_match:
            match_ratio = _parameter_match_ratio(pred.arguments, truth.arguments)
        else:
            match_ratio = 0.0
        total_param_match += match_ratio
        total_coverage += 1 if tool_match and match_ratio == 1.0 else 0

    count = float(len(predictions))
    return MetricResult(
        coverage_rate=total_coverage / count,
        tool_match_rate=total_tool_match / count,
        parameter_match_rate=total_param_match / count,
        avg_questions=total_questions / count,
    )


def _parameter_match_ratio(
    predicted: Mapping[str, object],
    ground_truth: Mapping[str, object],
) -> float:
    if not ground_truth:
        return 1.0
    matched = 0
    for key, value in ground_truth.items():
        if key in predicted and predicted[key] == value:
            matched += 1
    return matched / float(len(ground_truth))
