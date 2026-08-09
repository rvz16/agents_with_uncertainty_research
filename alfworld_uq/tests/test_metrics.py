import pytest

from experiments.analyze_trajectories import auprc, auroc, metric_values, prr


def test_classification_metrics() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, probabilities) == 1.0
    assert auprc(labels, probabilities) == 1.0
    values = metric_values(labels, probabilities)
    assert values["brier"] == pytest.approx(0.025)
    assert values["ece"] >= 0.0
    assert values["prr_at_0_5"] == pytest.approx(1.0)


def test_prr_distinguishes_oracle_and_reversed_rankings() -> None:
    labels = [0, 0, 1, 1]
    assert prr([0.1, 0.2, 0.8, 0.9], labels) == pytest.approx(1.0)
    assert prr([0.9, 0.8, 0.2, 0.1], labels) < 0.0


def test_ranking_metrics_are_undefined_for_single_class() -> None:
    assert auroc([0, 0], [0.1, 0.2]) is None
    assert auprc([0, 0], [0.1, 0.2]) is None
