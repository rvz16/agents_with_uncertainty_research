import math

import pytest

from uq.aggregation import aggregate_trajectory, cvar, ewma
from uq.perplexity import perplexity
from uq.seqprob import (
    mean_token_log_probability,
    sequence_probability,
    sum_log_probability,
)


def test_token_logprob_metrics_are_distinct() -> None:
    values = [-1.0, -2.0]
    assert sum_log_probability(values) == -3.0
    assert mean_token_log_probability(values) == -1.5
    assert sequence_probability(values) == pytest.approx(math.exp(-3.0))
    assert perplexity(values) == pytest.approx(math.exp(1.5))


def test_all_trajectory_aggregations() -> None:
    result = aggregate_trajectory(
        [1.0, 2.0, None, 5.0, 4.0],
        threshold=3.0,
        higher_is_uncertain=True,
        ewma_alpha=0.5,
        last_k=2,
        cvar_fraction=0.5,
    )
    assert result["last"] == 4.0
    assert result["mean"] == 3.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert result["median"] == 3.0
    assert result["ewma"] == pytest.approx(3.625)
    assert result["fraction_uncertain"] == 0.5
    assert result["last_k_mean"] == 4.5
    assert result["cvar"] == 4.5


def test_low_tail_cvar_and_validation() -> None:
    assert cvar([1.0, 2.0, 8.0, 9.0], fraction=0.5, higher_is_uncertain=False) == 1.5
    assert ewma([], alpha=0.3) is None
    with pytest.raises(ValueError):
        ewma([1.0], alpha=0.0)
