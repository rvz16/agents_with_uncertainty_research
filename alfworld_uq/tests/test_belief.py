import math

import pytest

from belief.binary_bayes import BinaryBayesUQ, DoubleBinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def test_binary_bayes_uses_calibration_likelihoods() -> None:
    model = BinaryBayesUQ.fit(
        [[1.0, 1.2], [4.0, 5.0]],
        [1, 0],
        threshold_quantile=0.5,
        higher_is_uncertain=True,
    )
    assert model.predict([1.1]) > model.prior
    assert model.predict([4.5]) < model.prior
    assert len(model.predict_sequence([1.1, 4.5])) == 2


def test_continuous_bayes_and_tempering() -> None:
    sequences = [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]]
    labels = [1, 0]
    full = ContinuousBayesUQ.fit(sequences, labels, lambda_=1.0)
    tempered = ContinuousBayesUQ.fit(sequences, labels, lambda_=0.25)
    full_prediction = full.predict([1.5])
    tempered_prediction = tempered.predict([1.5])
    assert full_prediction > full.prior
    assert tempered_prediction > tempered.prior
    assert abs(_logit(tempered_prediction)) < abs(_logit(full_prediction))


def test_continuous_bayes_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError):
        ContinuousBayesUQ.fit([[1.0], [2.0]], [1, 0], lambda_=-0.1)


@pytest.mark.parametrize("mode", ["sep", "lr_pos", "lr_neg"])
def test_binary_bayes_likelihood_ratio_threshold_modes(mode: str) -> None:
    model = BinaryBayesUQ.fit(
        [[0.1, 0.2], [0.2, 0.3], [0.8, 0.9], [0.9, 1.0]],
        [1, 1, 0, 0],
        threshold_mode=mode,
        higher_is_uncertain=True,
    )
    assert model.predict([0.15]) > model.prior
    assert model.predict([0.95]) < model.prior


def test_double_binary_bayes_applies_both_thresholds() -> None:
    model = DoubleBinaryBayesUQ.fit(
        [[0.1, 0.2], [0.2, 0.3], [0.8, 0.9], [0.9, 1.0]],
        [1, 1, 0, 0],
        higher_is_uncertain=True,
    )
    manual = model.negative.update(
        model.positive.update(model.prior, 0.15), 0.15
    )
    assert model.predict([0.15]) == pytest.approx(manual)
