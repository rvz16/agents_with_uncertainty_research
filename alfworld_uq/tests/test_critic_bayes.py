import pytest

from belief.critic_bayes import CriticBayesState


def test_critic_bayes_state_updates_from_multiple_critics() -> None:
    observations = [
        {"format": True, "action": True},
        {"format": True, "action": True},
        {"format": False, "action": False},
        {"format": False, "action": False},
    ]
    model = CriticBayesState.fit(observations, [1, 1, 0, 0])

    assert model.predict({"format": True, "action": True}) > model.prior
    assert model.predict({"format": False, "action": False}) < model.prior


def test_critic_bayes_state_uses_beta_smoothing() -> None:
    model = CriticBayesState.fit(
        [{"critic": True}, {"critic": False}],
        [1, 0],
    )
    likelihood = model.likelihoods["critic"]
    assert likelihood.p_pass_success == pytest.approx(2 / 3)
    assert likelihood.p_pass_failure == pytest.approx(1 / 3)


def test_stepwise_state_uses_episode_prior_and_all_observations() -> None:
    model = CriticBayesState.fit(
        [{"critic": True}, {"critic": True}, {"critic": False}],
        [1, 1, 0],
        prior=0.25,
    )
    one_pass = model.predict_sequence([{"critic": True}])
    two_passes = model.predict_sequence(
        [{"critic": True}, {"critic": True}]
    )

    assert model.prior == pytest.approx(0.25)
    assert two_passes > one_pass > model.prior
