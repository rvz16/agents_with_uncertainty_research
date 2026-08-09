from __future__ import annotations

from code_uq.analysis.entropy_kl_trajectory import token_entropy_kl, token_self_certainty
from code_uq.analysis.experiment2_uq_bayes_critic import (
    kfold_continuous_fuse,
    kfold_fuse,
)
from code_uq.environments.fitted_live.function_adapters import make_function_adapter


def _rows() -> list[dict]:
    rows = []
    for index in range(20):
        quality = index % 2
        rows.append(
            {
                "iid": f"episode-{index}",
                "bayes": 0.65 if quality else 0.35,
                "quality": quality,
                "feat_raw": -1.0 if quality else -8.0,
            }
        )
    return rows


def test_binary_and_continuous_fusion_are_cross_fitted() -> None:
    rows = _rows()
    binary = kfold_fuse(rows, True, k=5, seed=3, mode="double")
    continuous = kfold_continuous_fuse(rows, k=5, seed=3, lambda_=0.25)

    assert set(binary) == {row["iid"] for row in rows}
    assert set(continuous) == set(binary)
    assert all(0.0 < score < 1.0 for score in binary.values())
    assert all(0.0 < score < 1.0 for score in continuous.values())


def test_distribution_signals_accept_top_logprobs() -> None:
    top_logprobs = [{"logprob": -0.1}, {"logprob": -1.2}, {"logprob": -2.5}]
    entropy, kl_uniform = token_entropy_kl(top_logprobs)

    assert entropy > 0.0
    assert kl_uniform >= 0.0
    assert token_self_certainty(top_logprobs) > 0.0


def test_function_adapter_can_be_constructed_without_loading_a_dataset() -> None:
    adapter = make_function_adapter(
        benchmark="lcb_hard",
        n_instances=2,
        seed=42,
        lcb_version="all",
        plus_input_cap=20,
        lcb_private_test_cap=5,
        platform="leetcode",
    )

    assert adapter.benchmark == "lcb_hard"
    assert adapter.n_instances == 2
