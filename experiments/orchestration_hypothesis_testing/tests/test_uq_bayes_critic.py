from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import experiment2_uq_bayes_critic as experiment  # noqa: E402


def test_continuous_and_tempered_fusion_use_existing_belief() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "iid": f"success-{index}",
                "bayes": 0.5,
                "quality": 1,
                "feat_raw": 0.1 + index * 0.01,
            }
        )
        rows.append(
            {
                "iid": f"failure-{index}",
                "bayes": 0.5,
                "quality": 0,
                "feat_raw": 0.8 + index * 0.01,
            }
        )

    full = experiment.kfold_continuous_fuse(rows, k=5, seed=0, lambda_=1.0)
    tempered = experiment.kfold_continuous_fuse(rows, k=5, seed=0, lambda_=0.25)

    assert all(full[f"success-{index}"] > 0.5 for index in range(10))
    assert all(full[f"failure-{index}"] < 0.5 for index in range(10))
    assert all(
        abs(experiment._logit(tempered[row["iid"]]))
        <= abs(experiment._logit(full[row["iid"]]))
        for row in rows
    )


def test_load_trajectory_feature_orders_and_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    rows = [
        {"instance_id": "a", "patch_idx": 2, "score": 3.0},
        {"instance_id": "a", "patch_idx": 0, "score": 1.0},
        {"instance_id": "a", "patch_idx": 1, "score": 2.0},
        {
            "instance_id": "b",
            "patch_idx": 0,
            "score": 99.0,
            "logprobs_supported": False,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert experiment.load_trajectory_feature(path, "score", "last") == {"a": 3.0}
    assert experiment.load_trajectory_feature(path, "score", "mean") == {"a": 2.0}


def test_probability_metrics_are_exact_for_perfect_predictions() -> None:
    metrics = experiment.probability_metrics([0, 1], [0.0, 1.0])
    assert metrics["auroc"] == 1.0
    assert metrics["brier"] == pytest.approx(1e-12)
