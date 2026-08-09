from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from trajectory_uq_toolkit.analyze import _parse_directions, analyze
from trajectory_uq_toolkit.collect import collect
from trajectory_uq_toolkit.metrics import auprc, prr
from trajectory_uq_toolkit.schema import validate_episode
from trajectory_uq_toolkit.signals import summarize_token_logprobs


ROOT = Path(__file__).resolve().parents[1]


def test_token_summaries_do_not_require_raw_storage() -> None:
    result = summarize_token_logprobs(
        [-0.1, -0.2],
        [[-0.1, -1.1, -2.1], [-0.2, -0.8, -1.8]],
    )
    assert result["sum_logprob"] == pytest.approx(-0.3)
    assert result["num_tokens"] == 2
    assert result["mean_token_entropy"] > 0
    assert math.isfinite(result["self_certainty"])


def test_schema_rejects_non_binary_critic() -> None:
    with pytest.raises(ValueError, match="must be bool"):
        validate_episode(
            {
                "episode_id": "bad",
                "environment": "toy",
                "success": 1,
                "generations": [{"signals": {"perplexity": 1.2}, "critics": {"valid": 1}}],
            }
        )


def test_constant_scores_are_not_given_optimistic_tie_breaks() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.5] * len(labels)
    assert auprc(labels, scores) == pytest.approx(0.5)
    assert prr(scores, labels) == 0.0


def test_toy_collection_and_full_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    succeeded, failed = collect(
        adapter_reference=str(ROOT / "examples" / "toy_adapter.py"),
        config={"num_episodes": 40, "min_generations": 2, "max_generations": 4},
        output_dir=run_dir,
        limit=None,
        seed=11,
        workers=3,
    )
    assert (succeeded, failed) == (40, 0)
    records = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 40
    assert "top_logprobs" not in json.dumps(records[0])

    analysis_dir = tmp_path / "analysis"
    rows = analyze(
        episodes_path=run_dir / "episodes.jsonl",
        output_dir=analysis_dir,
        calibration_fraction=0.5,
        seed=3,
        tempered_lambda=0.25,
        signal_directions=_parse_directions([]),
        critic_set_specs=[],
    )
    names = {row["model"] for row in rows}
    assert "prior" in names
    assert "uq:binary:double" in names
    assert "critic:all" in names
    assert "critic:all+uq:continuous_tempered" in names
    assert "stepwise_critics:all" in names
    written = list(csv.DictReader((analysis_dir / "metrics.csv").open()))
    assert len(written) == len(rows)


def test_collection_resumes_without_duplicate_rows(tmp_path: Path) -> None:
    kwargs = {
        "adapter_reference": str(ROOT / "examples" / "toy_adapter.py"),
        "config": {"num_episodes": 8},
        "output_dir": tmp_path,
        "limit": None,
        "seed": 0,
        "workers": 1,
    }
    assert collect(**kwargs) == (8, 0)
    assert collect(**kwargs) == (0, 0)
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 8
