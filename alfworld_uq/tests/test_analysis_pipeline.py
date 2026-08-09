import json
import subprocess
import sys
from pathlib import Path


def test_full_posthoc_analysis_from_jsonl(tmp_path: Path) -> None:
    trajectories_path = tmp_path / "trajectories.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    judge_scores_path = tmp_path / "judge_scores.jsonl"
    output_dir = tmp_path / "analysis"
    trajectories = []
    episodes = []
    for episode_index in range(8):
        success = episode_index % 2 == 0
        episode_id = f"episode-{episode_index}"
        episodes.append(
            {
                "episode_id": episode_id,
                "task_type": "pick_and_place_simple",
                "final_success": success,
            }
        )
        for step in range(1, 5):
            base = 1.2 if success else 4.0
            perplexity = base + 0.05 * step + 0.01 * episode_index
            mean_logprob = -perplexity
            metrics = {
                "num_tokens": 2,
                "perplexity": perplexity,
                "sum_logprob": 2.0 * mean_logprob,
                "mean_token_logprob": mean_logprob,
                "sequence_probability": float(
                    __import__("math").exp(2.0 * mean_logprob)
                ),
                "verbalized_confidence": 0.8 if success else 0.2,
            }
            trajectories.append(
                {
                    "episode_id": episode_id,
                    "step": step,
                    "final_success": success,
                    "logprobs_available": True,
                    "uq": {
                        "thought": metrics,
                        "action": metrics,
                        "combined": metrics,
                    },
                }
            )
    trajectories_path.write_text(
        "\n".join(json.dumps(row) for row in trajectories) + "\n",
        encoding="utf-8",
    )
    episodes_path.write_text(
        "\n".join(json.dumps(row) for row in episodes) + "\n",
        encoding="utf-8",
    )
    judge_scores_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_id": row["episode_id"],
                    "judge_pass": row["final_success"],
                    "confidence": 0.9,
                    "status": "ok",
                }
            )
            for row in episodes
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.analyze_trajectories",
            "--trajectories",
            str(trajectories_path),
            "--output-dir",
            str(output_dir),
            "--judge-scores",
            str(judge_scores_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "metrics.csv").stat().st_size > 200
    assert (output_dir / "prefix_metrics.csv").stat().st_size > 200
    assert "prr_at_0_5" in (output_dir / "metrics.csv").read_text().splitlines()[0]
    assert "prr_at_0_5" in (
        output_dir / "prefix_metrics.csv"
    ).read_text().splitlines()[0]
    assert (output_dir / "risk_coverage.csv").stat().st_size > 200
    assert (output_dir / "model_parameters.csv").stat().st_size > 200
    assert (output_dir / "critic_likelihoods.csv").stat().st_size > 100
    assert (output_dir / "bayes_states.csv").stat().st_size > 200
    assert (output_dir / "belief_trajectories.jsonl").stat().st_size > 0
    assert "| Model | AUROC | AUPRC | PRR@0.5 |" in (
        output_dir / "report.md"
    ).read_text()
    metrics_text = (output_dir / "metrics.csv").read_text()
    assert "bayes_state" in metrics_text
    assert "bayes_state_plus_tempered" in metrics_text
    assert "stepwise_bayes_state" in metrics_text
    assert "stepwise_bayes_state_plus_tempered" in metrics_text
    assert "llm_judge_state" in metrics_text
    assert "bayes_state_plus_judge" in metrics_text
    assert "bayes_state_plus_judge_plus_tempered" in metrics_text
    for name in (
        "prefix_metrics.png",
        "belief_by_outcome.png",
        "trajectory_examples.png",
        "risk_coverage.png",
    ):
        assert (output_dir / name).stat().st_size > 1_000
