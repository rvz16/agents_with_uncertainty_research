"""The prefix report exists to make one channel impossible, so test that."""
import json

from experiments.prefix_report import load_cohort, report


def _write_run(tmp_path, episodes):
    """episodes: {episode_id: (num_steps, success)}"""
    trajectories, summaries = [], []
    for episode_id, (steps, success) in episodes.items():
        summaries.append(
            {"episode_id": episode_id, "num_steps": steps, "final_success": success}
        )
        for step in range(1, steps + 1):
            trajectories.append(
                {
                    "episode_id": episode_id,
                    "step": step,
                    "format_valid": True,
                    "action_valid": True,
                    "fallback_reason": None,
                    "uq": {"combined": {"mean_token_logprob": -0.1 * step}},
                }
            )
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in summaries)
    )
    (tmp_path / "trajectories.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trajectories)
    )
    return tmp_path


def test_short_episodes_leave_the_cohort_so_every_prefix_is_the_same_length(tmp_path):
    run = _write_run(tmp_path, {"win": (4, True), "long_win": (12, True), "loss": (30, False)})
    cohort = load_cohort(run, prefix_steps=10)

    assert "win" not in cohort["ids"]  # 4 steps cannot yield a 10-step prefix
    assert sorted(cohort["ids"]) == ["long_win", "loss"]
    assert {len(rows) for rows in cohort["prefixes"].values()} == {10}


def test_the_stopping_rule_is_reported_rather_than_used(tmp_path):
    """Length stays visible as a diagnostic, and out of every scored signal.

    On a run where success is exactly 'stopped early', the artefact scores a
    perfect AUROC while the prefix signals cannot see it: the prefixes are
    identical in length and, here, identical in content.
    """
    episodes = {f"win{i}": (5 + i, True) for i in range(4)}
    episodes.update({f"loss{i}": (30, False) for i in range(4)})
    run = _write_run(tmp_path, episodes)

    result = report(run, prefix_steps=5, seeds=2, fraction=0.5)
    assert result["length_artefact_auroc"] == 1.0
    # every episode's first five steps are the same, so nothing can separate them
    assert result["best_prefix_signal"]["auroc"] == 0.5
