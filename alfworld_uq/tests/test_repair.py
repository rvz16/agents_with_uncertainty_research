from experiments.repair_api_errors import replace_episodes


def test_replace_episodes_preserves_order_and_replaces_trajectory() -> None:
    episodes = [
        {"episode_id": "a", "stop_reason": "success"},
        {"episode_id": "b", "stop_reason": "api_error"},
    ]
    trajectories = [
        {"episode_id": "a", "step": 1},
        {"episode_id": "b", "step": 1, "error": "timeout"},
    ]
    replacement_summary = {"episode_id": "b", "stop_reason": "max_steps"}
    replacement_rows = [
        {"episode_id": "b", "step": 2},
        {"episode_id": "b", "step": 1},
    ]
    new_episodes, new_rows = replace_episodes(
        episodes,
        trajectories,
        {1: (replacement_summary, replacement_rows)},
    )
    assert [row["episode_id"] for row in new_episodes] == ["a", "b"]
    assert [row["step"] for row in new_rows] == [1, 1, 2]
    assert "error" not in new_rows[1]
