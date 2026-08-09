from pathlib import Path

import pytest

from environments.alfworld_env import ALFWorldTextEnv


DATA_ROOT = Path("~/.cache/alfworld").expanduser()


@pytest.mark.integration
@pytest.mark.skipif(
    not (DATA_ROOT / "json_2.1.1").exists(), reason="ALFWorld data is not downloaded"
)
def test_text_only_environment_reset_and_step() -> None:
    env = ALFWorldTextEnv(
        data_root=DATA_ROOT,
        split="valid_seen",
        max_steps=2,
        num_episodes=1,
        seed=0,
    )
    try:
        state = env.reset()
        assert state.task
        assert state.task_type != "unknown"
        assert state.admissible_actions
        result = env.step(state.admissible_actions[0])
        assert isinstance(result.observation, str)
        assert isinstance(result.done, bool)
    finally:
        env.close()

    exact_env = ALFWorldTextEnv(
        data_root=DATA_ROOT,
        split="valid_seen",
        max_steps=1,
        num_episodes=1,
        gamefile=Path(state.gamefile),
    )
    try:
        exact_state = exact_env.reset()
        assert exact_state.episode_id == state.episode_id
        assert exact_state.gamefile == state.gamefile
    finally:
        exact_env.close()
