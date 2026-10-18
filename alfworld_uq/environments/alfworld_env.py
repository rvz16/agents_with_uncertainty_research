from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


SPLITS = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


@dataclass
class EpisodeState:
    episode_id: str
    task_type: str
    task: str
    observation: str
    admissible_actions: list[str]
    gamefile: str


@dataclass
class StepResult:
    observation: str
    admissible_actions: list[str]
    done: bool
    won: bool
    progress: float | None
    goal_conditions: tuple[int, int] | None = None
    expert_plan_length: int | None = None


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _extract_task(observation: str) -> str:
    match = re.search(r"(?is)Your task is to:\s*(.+?)(?:\n\n|$)", observation)
    return match.group(1).strip() if match else observation.strip()


class ALFWorldTextEnv:
    def __init__(
        self,
        *,
        data_root: Path,
        split: str = "valid_seen",
        max_steps: int = 30,
        num_episodes: int = 10,
        episode_offset: int = 0,
        seed: int = 0,
        task_types: list[int] | None = None,
        gamefile: Path | None = None,
        graded: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLITS)}")
        dataset_root = data_root / "json_2.1.1"
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"ALFWorld data not found at {dataset_root}. Run `alfworld-download`."
            )

        config = {
            "dataset": {
                "data_path": str(dataset_root / "train"),
                "eval_id_data_path": str(dataset_root / "valid_seen"),
                "eval_ood_data_path": str(dataset_root / "valid_unseen"),
                "num_train_games": 0,
                "num_eval_games": 0,
            },
            "env": {
                "goal_desc_human_anns_prob": 0.0,
                "task_types": task_types or [1, 2, 3, 4, 5, 6],
                "domain_randomization": False,
                "expert_type": "handcoded",
            },
            "general": {"training_method": "dqn"},
            "rl": {"training": {"max_nb_steps_per_episode": max_steps}},
            "logic": {
                "domain": str(data_root / "logic" / "alfred.pddl"),
                "grammar": str(data_root / "logic" / "alfred.twl2"),
            },
        }
        manager = AlfredTWEnv(config, train_eval=SPLITS[split])
        games = sorted(manager.game_files)
        random.Random(seed).shuffle(games)
        if gamefile is not None:
            resolved_gamefile = str(gamefile.expanduser().resolve())
            if not Path(resolved_gamefile).exists():
                raise FileNotFoundError(f"ALFWorld gamefile not found: {resolved_gamefile}")
            games = [resolved_gamefile]
            episode_offset = 0
            num_episodes = 1
        if episode_offset < 0:
            raise ValueError("episode_offset must be non-negative")
        if episode_offset + num_episodes > len(games):
            raise ValueError(
                f"Requested episodes [{episode_offset}, "
                f"{episode_offset + num_episodes}), but split {split} has "
                f"only {len(games)} supported games."
            )
        manager.game_files = games[episode_offset : episode_offset + num_episodes]
        manager.num_games = len(manager.game_files)
        self.env = self._graded_env(manager, max_steps) if graded else manager.init_env(batch_size=1)
        self.graded = graded
        self.data_root = data_root
        self._episode_number = 0

    @staticmethod
    def _graded_env(manager: AlfredTWEnv, max_steps: int) -> Any:
        """The same games, asked for the goal formula as well as the outcome.

        ``AlfredTWEnv.init_env`` registers ``EnvInfos(won, admissible_commands,
        extras=["gamefile"])`` and nothing else, so the text environment never
        reports ``goal_condition_success_rate`` -- that is computed only by the
        embodied THOR environment, which we do not run. The quest is still
        available as logic: ``win_facts`` is the winning conjunction and
        ``facts`` is everything true in the current state, so the fraction of
        goal conditions satisfied can be computed at any step.
        """
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler, AlfredExpert, AlfredInfos,
        )

        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                                   win_facts=True, extras=["gamefile", "expert_plan"])
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(expert_type="handcoded")]
        env_id = textworld.gym.register_games(
            manager.game_files, infos, batch_size=1, asynchronous=True,
            max_episode_steps=max_steps, wrappers=wrappers)
        return textworld.gym.make(env_id)

    @staticmethod
    def _satisfied(infos: dict) -> tuple[int, int] | None:
        """How many of the quest's winning conditions hold right now.

        ``win_facts`` arrives as one list of propositions per winning quest;
        a state satisfies a quest when every proposition in it is in ``facts``.
        We report the best-satisfied quest, since any one of them ends the
        episode. Shapes differ between TextWorld backends, so anything we do
        not recognise returns None rather than a made-up number.
        """
        facts = _first(infos.get("facts"))
        wins = _first(infos.get("win_facts"))
        if not facts or not wins:
            return None
        present = {str(fact) for fact in facts}
        quests = wins if isinstance(wins[0], (list, tuple)) else [wins]
        best = (0, 0)
        for quest in quests:
            need = [str(fact) for fact in quest]
            if not need:
                continue
            hit = sum(1 for fact in need if fact in present)
            if not best[1] or hit / len(need) > best[0] / best[1]:
                best = (hit, len(need))
        return best if best[1] else None

    @staticmethod
    def _metadata(gamefile: str, observation: str) -> tuple[str, str, str]:
        game_path = Path(gamefile)
        traj_path = game_path.parent / "traj_data.json"
        task_type = "unknown"
        task = _extract_task(observation)
        if traj_path.exists():
            payload = json.loads(traj_path.read_text(encoding="utf-8"))
            task_type = payload.get("task_type", task_type)
            annotations = payload.get("turk_annotations", {}).get("anns", [])
            if annotations and annotations[0].get("task_desc"):
                task = annotations[0]["task_desc"].strip()
        digest = hashlib.sha1(str(game_path).encode("utf-8")).hexdigest()[:12]
        return f"{task_type}-{digest}", task_type, task

    def reset(self) -> EpisodeState:
        observations, infos = self.env.reset()
        observation = str(_first(observations, ""))
        gamefile = str(_first(infos.get("extra.gamefile"), ""))
        episode_id, task_type, task = self._metadata(gamefile, observation)
        self._episode_number += 1
        return EpisodeState(
            episode_id=episode_id,
            task_type=task_type,
            task=task,
            observation=observation,
            admissible_actions=list(_first(infos.get("admissible_commands"), [])),
            gamefile=gamefile,
        )

    def step(self, action: str) -> StepResult:
        observations, _, dones, infos = self.env.step([action])
        conditions = self._satisfied(infos)
        plan = _first(infos.get("extra.expert_plan"))
        return StepResult(
            observation=str(_first(observations, "")),
            admissible_actions=list(_first(infos.get("admissible_commands"), [])),
            done=bool(_first(dones, False)),
            won=bool(_first(infos.get("won"), False)),
            progress=(conditions[0] / conditions[1]) if conditions else None,
            goal_conditions=conditions,
            expert_plan_length=len(plan) if isinstance(plan, (list, tuple)) else None,
        )

    def close(self) -> None:
        self.env.close()
