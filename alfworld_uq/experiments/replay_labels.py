"""Per-step labels by replaying a recorded episode in the environment.

Every prefix in this work carries its episode's final outcome as its label, so
a prefix two steps into a failed episode is labelled a failure although nothing
had gone wrong yet. The alternative is to label each prefix by what was
actually true at that step, which turns one episode into as many labelled
examples as it has steps and asks a question about the present rather than a
forecast.

Nothing new has to be generated for this: the recorded trajectories contain the
action the environment executed at every step, so replaying them in the same
game reproduces the state sequence exactly and the environment can be asked
what it thinks at each point. The replay is checked rather than trusted --
every replayed observation is compared with the recorded one, and an episode
whose observations diverge is marked rather than silently relabelled.

``--probe`` answers the question this was written around: our wrapper asks for
``goal_condition_success_rate`` and the recorded runs show ``None`` at every
step, so before building anything on a graded label we dump the keys the
environment actually returns.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def remap(gamefile: str, data_root: Path) -> Path:
    """The recorded path is the container's; keep it rooted at this data_root."""
    marker = "json_2.1.1"
    if marker in gamefile:
        return data_root / marker / gamefile.split(marker, 1)[1].lstrip("/")
    return Path(gamefile)


def episodes_of(run: Path) -> list[dict]:
    steps: dict[str, list[dict]] = {}
    for line in open(run / "trajectories.jsonl"):
        if line.strip():
            row = json.loads(line)
            steps.setdefault(row["episode_id"], []).append(row)
    out = []
    for line in open(run / "episodes.jsonl"):
        if not line.strip():
            continue
        episode = json.loads(line)
        rows = sorted(steps.get(episode["episode_id"], []), key=lambda r: int(r.get("step", 0)))
        if rows:
            out.append({"id": episode["episode_id"], "gamefile": episode.get("gamefile"),
                        "label": int(bool(episode["final_success"])), "rows": rows})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=None,
                   help="a run directory including the terminal step (a _finished one)")
    p.add_argument("--input", type=Path, default=None,
                   help="a compact file written by --dump; the run directories are symlinks into a "
                        "local cache and do not travel to a worker, this does")
    p.add_argument("--dump", action="store_true",
                   help="write the compact replay input and stop, without touching the environment")
    p.add_argument("--cohort", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path,
                   default=Path(os.environ.get("ALFWORLD_DATA", "/root/.cache/alfworld")))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--probe", type=int, default=0,
                   help="replay this many episodes and print every info key the environment "
                        "returns, instead of writing labels")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    if a.input:
        episodes = [json.loads(line) for line in open(a.input) if line.strip()]
    elif a.run:
        episodes = episodes_of(a.run)
    else:
        raise SystemExit("one of --run or --input is required")
    if a.limit:
        episodes = episodes[: a.limit]

    if a.dump:
        path = a.out / f"{a.cohort}.jsonl"
        with open(path, "w") as out:
            for episode in episodes:
                out.write(json.dumps({
                    "id": episode["id"], "gamefile": episode["gamefile"], "label": episode["label"],
                    "rows": [{"step": int(r.get("step", k)), "action": r.get("action"),
                              "observation": r.get("observation")}
                             for k, r in enumerate(episode["rows"])]}) + "\n")
        print(f"[replay] {a.cohort}: {len(episodes)} episodes -> {path} "
              f"({path.stat().st_size // 1024} KB)")
        return

    from environments.alfworld_env import ALFWorldTextEnv

    source = a.run.name if a.run else a.input.name
    print(f"[replay] {a.cohort}: {len(episodes)} episodes from {source}", flush=True)

    handle = None if a.probe else open(a.out / f"labels_{a.cohort}.jsonl", "w")
    agree = diverged = missing = 0
    for index, episode in enumerate(episodes[: a.probe or None], 1):
        gamefile = remap(episode["gamefile"] or "", a.data_root)
        if not gamefile.exists():
            missing += 1
            continue
        env = ALFWorldTextEnv(data_root=a.data_root, gamefile=gamefile, graded=True,
                              max_steps=len(episode["rows"]) + 5)
        try:
            env.reset()
            ok = True
            rows = []
            for k, row in enumerate(episode["rows"]):
                action = (row.get("action") or "").strip()
                if not action:  # a generation that executed nothing: the state does not move
                    rows.append({"step": k, "action": None, "executed": False})
                    continue
                if a.probe:
                    observations, _, dones, infos = env.env.step([action])
                    print(f"  step {k}: info keys = {sorted(infos.keys())}", flush=True)
                    facts = infos.get("facts")
                    wins = infos.get("win_facts")
                    first_facts = facts[0] if isinstance(facts, list) and facts else None
                    first_wins = wins[0] if isinstance(wins, list) and wins else None
                    print(f"      facts: {len(first_facts) if first_facts else 0} propositions, "
                          f"sample {[str(f) for f in (first_facts or [])[:3]]}", flush=True)
                    print(f"      win_facts: {type(first_wins).__name__}, "
                          f"{str(first_wins)[:200]}", flush=True)
                    print(f"      satisfied = {ALFWorldTextEnv._satisfied(infos)}", flush=True)
                    plan = infos.get("extra.expert_plan")
                    print(f"      expert_plan = {str(plan)[:120]}", flush=True)
                    if k >= 2:
                        break
                    continue
                result = env.step(action)
                match = norm(result.observation) == norm(row.get("observation") or "")
                ok = ok and match
                rows.append({"step": k, "action": action, "executed": True,
                             "progress": result.progress,
                             "goal_conditions": list(result.goal_conditions or ()),
                             "plan_left": result.expert_plan_length,
                             "won": bool(result.won),
                             "done": bool(result.done), "obs_match": match})
            if a.probe:
                print(f"[probe] episode {index} done", flush=True)
                env.close()
                continue
            final_won = any(r.get("won") for r in rows)
            agree += int(final_won == bool(episode["label"]))
            diverged += int(not ok)
            for r in rows:
                handle.write(json.dumps({"id": episode["id"], "cohort": a.cohort,
                                         "episode_label": episode["label"],
                                         "replay_won": final_won, "diverged": not ok, **r}) + "\n")
        finally:
            env.close()
        if index % 20 == 0:
            print(f"[replay] {index}/{len(episodes)}  agree={agree} diverged={diverged}", flush=True)
    if handle:
        handle.close()
        print(f"[replay] {a.cohort}: final outcome reproduced for {agree}/{len(episodes) - missing}, "
              f"{diverged} episodes with a diverging observation, {missing} gamefiles missing",
              flush=True)


if __name__ == "__main__":
    main()
