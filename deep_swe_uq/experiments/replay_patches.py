"""Reconstruct what the repository looked like partway through a trajectory.

Grading an intermediate point needs a patch for that point, and the logs hold
commands rather than repository states. So the commands are replayed inside the
task's own image, in order, and the working tree is diffed against the base
commit after each one that could have changed it. Those diffs are what
``grade_patch.py grade`` then scores with the task's own verifier.

Three things make this cheap enough to run. Only 17% of recorded commands touch
the repository at all, so the rest are replayed but never snapshotted. The
agent's own test and build invocations are skipped by default: they take most of
the wall time, cannot change tracked files, and the grading uses the official
suites anyway. And identical diffs are written once, since a command that
changes nothing produces nothing to grade.

One container per episode, not one per snapshot. The container is kept alive and
commands go through ``docker exec``, which is what makes a hundred episodes a
few hours rather than a few days.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

MUTATE = re.compile(r"(^|[|;&]\s*)(sed\s+-i|python\s*-\s*<<|cat\s*>|tee\s|patch\s|git\s+apply|"
                    r"apply_patch|mv\s|cp\s|rm\s|touch\s|mkdir\s|echo\s+.*>|>>?\s*\S)")
TESTY = re.compile(r"(^|[|;&]\s*)(pytest|python\s+-m\s+pytest|go\s+test|go\s+build|tox|nosetests|"
                   r"npm\s+(test|run|install)|cargo\s+(test|build)|make(\s|$))")


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def episodes_of(compact: Path) -> list[dict]:
    out = []
    for line in open(compact):
        if not line.strip():
            continue
        record = json.loads(line)
        if "partial" not in (record.get("rewards") or {}):
            continue
        out.append({"id": record["id"],
                    "commands": [(s.get("command") or "") for s in (record.get("steps") or [])],
                    "rewards": record["rewards"]})
    return out


def image_of(task_dir: Path) -> str | None:
    text = (task_dir / "task.toml").read_text()
    match = re.search(r'docker_image\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else None


def base_of(task_dir: Path) -> str | None:
    match = re.search(r'base_commit_hash\s*=\s*"([^"]+)"', (task_dir / "task.toml").read_text())
    return match.group(1) if match else None


def replay(episode: dict, task_dir: Path, out_dir: Path, a) -> int:
    image, base = image_of(task_dir), base_of(task_dir)
    if not image or not base:
        print(f"[replay] {episode['id']}: no image or base commit in task.toml", flush=True)
        return 0
    name = "replay_" + hashlib.sha1(episode["id"].encode()).hexdigest()[:12]
    run(["docker", "rm", "-f", name], 60)
    rc, out = run(["docker", "run", "-d", "--name", name, "--network=none",
                   "--entrypoint", "/bin/bash", image, "-c", "sleep infinity"], a.start_timeout)
    if rc:
        print(f"[replay] {episode['id']}: container failed: {out.strip()[-200:]}", flush=True)
        return 0

    target = out_dir / episode["id"]
    target.mkdir(parents=True, exist_ok=True)
    written, seen = 0, set()
    try:
        run(["docker", "exec", name, "git", "config", "--global", "--add",
             "safe.directory", "/app"], 60)
        mutating = [k for k, c in enumerate(episode["commands"]) if MUTATE.search(c)]
        wanted = set(mutating[-a.snapshots:] if a.snapshots and len(mutating) > a.snapshots
                     else mutating)
        for k, command in enumerate(episode["commands"]):
            if not command.strip():
                continue
            if a.skip_tests and TESTY.search(command):
                continue
            run(["docker", "exec", "-w", "/app", name, "bash", "-lc", command], a.command_timeout)
            if k not in wanted:
                continue
            rc, diff = run(["docker", "exec", "-w", "/app", name, "git", "diff", "--binary", base],
                           a.command_timeout)
            if rc or not diff.strip():
                continue
            digest = hashlib.sha1(diff.encode()).hexdigest()
            if digest in seen:  # the command changed nothing a diff can see
                continue
            seen.add(digest)
            (target / f"{k}.patch").write_text(diff)
            written += 1
    finally:
        run(["docker", "rm", "-f", name], 120)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--snapshots", type=int, default=8,
                   help="keep at most this many graded points per episode, the latest ones")
    p.add_argument("--skip-tests", action="store_true", default=True)
    p.add_argument("--run-tests", dest="skip_tests", action="store_false",
                   help="replay the agent's own test and build commands too; slower and it cannot "
                        "change what is graded, but it keeps the replay faithful")
    p.add_argument("--command-timeout", type=int, default=300)
    p.add_argument("--start-timeout", type=int, default=900)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    names = {d.name for d in a.tasks.iterdir() if d.is_dir()}
    episodes = episodes_of(a.compact)
    if a.limit:
        episodes = episodes[: a.limit]
    print(f"[replay] {len(episodes)} episodes, {len(names)} tasks available", flush=True)

    total, done = 0, 0
    started = time.time()
    for index, episode in enumerate(episodes, 1):
        stem = episode["id"].split("__")[0]
        hits = [n for n in names if n == stem or n.startswith(stem)]
        if len(hits) != 1:
            print(f"[replay] {episode['id']}: {len(hits)} matching tasks, skipped", flush=True)
            continue
        written = replay(episode, a.tasks / hits[0], a.out, a)
        total += written
        done += bool(written)
        print(f"[replay] {index}/{len(episodes)} {episode['id'][:34]}: {written} snapshots "
              f"({(time.time() - started) / 60:.1f} min elapsed)", flush=True)
    print(f"[replay] done: {total} patches over {done} episodes -> {a.out}")


if __name__ == "__main__":
    main()
