"""Grade a patch against a DeepSWE task, outside pier.

Each task ships its own verifier: tests/Dockerfile is the task image with the
hidden tests baked in, grader.py applies model.patch to a pristine checkout,
runs the suites and writes reward.json with the fail-to-pass and pass-to-pass
counts. Nothing about that needs the agent harness, so an arbitrary patch --
including the repository state partway through a recorded trajectory -- can be
graded the same way the benchmark grades a submission.

``validate`` is the gate everything else waits on: it grades each task's own
golden solution, which must come back with every fail-to-pass test passing. A
task whose golden does not score 1.0 has an environment problem, and grading a
model's patch against it would produce a plausible number with no meaning. This
was not hypothetical -- on an arm64 laptop the amd64 images run under emulation
and even the pass-to-pass tests fail on the untouched base state.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def image_for(task_dir: Path) -> str:
    return f"deepswe-verif:{task_dir.name}"


def reclaim(task_dir: Path | None) -> None:
    """Drop the task image and the build cache it left behind.

    ``docker rmi`` alone is not enough: each verifier image pulls a full language
    runtime, and the layers stay in the buildx/containerd content store, which
    filled the worker's disk after fourteen tasks.
    """
    if task_dir is not None:
        run(["docker", "rmi", "-f", image_for(task_dir)], 300)
    run(["docker", "builder", "prune", "-f"], 600)


def build(task_dir: Path, timeout: int) -> bool:
    rc, out = run(["docker", "build", "-q", "-t", image_for(task_dir), str(task_dir / "tests")],
                  timeout)
    if rc:
        print(f"[grade] build failed for {task_dir.name}: {out.strip()[-300:]}", flush=True)
    return rc == 0


def grade(task_dir: Path, patch: Path | None, work: Path, timeout: int) -> dict | None:
    """Run the task's own verifier over one patch and return its reward.json."""
    artifacts, verifier = work / "artifacts", work / "verifier"
    for d in (artifacts, verifier):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    if patch is not None:
        shutil.copy(patch, artifacts / "model.patch")
    rc, out = run(["docker", "run", "--rm", "--network=none",
                   "-v", f"{artifacts}:/logs/artifacts", "-v", f"{verifier}:/logs/verifier",
                   image_for(task_dir), "bash", "/tests/test.sh"], timeout)
    reward = verifier / "reward.json"
    if not reward.exists():
        print(f"[grade] {task_dir.name}: no reward.json (rc={rc}) {out.strip()[-200:]}", flush=True)
        return None
    return json.loads(reward.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["validate", "grade"])
    p.add_argument("--tasks", type=Path, required=True, help="the deep-swe tasks directory")
    p.add_argument("--work", type=Path, default=Path("/tmp/grade_work"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None, help="task directory names")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--patches", type=Path, default=None,
                   help="grade mode: a directory of <task>/<step>.patch files")
    p.add_argument("--build-timeout", type=int, default=2400)
    p.add_argument("--run-timeout", type=int, default=2400)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    a.work.mkdir(parents=True, exist_ok=True)
    names = sorted(d.name for d in a.tasks.iterdir() if (d / "tests" / "test.sh").exists())
    if a.only:
        names = [n for n in names if n in set(a.only)]
    if a.command == "grade":
        # Only the episodes a replay actually produced snapshots for: building an
        # image for a task with no patches costs a couple of minutes of disk
        # churn and grades nothing.
        #
        # replay names its output directory by episode id, which carries a suffix
        # the task directory does not -- abs-module-cache-flags__A4bbkQC against
        # abs-module-cache-flags -- so a lookup by task name misses every time.
        # The first sharded run completed with "0 of 113" for exactly that reason.
        # Resolve it the way replay does; on DG all 112 episodes resolve to one
        # task each.
        available = len(names)
        known = set(names)
        work = []
        for directory in sorted(d for d in (a.patches.iterdir() if a.patches and a.patches.is_dir()
                                            else []) if d.is_dir()):
            if not any(directory.glob("*.patch")):
                continue
            stem = directory.name.split("__")[0]
            hits = [n for n in known if n == stem or n.startswith(stem)]
            if len(hits) != 1:
                print(f"[grade] {directory.name}: {len(hits)} matching tasks, skipped", flush=True)
                continue
            work.append((hits[0], directory))
        print(f"[grade] {len(work)} episodes with patches, of {available} tasks available",
              flush=True)
    else:
        work = [(n, None) for n in names]
    if a.limit:
        work = work[: a.limit]
    print(f"[grade] {len(work)} to process", flush=True)

    out_path = a.out / ("validation.jsonl" if a.command == "validate" else "labels.jsonl")
    handle = open(out_path, "a")
    ok = bad = 0
    for index, (name, patch_dir) in enumerate(work, 1):
        task_dir = a.tasks / name
        started = time.time()
        if not build(task_dir, a.build_timeout):
            bad += 1
            reclaim(task_dir)  # a failed build leaves its layers behind too
            continue
        if a.command == "validate":
            golden = task_dir / "solution" / "solution.patch"
            reward = grade(task_dir, golden if golden.exists() else None, a.work, a.run_timeout)
            passed = bool(reward and reward.get("f2p") == 1.0 and reward.get("p2p") == 1.0)
            ok += passed
            bad += not passed
            handle.write(json.dumps({"task": name, "golden_ok": passed, "reward": reward,
                                     "seconds": round(time.time() - started, 1)}) + "\n")
            handle.flush()
            print(f"[grade] {index}/{len(work)} {name}: golden "
                  f"{'OK' if passed else 'FAILED'} {reward} ({time.time() - started:.0f}s)",
                  flush=True)
        else:
            for patch in sorted(patch_dir.glob("*.patch"), key=lambda f: int(f.stem)):
                reward = grade(task_dir, patch, a.work, a.run_timeout)
                # the episode id is what joins these labels back to a trajectory;
                # the task name alone cannot do it
                handle.write(json.dumps({"episode": patch_dir.name, "task": name,
                                         "step": int(patch.stem), "reward": reward}) + "\n")
                handle.flush()
                print(f"[grade] {name} step {patch.stem}: "
                      f"{(reward or {}).get('partial')}", flush=True)
        reclaim(task_dir)
    handle.close()
    print(f"[grade] done: {ok} ok, {bad} failed -> {out_path}")


if __name__ == "__main__":
    main()
