"""Sharded SWE-Bench harness invocation across multiple SSH hosts.

Drop-in alternative to a single `python -m swebench.harness.run_evaluation`
call. Splits the predictions file into N equal chunks (by sorted instance_id),
runs one chunk per host in parallel, then merges the per-shard harness
reports into one canonical report file at the path the caller expects.

Why
---
The harness is forced to ``--max-workers=1`` (containerd metadata race) so a
single host is bottlenecked by Docker / podman serial throughput, ~60s per
Lite instance and ~75s per Verified instance. With N hosts the wall-clock
drops by close to N for everything except the cold env-image build on a
fresh shard host. The shards are disjoint by instance_id so per-instance
logs from different shards can be merged into one logs/run_evaluation/<run_id>
tree without collision.

Activation
----------
Set ``EVAL_SHARDS`` to a comma-separated list of remote hosts. The local
host is *always* shard 0; the remote hosts become shards 1..N.

  EVAL_SHARDS=mbz3                  # 2 shards: [local, mbz3]
  EVAL_SHARDS=mbz3,mbz4             # 3 shards: [local, mbz3, mbz4]

Each remote host token may be enriched with an explicit repo root and
python binary if the defaults below do not apply:

  EVAL_SHARDS=mbz3:/path/to/repo:/path/to/python

When ``EVAL_SHARDS`` is empty / unset, ``run_swebench_eval`` in
``spot_check_generators`` behaves exactly as before (single-host).

Each remote host MUST already have:
  * The same git checkout under the resolved repo root.
  * The matching swebench install + ``scripts/patch_swebench_harness.py``
    applied (idempotent).
  * Rootless podman socket reachable at
    ``unix:///run/user/$(id -u)/podman/podman.sock``.
  * ``.env`` and TMPDIR configured the same way as the orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

DEFAULT_REMOTE_REPO = "/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research"
DEFAULT_REMOTE_PYTHON = "/home/vlad.smirnov/miniconda3/bin/python"
DEFAULT_REMOTE_PODMAN_SOCKET = "unix:///run/user/$(id -u)/podman/podman.sock"


@dataclass(frozen=True)
class Shard:
    """One execution target. ``host=None`` ⇒ run locally without SSH."""
    host: str | None
    remote_repo: str
    remote_python: str

    @property
    def is_local(self) -> bool:
        return self.host is None

    @property
    def label(self) -> str:
        return self.host or "local"


def parse_shards_env(value: str | None) -> list[Shard]:
    """Parse the ``EVAL_SHARDS`` env value into a Shard list.

    Always returns at least the local shard. Returns ``[]`` if value is
    empty (caller treats that as "no sharding, run plain harness").
    """
    if value is None or not value.strip():
        return []
    shards: list[Shard] = [Shard(host=None, remote_repo="", remote_python="")]
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) == 1:
            host, repo, py = parts[0], DEFAULT_REMOTE_REPO, DEFAULT_REMOTE_PYTHON
        elif len(parts) == 3:
            host, repo, py = parts
        else:
            raise ValueError(
                f"EVAL_SHARDS token {tok!r} must be 'host' or "
                "'host:remote_repo:remote_python'"
            )
        shards.append(Shard(host=host, remote_repo=repo, remote_python=py))
    return shards


def _split_predictions(
    predictions_path: Path, n_shards: int, scratch: Path
) -> tuple[list[Path], list[list[dict]], str]:
    """Sort predictions by instance_id and split into ``n_shards`` shard files.

    Returns the per-shard file paths, the per-shard row lists, and the
    common ``model_name_or_path`` (used to build the canonical report
    filename).
    """
    rows: list[dict] = []
    for line in predictions_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty predictions file: {predictions_path}")
    rows.sort(key=lambda r: r["instance_id"])
    models = {r.get("model_name_or_path") for r in rows}
    if len(models) != 1:
        raise ValueError(
            f"predictions must share one model_name_or_path; got {models}"
        )
    model = next(iter(models))

    chunks: list[list[dict]] = []
    paths: list[Path] = []
    n = len(rows)
    for i in range(n_shards):
        start = (n * i) // n_shards
        end = (n * (i + 1)) // n_shards
        chunk = rows[start:end]
        chunks.append(chunk)
        shard_path = scratch / f"predictions_shard{i}.jsonl"
        with shard_path.open("w") as f:
            for row in chunk:
                f.write(json.dumps(row) + "\n")
        paths.append(shard_path)
    return paths, chunks, model


def _ssh(host: str, remote_cmd: str, *, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", host, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _rsync_to(local: Path, host: str, remote_path: str) -> None:
    subprocess.run(
        ["rsync", "-a", "--mkpath", str(local), f"{host}:{remote_path}"],
        check=True,
    )


def _rsync_from(host: str, remote_path: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", f"{host}:{remote_path}", str(local)],
        check=True,
    )


def _build_harness_cmd(
    python: str, predictions_abs: str, run_id: str, dataset_name: str
) -> list[str]:
    """Mirror the cmd used by spot_check_generators.run_swebench_eval."""
    return [
        python, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", predictions_abs,
        "--max_workers", "1",
        "--run_id", run_id,
        "--cache_level", "env",
        "--namespace", "none",
    ]


def _run_local_shard(
    shard_path: Path,
    shard_run_id: str,
    dataset_name: str,
    work_dir: Path,
    model: str,
) -> Path:
    """Run harness in-process via subprocess, like the original wrapper."""
    work_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the podman shim machinery from spot_check_generators by importing
    # late (the caller has typically already set DOCKER_HOST etc.)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import spot_check_generators as scg  # noqa: E402

    env = os.environ.copy()
    docker_host = env.get("DOCKER_HOST", "")
    if "podman" in docker_host.lower() or env.get("SWEBENCH_PODMAN_COMPAT"):
        shim_dir = scg._write_podman_shim(work_dir / ".podman_compat")
        env["PYTHONPATH"] = (
            str(shim_dir) + os.pathsep + env.get("PYTHONPATH", "")
        )

    cmd = _build_harness_cmd(sys.executable, str(shard_path.resolve()),
                             shard_run_id, dataset_name)
    log.info("local shard %s: %s", shard_run_id, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
    (work_dir / "eval_logs").mkdir(parents=True, exist_ok=True)
    (work_dir / "eval_logs" / f"{shard_run_id}.stdout.log").write_text(proc.stdout)
    (work_dir / "eval_logs" / f"{shard_run_id}.stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        log.warning("local shard %s exit=%d", shard_run_id, proc.returncode)
        log.warning("stderr tail:\n%s", proc.stderr[-2000:])
    log.info("local shard %s stdout tail:\n%s", shard_run_id, proc.stdout[-2000:])
    report_path = work_dir / f"{model}.{shard_run_id}.json"
    if not report_path.exists():
        raise RuntimeError(
            f"local shard {shard_run_id}: no report at {report_path}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return report_path


def _run_remote_shard(
    shard: Shard,
    shard_path: Path,
    shard_run_id: str,
    dataset_name: str,
    local_work_dir: Path,
    model: str,
) -> Path:
    """rsync shard predictions to ``shard.host``, run the harness over SSH,
    rsync back the report + per-instance log directory.
    """
    host = shard.host  # type: ignore[assignment]
    assert host is not None
    # Use a per-(host, run_id) work dir on the remote so concurrent shards
    # for different run_ids do not stomp each other.
    remote_work = (
        f"{shard.remote_repo}/experiments/orchestration_hypothesis_testing/"
        f".eval_shards/{shard_run_id}"
    )
    remote_predictions = f"{remote_work}/predictions.jsonl"

    # 1. Ensure remote work dir exists.
    res = _ssh(host, f"mkdir -p {remote_work} && rm -rf {remote_work}/logs "
                     f"{remote_work}/eval_logs {remote_work}/*.json")
    if res.returncode != 0:
        raise RuntimeError(f"shard {host}: mkdir failed: {res.stderr}")

    # 2. rsync predictions over.
    _rsync_to(shard_path, host, remote_predictions)

    # 3. Run harness over SSH. The remote shell needs DOCKER_HOST,
    #    SWEBENCH_PODMAN_COMPAT, SWEBENCH_NAMESPACE, TMPDIR / BUILDAH_TMPDIR
    #    sourced from ~/.bashrc — invoke via `bash -lc` so the user's
    #    interactive env (which the operator already configured) loads.
    cmd = _build_harness_cmd(shard.remote_python, remote_predictions,
                             shard_run_id, dataset_name)
    # Quote each arg for safe bash transport. ``shlex.join`` would emit POSIX
    # single-quoting which is correct here.
    import shlex
    cmd_str = shlex.join(cmd)
    env_setup = (
        f"export DOCKER_HOST={DEFAULT_REMOTE_PODMAN_SOCKET}; "
        "export SWEBENCH_PODMAN_COMPAT=1; "
        "export SWEBENCH_NAMESPACE=none; "
    )
    remote_invocation = (
        f"set -e; cd {remote_work}; "
        f"{env_setup}"
        f"{cmd_str} > stdout.log 2> stderr.log; echo $? > rc"
    )
    log.info("remote shard %s: launching harness", host)
    res = _ssh(host, f"bash -lc {shlex.quote(remote_invocation)}", timeout=None)
    # Always rsync back logs even on failure, for debugging.
    eval_logs_local = local_work_dir / "eval_logs"
    eval_logs_local.mkdir(parents=True, exist_ok=True)
    try:
        _rsync_from(host, f"{remote_work}/stdout.log",
                    eval_logs_local / f"{shard_run_id}.stdout.log")
        _rsync_from(host, f"{remote_work}/stderr.log",
                    eval_logs_local / f"{shard_run_id}.stderr.log")
    except subprocess.CalledProcessError:
        pass  # remote stdout/stderr may be absent if ssh died mid-run
    if res.returncode != 0:
        raise RuntimeError(
            f"shard {host}: ssh harness exit={res.returncode}\n"
            f"stderr: {res.stderr[-1000:]}"
        )

    # 4. rsync back the per-shard report file (and the per-instance log tree
    #    so debugging still works after the shards are merged).
    remote_report = f"{remote_work}/{model}.{shard_run_id}.json"
    local_report = local_work_dir / f"{model}.{shard_run_id}.json"
    _rsync_from(host, remote_report, local_report)
    # Per-instance logs: shard wrote to remote_work/logs/run_evaluation/<shard_run_id>/<model>/.
    remote_logs = f"{remote_work}/logs/run_evaluation/{shard_run_id}/"
    local_logs_dir = local_work_dir / "logs" / "run_evaluation" / shard_run_id
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["rsync", "-a", f"{host}:{remote_logs}", str(local_logs_dir) + "/"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning("shard %s: rsync logs failed (continuing): %s", host, e)

    return local_report


def _merge_reports(shard_reports: list[dict]) -> dict:
    """Sum the count fields, union the id lists. Shards have disjoint
    instance_ids by construction. Non-count scalars (e.g. ``schema_version``)
    are taken from the first shard; they must match across shards anyway.
    """
    if not shard_reports:
        return {}
    # Per the SWE-bench harness report schema, the count fields are exactly
    # the int-typed keys ending in ``_instances``. Everything else int (today
    # just ``schema_version``) is a metadata scalar that should be preserved
    # as-is, not summed.
    merged: dict = {}
    first = shard_reports[0]
    for k, v in first.items():
        if isinstance(v, int):
            if k.endswith("_instances"):
                merged[k] = sum(r.get(k, 0) for r in shard_reports)
            else:
                merged[k] = v
        elif isinstance(v, list):
            seen: set = set()
            for r in shard_reports:
                for x in r.get(k, []):
                    seen.add(x)
            merged[k] = sorted(seen)
        else:
            merged[k] = v
    merged["_shard_meta"] = {
        "n_shards": len(shard_reports),
        "per_shard_resolved": [r.get("resolved_instances", 0) for r in shard_reports],
        "per_shard_total": [r.get("total_instances", 0) for r in shard_reports],
    }
    return merged


def _consolidate_logs(
    work_dir: Path, run_id: str, shard_run_ids: list[str], model: str
) -> None:
    """Move per-instance logs from each shard-specific run_id subdir into the
    canonical run_id subdir.

    Shard layouts:
        work_dir/logs/run_evaluation/<shard_run_id>/<model>/<instance_id>/...
    Canonical layout (what refine_swe / from_spotcheck consult):
        work_dir/logs/run_evaluation/<run_id>/<model>/<instance_id>/...
    """
    canonical = work_dir / "logs" / "run_evaluation" / run_id / model
    canonical.mkdir(parents=True, exist_ok=True)
    for shard_run_id in shard_run_ids:
        src = work_dir / "logs" / "run_evaluation" / shard_run_id / model
        if not src.exists():
            continue
        for inst_dir in src.iterdir():
            dst = canonical / inst_dir.name
            if dst.exists():
                # Should not happen for disjoint shards; preserve both just in case.
                continue
            shutil.move(str(inst_dir), str(dst))
        # Drop the now-empty shard dir tree.
        try:
            shutil.rmtree(work_dir / "logs" / "run_evaluation" / shard_run_id)
        except OSError:
            pass


def run_sharded(
    predictions_path: Path,
    run_id: str,
    work_dir: Path,
    dataset_name: str,
    shards: list[Shard],
) -> Path:
    """Run one harness evaluation, sharded across ``shards``, return the
    canonical merged report path under ``work_dir``.

    Drop-in replacement for ``run_swebench_eval`` when ``EVAL_SHARDS`` is set.
    """
    predictions_path = predictions_path.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    scratch = work_dir / ".eval_shards_scratch" / run_id
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    shard_paths, _chunks, model = _split_predictions(
        predictions_path, len(shards), scratch
    )

    shard_run_ids = [f"{run_id}_shard{i}" for i in range(len(shards))]
    log.info(
        "sharding %s: %d shards [%s]  predictions=%s  work_dir=%s",
        run_id, len(shards), ", ".join(s.label for s in shards),
        predictions_path, work_dir,
    )

    t0 = time.time()
    shard_report_paths: list[Path | None] = [None] * len(shards)
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futs = {}
        for i, shard in enumerate(shards):
            if shard.is_local:
                fut = pool.submit(
                    _run_local_shard, shard_paths[i], shard_run_ids[i],
                    dataset_name, work_dir, model,
                )
            else:
                fut = pool.submit(
                    _run_remote_shard, shard, shard_paths[i], shard_run_ids[i],
                    dataset_name, work_dir, model,
                )
            futs[fut] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                shard_report_paths[i] = fut.result()
                log.info("shard %d (%s) done in %ds",
                         i, shards[i].label, int(time.time() - t0))
            except Exception as e:
                log.exception("shard %d (%s) failed", i, shards[i].label)
                errors.append(e)

    if errors:
        raise RuntimeError(
            f"{len(errors)}/{len(shards)} shard(s) failed; first error: {errors[0]}"
        )

    # Merge reports.
    shard_reports = [json.loads(p.read_text()) for p in shard_report_paths if p]
    merged = _merge_reports(shard_reports)
    canonical = work_dir / f"{model}.{run_id}.json"
    canonical.write_text(json.dumps(merged, indent=2))

    # Move per-instance log directories under the canonical run_id.
    _consolidate_logs(work_dir, run_id, shard_run_ids, model)

    # Keep shard reports as .pre_merge.bak so we can audit the merge.
    for i, p in enumerate(shard_report_paths):
        if p and p.exists():
            try:
                p.rename(p.with_suffix(".json.shard_bak"))
            except OSError:
                pass

    log.info("sharded eval done in %ds: %s", int(time.time() - t0), canonical)
    return canonical
