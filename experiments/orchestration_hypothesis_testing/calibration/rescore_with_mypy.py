#!/usr/bin/env python3
"""Re-score existing calibration data with mypy as L4 critic.

mypy is a cheap, deterministic critic with genuine noise: it catches
type errors but misses logic errors. Unlike L2 (which runs the
fail-to-pass test and is thus a cheap oracle by construction), mypy
doesn't know what the bug is — it just checks types. This gives a
realistic noisy critic for the partial-information regime.

For each patch:
  1. Look up instance in SWE-bench Lite
  2. Checkout base commit in /tmp/calibration_repos/sympy__sympy
  3. Reconstruct modified files from the patch diff
  4. Run mypy on the changed files in isolation
  5. Record pass (no new errors) / fail

Output: raw_results_v3.jsonl with L4_mypy added to critic_results.

Usage:
    python rescore_with_mypy.py --limit 10  # quick test
    python rescore_with_mypy.py              # all 231 patches
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from generate_calibration_data import (  # noqa: E402
    _apply_patch,
    _get_changed_files_from_patch,
    _reset_repo,
    setup_repo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "data" / "raw_results_v2.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "raw_results_v3.jsonl"
)
DEFAULT_WORKDIR = Path("/tmp/calibration_repos")
MYPY_BIN = Path.home() / "miniconda3/envs/swebench_py39/bin/mypy"
MYPY_TIMEOUT = 60


@dataclass(frozen=True)
class MypyResult:
    passed: bool
    detail: str


def _count_mypy_errors(content: str, fname: str) -> tuple[int, str]:
    """Run mypy on file content, return (error_count, summary_detail)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / fname.replace("/", "_")
        tmp_path.write_text(content)
        try:
            result = subprocess.run(
                [
                    str(MYPY_BIN),
                    "--ignore-missing-imports",
                    "--follow-imports=skip",
                    "--no-error-summary",
                    "--show-error-codes",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
                timeout=MYPY_TIMEOUT,
            )
            lines = (result.stdout + result.stderr).strip().split("\n")
            errs = [l for l in lines if ": error:" in l]
            detail = "; ".join(errs[:3])[:300]
            return len(errs), detail
        except subprocess.TimeoutExpired:
            return -1, "mypy timeout"
        except Exception as e:
            return -1, f"mypy error: {str(e)[:100]}"


def compare_mypy(
    original_content: str,
    modified_content: str,
    fname: str,
) -> MypyResult:
    """L4 passes iff the patch does NOT introduce new mypy errors.

    We compare error count before and after the patch. This handles the
    reality that sympy's own codebase already has many pre-existing type
    issues — we only care about whether OUR patch adds new ones.
    """
    orig_errs, _ = _count_mypy_errors(original_content, fname)
    mod_errs, mod_detail = _count_mypy_errors(modified_content, fname)

    if orig_errs < 0 or mod_errs < 0:
        return MypyResult(passed=False, detail=mod_detail)

    if mod_errs <= orig_errs:
        return MypyResult(passed=True, detail=f"errors {orig_errs}->{mod_errs}")
    return MypyResult(
        passed=False,
        detail=f"new errors {orig_errs}->{mod_errs}: {mod_detail}",
    )


def apply_patch_and_read_file(
    repo_path: Path,
    patch: str,
    file_path: str,
) -> str | None:
    """Apply patch in-place, read the modified file content, return it.

    The caller must reset the repo before/after.
    """
    if not patch.strip():
        return None
    result = _apply_patch(repo_path, patch)
    if result.returncode != 0:
        return None
    target = repo_path / file_path
    if not target.exists():
        return None
    try:
        return target.read_text()
    except Exception:
        return None


def rescore(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load existing records
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d records", len(records))

    # Load SWE-bench Lite for base_commit lookup
    log.info("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_id = {d["instance_id"]: d for d in ds}
    log.info("Dataset loaded: %d instances", len(by_id))

    # Filter to the one repo we care about (sympy)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        key = f"{r['instance_id']}_{r['patch_id']}"
                        completed.add(key)
                    except (json.JSONDecodeError, KeyError):
                        pass
        log.info("Resume: %d records already scored", len(completed))

    if args.limit > 0:
        records = records[: args.limit]

    n_y1 = sum(1 for r in records if r["ground_truth"] == 1)
    n_y0 = sum(1 for r in records if r["ground_truth"] == 0)
    n_y1_mypy = 0
    n_y0_mypy = 0
    n_done = 0

    current_repo = None
    current_instance = None

    for i, record in enumerate(records):
        key = f"{record['instance_id']}_{record['patch_id']}"
        if key in completed:
            continue

        instance_id = record["instance_id"]
        instance = by_id.get(instance_id)
        if instance is None:
            log.warning("Instance %s not in dataset, skipping", instance_id)
            continue

        # Setup repo once per instance
        if current_instance != instance_id:
            try:
                current_repo = setup_repo(
                    instance["repo"], instance["base_commit"], workdir
                )
                current_instance = instance_id
            except Exception as e:
                log.error("setup_repo failed for %s: %s", instance_id, e)
                continue

        # Apply patch, extract modified file, run mypy
        patch = record.get("patch", "")
        changed = [
            f for f in _get_changed_files_from_patch(patch) if f.endswith(".py")
        ]

        if not changed:
            l4 = MypyResult(passed=False, detail="no python files in patch")
        else:
            # Capture original file contents BEFORE applying the patch
            _reset_repo(current_repo)
            originals: dict[str, str] = {}
            for fpath in changed[:3]:
                target = current_repo / fpath
                if target.exists():
                    try:
                        originals[fpath] = target.read_text()
                    except Exception:
                        pass

            if not originals:
                l4 = MypyResult(passed=False, detail="no original files readable")
            else:
                apply_ok = _apply_patch(current_repo, patch).returncode == 0
                if not apply_ok:
                    l4 = MypyResult(passed=False, detail="patch apply failed")
                else:
                    any_new_errors = False
                    details = []
                    for fpath, orig in originals.items():
                        target = current_repo / fpath
                        if not target.exists():
                            continue
                        try:
                            modified = target.read_text()
                        except Exception:
                            continue
                        if modified == orig:
                            continue  # no change in this file
                        r = compare_mypy(orig, modified, fpath)
                        if not r.passed:
                            any_new_errors = True
                            details.append(r.detail)
                    if any_new_errors:
                        l4 = MypyResult(passed=False, detail="; ".join(details)[:300])
                    else:
                        l4 = MypyResult(passed=True, detail="no new mypy errors")
                _reset_repo(current_repo)

        record["critic_results"]["L4_mypy"] = {
            "passed": l4.passed,
            "detail": l4.detail,
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        n_done += 1
        if l4.passed:
            if record["ground_truth"] == 1:
                n_y1_mypy += 1
            else:
                n_y0_mypy += 1

        if (i + 1) % 20 == 0:
            log.info(
                "[%d/%d] Y1 mypy-pass=%d Y0 mypy-pass=%d",
                i + 1, len(records), n_y1_mypy, n_y0_mypy,
            )

    log.info("=" * 60)
    log.info("Rescore complete: %d records", n_done)
    y1_count = sum(1 for r in records if r["ground_truth"] == 1)
    y0_count = sum(1 for r in records if r["ground_truth"] == 0)
    if y1_count:
        log.info("P(L4 pass | Y=1) = %d/%d = %.3f", n_y1_mypy, y1_count, n_y1_mypy / y1_count)
    if y0_count:
        log.info("P(L4 pass | Y=0) = %d/%d = %.3f", n_y0_mypy, y0_count, n_y0_mypy / y0_count)
    if y1_count and y0_count:
        gap = (n_y1_mypy / y1_count) - (n_y0_mypy / y0_count)
        log.info("L4 gap = %.3f", gap)
    log.info("Output: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rescore(args)


if __name__ == "__main__":
    main()
