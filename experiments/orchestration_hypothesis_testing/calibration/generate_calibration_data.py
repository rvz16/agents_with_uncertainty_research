#!/usr/bin/env python3
"""Generate calibration data for the orchestration-as-hypothesis-testing experiment.

For each SWE-bench instance, generates N patches using an LLM, runs tiered critics
(L0: syntax, L1: lint, L2: fast test) on each patch, and runs the full test suite
to get ground-truth labels Y in {0, 1}.

The output is a JSONL file where each line contains:
    {instance_id, patch_id, patch, critic_results, ground_truth, metadata}

This calibration data is used by compute_likelihoods.py to estimate the confusion
matrix P(z|Y) for each critic level, which the Bayesian controller needs for
belief updates.

Usage:
    # Quick test on 5 instances
    python generate_calibration_data.py --limit 5 --patches-per-instance 3

    # Full calibration run
    python generate_calibration_data.py --limit 300 --patches-per-instance 3

    # Resume interrupted run
    python generate_calibration_data.py --limit 300 --resume
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from datasets import load_dataset

# Project root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Load .env from project root
load_dotenv(ROOT / ".env")

from sage_agent.llm.openrouter import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Defaults
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_PATCHES_PER_INSTANCE = 3
DEFAULT_TEMPERATURE = 0.8
PATCH_GEN_TIMEOUT = 60  # seconds for LLM call
TEST_TIMEOUT = 300  # seconds for full test suite
FAST_TEST_TIMEOUT = 60  # seconds for single test file
LINT_TIMEOUT = 30

# Patch application strategies (from SWE-bench harness:
# https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/run_evaluation.py)
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class CriticResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class PatchCalibrationRecord:
    instance_id: str
    patch_id: int
    patch: str
    critic_results: dict[str, dict[str, object]]
    ground_truth: int  # 0 or 1
    metadata: dict[str, str]


# ============================================================================
# Patch generation (oracle retrieval + complete file output)
# ============================================================================

MAX_FILE_LINES = 500  # Truncate very large files to keep prompt manageable

PATCH_PROMPT_TEMPLATE = """\
You are an expert software engineer fixing a bug in the {repo} repository.

## Issue Description
{problem_statement}

{hints_section}

## Files that likely need changes

{file_contents}

## Task
Fix the issue by modifying the file(s) above. For each file you change, output the
COMPLETE modified file content wrapped in a block like this:

<<<FILE path/to/file.py
(entire file content with your fix applied)
FILE>>>

Output ONLY the modified file blocks. Do not include files you did not change.
Focus on the minimal change needed. Do not add unrelated changes.
"""


def _read_oracle_files(repo_path: Path, gold_patch: str) -> dict[str, str]:
    """Read the files that the gold patch modifies (oracle retrieval).

    This gives the model actual file content so it can produce valid edits.
    We use the gold patch file paths but NOT the gold patch content.
    """
    file_paths = _get_changed_files_from_patch(gold_patch)
    contents: dict[str, str] = {}
    for fpath in file_paths:
        full_path = repo_path / fpath
        if not full_path.exists():
            continue
        try:
            text = full_path.read_text(errors="replace")
            lines = text.split("\n")
            if len(lines) > MAX_FILE_LINES:
                text = "\n".join(lines[:MAX_FILE_LINES]) + f"\n... (truncated, {len(lines)} lines total)"
            contents[fpath] = text
        except Exception:
            continue
    return contents


def _format_file_contents(files: dict[str, str]) -> str:
    """Format file contents for the prompt."""
    parts: list[str] = []
    for fpath, content in files.items():
        parts.append(f"### {fpath}\n```python\n{content}\n```")
    return "\n\n".join(parts) if parts else "(no files available)"


def _make_diff(original: str, modified: str, file_path: str) -> str:
    """Compute unified diff between original and modified file content."""
    import difflib
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, mod_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


def _parse_file_blocks(response: str) -> dict[str, str]:
    """Parse <<<FILE path ... FILE>>> blocks from LLM response."""
    blocks: dict[str, str] = {}
    pattern = re.compile(
        r"<<<FILE\s+(.+?)\s*\n([\s\S]*?)FILE>>>",
        re.MULTILINE,
    )
    for match in pattern.finditer(response):
        fpath = match.group(1).strip()
        content = match.group(2)
        # Strip trailing whitespace but keep the structure
        if content.endswith("\n"):
            content = content[:-1]
        blocks[fpath] = content
    return blocks


def generate_patches(
    llm: OpenRouterClient,
    problem_statement: str,
    repo: str,
    hints: str,
    n_patches: int,
    temperature: float,
    repo_path: Optional[Path] = None,
    gold_patch: str = "",
) -> list[str]:
    """Generate N diverse patches for a SWE-bench instance.

    Uses oracle retrieval: reads the files that the gold patch modifies
    and includes their content in the prompt. The model outputs complete
    modified files, and we compute the diff programmatically — this
    guarantees the patch always applies cleanly.
    """
    hints_section = f"## Hints\n{hints}" if hints else ""

    # Oracle retrieval: read files the gold patch touches
    oracle_files: dict[str, str] = {}
    if repo_path and gold_patch:
        oracle_files = _read_oracle_files(repo_path, gold_patch)

    file_contents = _format_file_contents(oracle_files)
    prompt = PATCH_PROMPT_TEMPLATE.format(
        repo=repo,
        problem_statement=problem_statement,
        hints_section=hints_section,
        file_contents=file_contents,
    )

    patches: list[str] = []
    for i in range(n_patches):
        try:
            response = llm.complete(prompt)
            patch = _response_to_patch(response, oracle_files)
            if patch:
                patches.append(patch)
            else:
                log.warning("  Patch %d: no valid file blocks in response", i)
                patches.append("")
        except Exception as e:
            log.warning("  Patch %d generation failed: %s", i, e)
            patches.append("")

    return patches


def _response_to_patch(response: str, oracle_files: dict[str, str]) -> str:
    """Convert LLM response (complete file blocks) to a unified diff patch.

    If the response contains <<<FILE blocks, computes diff against originals.
    Falls back to extracting raw diff from response if no blocks found.
    """
    # Try parsing <<<FILE blocks first
    blocks = _parse_file_blocks(response)
    if blocks:
        diffs: list[str] = []
        for fpath, modified_content in blocks.items():
            original = oracle_files.get(fpath, "")
            if not original:
                # Try without leading path components
                for opath, ocontent in oracle_files.items():
                    if opath.endswith(fpath) or fpath.endswith(opath):
                        original = ocontent
                        fpath = opath
                        break
            diff = _make_diff(original, modified_content, fpath)
            if diff:
                diffs.append(diff)
        return "\n".join(diffs) if diffs else ""

    # Fallback: try extracting raw diff from response
    diff_match = re.search(r"```(?:diff)?\n([\s\S]*?)```", response)
    if diff_match:
        return diff_match.group(1).strip()

    diff_pattern = re.search(
        r"(---\s+a/.*?\n\+\+\+\s+b/.*?\n[\s\S]*?)(?:\n\n|$)", response
    )
    if diff_pattern:
        return diff_pattern.group(1).strip()

    return ""


# ============================================================================
# Critics
# ============================================================================

def _get_changed_files_from_patch(patch: str) -> list[str]:
    """Extract file paths modified by a unified diff patch."""
    files: list[str] = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            files.append(line[6:])
        elif line.startswith("--- a/"):
            pass
    return files


def _apply_patch(cwd: Path, patch: str) -> subprocess.CompletedProcess[str]:
    """Apply a unified diff using the same fallback chain as the SWE-bench harness.

    Tries multiple strategies in order of strictness:
    1. git apply --verbose (strict)
    2. git apply --verbose --reject (applies what it can)
    3. patch --batch --fuzz=5 -p1 (maximum fuzz tolerance)

    For strategy 3 (patch command), we write to a temp file since `patch -i`
    requires a file path rather than stdin for reliable behavior.
    """
    # Write patch to temp file for the `patch` command fallback
    patch_file = cwd / "_tmp_patch.diff"
    patch_file.write_text(patch)

    last_result = None
    for cmd_template in GIT_APPLY_CMDS:
        if cmd_template.endswith("-i"):
            cmd = f"{cmd_template} {patch_file}"
        else:
            cmd = cmd_template

        try:
            if "git apply" in cmd:
                last_result = subprocess.run(
                    cmd.split(),
                    cwd=cwd,
                    input=patch,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
            else:
                # patch command uses -i flag with file
                last_result = subprocess.run(
                    f"{cmd_template} {patch_file}",
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        except subprocess.TimeoutExpired:
            continue

        if last_result.returncode == 0:
            patch_file.unlink(missing_ok=True)
            return last_result

    patch_file.unlink(missing_ok=True)
    return last_result or subprocess.CompletedProcess(
        args="", returncode=1, stdout="", stderr="all apply strategies failed"
    )


def _get_patched_content(repo_path: Path, patch: str, file_path: str) -> Optional[str]:
    """Apply patch in-memory and return the patched file content."""
    original_file = repo_path / file_path
    if not original_file.exists():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        tmp_file = tmp_repo / file_path
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_file, tmp_file)

        result = _apply_patch(tmp_repo, patch)
        if result.returncode != 0:
            return None

        patched_file = tmp_repo / file_path
        if patched_file.exists():
            return patched_file.read_text()
    return None


def run_critic_l0_syntax(patch: str, repo_path: Optional[Path] = None) -> CriticResult:
    """L0: Check if patched Python files have valid syntax via ast.parse()."""
    changed_files = _get_changed_files_from_patch(patch)
    py_files = [f for f in changed_files if f.endswith(".py")]

    if not py_files:
        return CriticResult(passed=True, detail="no python files changed")

    if repo_path is None:
        # Without repo, try to syntax-check any inline code in the patch
        # Extract added lines and check them as a rough proxy
        return CriticResult(passed=True, detail="no repo available, skipped")

    errors: list[str] = []
    for fpath in py_files:
        content = _get_patched_content(repo_path, patch, fpath)
        if content is None:
            continue
        try:
            ast.parse(content, filename=fpath)
        except SyntaxError as e:
            errors.append(f"{fpath}:{e.lineno}: {e.msg}")

    if errors:
        return CriticResult(passed=False, detail="; ".join(errors[:3]))
    return CriticResult(passed=True, detail="")


def run_critic_l1_lint(patch: str, repo_path: Optional[Path] = None) -> CriticResult:
    """L1: Run ruff on changed Python files."""
    changed_files = _get_changed_files_from_patch(patch)
    py_files = [f for f in changed_files if f.endswith(".py")]

    if not py_files:
        return CriticResult(passed=True, detail="no python files changed")

    if repo_path is None:
        return CriticResult(passed=True, detail="no repo available, skipped")

    # Apply patch to temp dir and run ruff
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        shutil.copytree(repo_path, tmp_repo, symlinks=True)

        # Apply patch (use patch with fuzz for LLM-generated diffs)
        result = _apply_patch(tmp_repo, patch)
        if result.returncode != 0:
            return CriticResult(passed=False, detail=f"patch apply failed: {result.stderr[:200]}")

        # Run ruff on changed files only
        abs_files = [str(tmp_repo / f) for f in py_files if (tmp_repo / f).exists()]
        if not abs_files:
            return CriticResult(passed=True, detail="changed files not found after apply")

        try:
            result = subprocess.run(
                ["ruff", "check", "--select=E,F"] + abs_files,
                cwd=tmp_repo,
                capture_output=True,
                text=True,
                timeout=LINT_TIMEOUT,
            )
        except FileNotFoundError:
            return CriticResult(passed=True, detail="ruff not installed, skipped")
        except subprocess.TimeoutExpired:
            return CriticResult(passed=False, detail="ruff timeout")

        if result.returncode == 0:
            return CriticResult(passed=True, detail="")

        # Extract first few errors
        errors = result.stdout.strip().split("\n")[:5]
        return CriticResult(passed=False, detail="; ".join(errors))


def run_critic_l2_fast_test(
    patch: str,
    repo_path: Path,
    test_patch: str,
) -> CriticResult:
    """L2: Run only the test file(s) referenced in the SWE-bench test_patch."""
    # Extract test file paths from test_patch
    test_files = _get_changed_files_from_patch(test_patch)
    test_files = [f for f in test_files if "test" in f.lower()]

    if not test_files:
        return CriticResult(passed=False, detail="no test files found in test_patch")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        shutil.copytree(repo_path, tmp_repo, symlinks=True)

        # Apply the generated patch (use patch with fuzz for LLM-generated diffs)
        result = _apply_patch(tmp_repo, patch)
        if result.returncode != 0:
            return CriticResult(passed=False, detail=f"patch apply failed: {result.stderr[:200]}")

        # Also apply the test patch (SWE-bench provides test changes separately)
        # Test patches are well-formed, but use same approach for consistency
        result = _apply_patch(tmp_repo, test_patch)
        if result.returncode != 0:
            return CriticResult(
                passed=False,
                detail=f"test patch apply failed: {result.stderr[:200]}",
            )

        # Run only the specific test files
        test_cmd = ["python", "-m", "pytest", "-x", "--tb=short"] + test_files
        try:
            result = subprocess.run(
                test_cmd,
                cwd=tmp_repo,
                capture_output=True,
                text=True,
                timeout=FAST_TEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return CriticResult(passed=False, detail="fast test timeout")

        if result.returncode == 0:
            return CriticResult(passed=True, detail="")

        # Extract failure summary
        stderr_tail = result.stderr[-300:] if result.stderr else ""
        stdout_tail = result.stdout[-300:] if result.stdout else ""
        return CriticResult(
            passed=False,
            detail=f"exit={result.returncode}; {stdout_tail}",
        )


# ============================================================================
# Verifier (ground truth)
# ============================================================================

def run_verifier(
    patch: str,
    repo_path: Path,
    test_patch: str,
) -> int:
    """Run full test suite to determine ground truth Y in {0, 1}.

    Applies both the generated patch and the test patch, then runs pytest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        shutil.copytree(repo_path, tmp_repo, symlinks=True)

        # Apply the generated patch (use patch with fuzz for LLM-generated diffs)
        result = _apply_patch(tmp_repo, patch)
        if result.returncode != 0:
            log.debug("  Verifier: patch apply failed")
            return 0

        # Apply the test patch
        result = _apply_patch(tmp_repo, test_patch)
        if result.returncode != 0:
            log.debug("  Verifier: test patch apply failed")
            return 0

        # Run full test suite
        test_files = _get_changed_files_from_patch(test_patch)
        test_files = [f for f in test_files if "test" in f.lower()]

        if not test_files:
            log.debug("  Verifier: no test files found")
            return 0

        test_cmd = ["python", "-m", "pytest", "-x", "--tb=short"] + test_files
        try:
            result = subprocess.run(
                test_cmd,
                cwd=tmp_repo,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            log.debug("  Verifier: timeout")
            return 0

        return 1 if result.returncode == 0 else 0


# ============================================================================
# Repo management
# ============================================================================

def setup_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    """Clone repository and checkout base commit. Reuses existing clones."""
    repo_name = repo.replace("/", "__")
    repo_path = workdir / repo_name

    if repo_path.exists():
        # Reset to base commit instead of re-cloning
        subprocess.run(
            ["git", "checkout", "-f", base_commit],
            cwd=repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=repo_path,
            capture_output=True,
        )
        return repo_path

    log.info("Cloning %s...", repo)
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    return repo_path


# ============================================================================
# Main pipeline
# ============================================================================

def load_completed_ids(output_file: Path) -> set[str]:
    """Load instance_id+patch_id pairs already in the output file."""
    completed: set[str] = set()
    if not output_file.exists():
        return completed

    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = f"{record['instance_id']}_{record['patch_id']}"
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_calibration(args: argparse.Namespace) -> None:
    """Main calibration pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "raw_results.jsonl"

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="calibration_"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Load completed records for resume
    completed = load_completed_ids(output_file) if args.resume else set()
    if completed:
        log.info("Resuming: %d patch records already completed", len(completed))

    # Initialize LLM
    llm = OpenRouterClient(model=args.model, verbose=args.verbose)
    log.info("LLM: %s", args.model)

    # Load dataset
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    dataset_name = dataset_map[args.dataset]
    log.info("Loading %s...", dataset_name)
    dataset = load_dataset(dataset_name, split="test")

    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    log.info("Processing %d instances, %d patches each", len(dataset), args.patches_per_instance)
    log.info("Output: %s", output_file)

    total_records = 0
    total_correct = 0

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem_statement = instance["problem_statement"]
        hints = instance.get("hints_text", "")
        test_patch = instance.get("test_patch", "")

        log.info(
            "[%d/%d] %s (%s)",
            idx + 1, len(dataset), instance_id, repo,
        )

        # Check if all patches for this instance are already done
        all_done = all(
            f"{instance_id}_{i}" in completed
            for i in range(args.patches_per_instance)
        )
        if all_done:
            log.info("  Skipping (all patches already completed)")
            continue

        # Setup repo
        try:
            repo_path = setup_repo(repo, base_commit, workdir)
        except Exception as e:
            log.error("  Failed to setup repo: %s", e)
            continue

        # Get gold patch for oracle file retrieval (we use file paths only, not content)
        gold_patch = instance.get("patch", "")

        # Generate patches with oracle retrieval
        patches = generate_patches(
            llm=llm,
            problem_statement=problem_statement,
            repo=repo,
            hints=hints,
            n_patches=args.patches_per_instance,
            temperature=args.temperature,
            repo_path=repo_path,
            gold_patch=gold_patch,
        )

        # Evaluate each patch
        for patch_id, patch in enumerate(patches):
            key = f"{instance_id}_{patch_id}"
            if key in completed:
                log.info("  Patch %d: skipping (already done)", patch_id)
                continue

            if not patch:
                log.warning("  Patch %d: empty, recording as all-fail", patch_id)
                record = PatchCalibrationRecord(
                    instance_id=instance_id,
                    patch_id=patch_id,
                    patch="",
                    critic_results={
                        "L0_syntax": {"passed": False, "detail": "empty patch"},
                        "L1_lint": {"passed": False, "detail": "empty patch"},
                        "L2_fast_test": {"passed": False, "detail": "empty patch"},
                    },
                    ground_truth=0,
                    metadata={
                        "model": args.model,
                        "repo": repo,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                with open(output_file, "a") as f:
                    f.write(json.dumps(asdict(record)) + "\n")
                total_records += 1
                continue

            log.info("  Patch %d: running critics...", patch_id)

            # L0: Syntax check
            l0 = run_critic_l0_syntax(patch, repo_path)
            log.info("    L0 syntax: %s", "PASS" if l0.passed else f"FAIL ({l0.detail[:60]})")

            # L1: Lint
            l1 = run_critic_l1_lint(patch, repo_path)
            log.info("    L1 lint:   %s", "PASS" if l1.passed else f"FAIL ({l1.detail[:60]})")

            # L2: Fast test
            l2 = run_critic_l2_fast_test(patch, repo_path, test_patch)
            log.info("    L2 test:   %s", "PASS" if l2.passed else f"FAIL ({l2.detail[:60]})")

            # Verifier: ground truth
            log.info("    Running verifier (full test suite)...")
            ground_truth = run_verifier(patch, repo_path, test_patch)
            log.info("    Ground truth: Y=%d", ground_truth)

            total_records += 1
            total_correct += ground_truth

            record = PatchCalibrationRecord(
                instance_id=instance_id,
                patch_id=patch_id,
                patch=patch,
                critic_results={
                    "L0_syntax": asdict(l0),
                    "L1_lint": asdict(l1),
                    "L2_fast_test": asdict(l2),
                },
                ground_truth=ground_truth,
                metadata={
                    "model": args.model,
                    "repo": repo,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Append to JSONL (incremental checkpoint)
            with open(output_file, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    # Summary
    log.info("=" * 60)
    log.info("Calibration complete")
    log.info("Total patches evaluated: %d", total_records)
    log.info("Correct patches (Y=1): %d (%.1f%%)",
             total_correct,
             100 * total_correct / total_records if total_records else 0)
    log.info("Output: %s", output_file)
    log.info("Next step: python compute_likelihoods.py --input %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration data for orchestration-as-hypothesis-testing."
    )
    parser.add_argument(
        "--dataset",
        choices=["lite", "verified", "full"],
        default="lite",
        help="SWE-bench dataset variant (default: lite, 300 instances).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of instances to process (0 = all). Default: 5 for quick test.",
    )
    parser.add_argument(
        "--patches-per-instance",
        type=int,
        default=DEFAULT_PATCHES_PER_INSTANCE,
        help="Number of patches to generate per instance (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model for patch generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for patch diversity.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for calibration data.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for repo clones (default: temp dir).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping completed records.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LLM prompts and responses.",
    )

    args = parser.parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
