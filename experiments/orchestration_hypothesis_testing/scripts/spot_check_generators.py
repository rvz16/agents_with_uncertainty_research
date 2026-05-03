#!/usr/bin/env python3
"""Spot-check generator base rate on SWE-bench Lite.

Goal: pick R1's default generator (per PRE_REGISTRATION.md S4) by measuring
the empirical base rate `P(Y=1)` of three candidates on a 20-instance random
sample of SWE-bench Lite. The pre-registration requires the chosen generator
to land base rate in `[0.30, 0.70]`; outside that window the controller has
no real decisions to make.

Pipeline
--------
1. Sample N_INSTANCES (default 20) from SWE-bench Lite with seed=42.
2. For each generator x instance x patch_id (default 3 per instance):
     - Read oracle files (the files the gold patch touches)
     - Prompt the generator with problem + oracle file content
     - Parse SEARCH/REPLACE blocks into a unified diff
3. Submit predictions to SWE-bench Docker harness for ground-truth Y.
4. Compute base rate per generator and write summary.

Usage
-----
    # full run
    python spot_check_generators.py

    # quick smoke test with 2 instances
    python spot_check_generators.py --n-instances 2 --n-patches 1

    # only generate (skip eval); useful to verify generation works first
    python spot_check_generators.py --skip-eval

    # only evaluate existing predictions (after a generate-only run)
    python spot_check_generators.py --skip-generate
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

# Project root (the worktree we're running from)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Load .env: try worktree root first, then walk up looking for a non-empty
# .env. When running inside a git worktree the OPENROUTER_API_KEY typically
# lives in the parent checkout's .env, not the worktree's.
def _load_env_chain() -> None:
    candidates = [ROOT / ".env"]
    cur = ROOT.parent
    for _ in range(5):
        candidates.append(cur / ".env")
        if cur.parent == cur:
            break
        cur = cur.parent
    for env_path in candidates:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)


_load_env_chain()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spot_check")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "spot_check"

# Generators under test. Each entry maps short name -> (model_id, base_url).
# A None base_url means use OpenRouter; otherwise it's a vLLM-served local
# OpenAI-compatible endpoint (no API key needed).
GENERATORS: dict[str, tuple[str, str | None]] = {
    "haiku45":   ("anthropic/claude-haiku-4.5", None),
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "http://127.0.0.1:8001/v1"),
    "qwen3_8b":  ("Qwen/Qwen3-8B",            "http://127.0.0.1:8002/v1"),
}

DEFAULT_N_INSTANCES = 20
DEFAULT_N_PATCHES = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 42
DEFAULT_MAX_WORKERS_GEN = 8     # parallel API calls
DEFAULT_MAX_WORKERS_EVAL = 4    # parallel Docker containers
LLM_TIMEOUT_S = 90
# No real truncation: 10K lines covers all SWE-bench Lite files except the very
# largest django files. Truncating earlier corrupts SEARCH matching because the
# model edits code that isn't in its prompt. Models with small context windows
# (e.g. Qwen2.5-7B on OpenRouter at 32K) will fail with HTTP 400 on big files;
# that's an informative outcome we report rather than paper over.
MAX_FILE_LINES = 10000

PROMPT_TEMPLATE = """\
You are an expert software engineer fixing a bug in the {repo} repository.

## Issue Description
{problem_statement}

{hints_section}

## Files that likely need changes

{file_contents}

## Task
Fix the issue by modifying the file(s) above. Output your changes as a SEARCH/REPLACE
block for each change:

<<<CHANGE path/to/file.py
SEARCH
(exact lines from the original file that you want to replace)
REPLACE
(the new lines that should replace the search block)
CHANGE>>>

You can output multiple CHANGE blocks if needed. The SEARCH text must match the
original file exactly (including indentation). Keep changes minimal and focused.
Do NOT include unchanged files. Do NOT output entire files.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationRecord:
    generator_key: str
    generator_model: str
    instance_id: str
    repo: str
    patch_id: int
    diff: str
    response_chars: int
    error: str = ""
    timestamp: str = ""


@dataclass
class GeneratorSummary:
    generator_key: str
    generator_model: str
    n_instances: int
    n_patches_attempted: int
    n_patches_nonempty: int
    n_patches_evaluated: int
    n_correct: int
    base_rate: float
    base_rate_per_instance: float  # mean of per-instance pass-rate
    by_repo: dict[str, dict[str, float | int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Patch generation (oracle retrieval, repo-free)
# ---------------------------------------------------------------------------

def get_changed_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a unified diff."""
    files: list[str] = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            files.append(line[6:])
    return files


def _strip_fence(text: str) -> str:
    """Strip a single surrounding ```...``` fence (or leading/trailing fence
    fragments) from a code block body."""
    text = text.strip("\n")
    # Whole-body fenced: ```lang\n...\n```
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n([\s\S]*?)\n?```\s*$", text)
    if m:
        return m.group(1).strip("\n")
    # Trailing-only fence
    text = re.sub(r"\n?```\s*$", "", text).rstrip("\n")
    # Leading-only fence
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", text)
    return text


def parse_change_blocks(response: str) -> list[tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks, tolerant of multiple sloppy variants.

    Accepts:
      <<<CHANGE path/to/file
      SEARCH
      ...
      REPLACE
      ...
      CHANGE>>>

    and the looser forms small models emit (often inside ```diff fences),
    including bodies whose SEARCH/REPLACE payloads are themselves fenced:

      CHANGE path/to/file
      SEARCH
      ```python
      ...
      ```
      REPLACE
      ```python
      ...
      ```
    """
    out: list[tuple[str, str, str]] = []

    # Strategy 1: canonical <<<CHANGE ... CHANGE>>>
    canonical = re.compile(
        r"<<<CHANGE\s+(.+?)\s*\n([\s\S]*?)CHANGE>>>", re.MULTILINE
    )
    consumed: list[tuple[int, int]] = []
    for match in canonical.finditer(response):
        fpath = match.group(1).strip()
        body = match.group(2)
        parts = re.split(r"^\s*SEARCH\s*$", body, maxsplit=1, flags=re.MULTILINE)
        if len(parts) < 2:
            continue
        rest = parts[1]
        parts2 = re.split(r"^\s*REPLACE\s*$", rest, maxsplit=1, flags=re.MULTILINE)
        if len(parts2) < 2:
            continue
        out.append((
            fpath,
            _strip_fence(parts2[0]).strip("\n"),
            _strip_fence(parts2[1]).strip("\n"),
        ))
        consumed.append(match.span())

    # Strategy 2: loose `CHANGE path ... SEARCH ... REPLACE ...` blocks.
    # We split the response on lines that start a new `CHANGE <path>` and
    # parse each chunk for a SEARCH/REPLACE pair. Skip ranges already
    # consumed by the canonical parser. The line may be inside a ``` fence
    # (Qwen2.5-7B does this), so we accept an optional leading ``` prefix.
    def _is_consumed(idx: int) -> bool:
        return any(s <= idx < e for s, e in consumed)

    # Accept both `CHANGE path` and `<<<CHANGE path` as chunk starters
    # (some models open with the canonical sigil but never close it).
    chunks = list(
        re.finditer(
            r"^\s*(?:```[a-zA-Z0-9_+-]*\s*)?(?:<<<\s*)?CHANGE\s+(\S+)\s*$",
            response, re.MULTILINE,
        )
    )
    for i, m in enumerate(chunks):
        if _is_consumed(m.start()):
            continue
        fpath = m.group(1).strip()
        body_start = m.end()
        body_end = chunks[i + 1].start() if i + 1 < len(chunks) else len(response)
        body = response[body_start:body_end]
        # Trim trailing closing-fence ``` if present
        body = re.sub(r"\n?```\s*$", "", body, flags=re.MULTILINE)
        parts = re.split(r"^\s*SEARCH\s*$", body, maxsplit=1, flags=re.MULTILINE)
        if len(parts) < 2:
            continue
        parts2 = re.split(r"^\s*REPLACE\s*$", parts[1], maxsplit=1, flags=re.MULTILINE)
        if len(parts2) < 2:
            continue
        out.append((
            fpath,
            _strip_fence(parts2[0]).strip("\n"),
            _strip_fence(parts2[1]).strip("\n"),
        ))

    return out


def make_diff(original: str, modified: str, file_path: str) -> str:
    """Compute unified diff; ensure trailing newline so `patch` is happy."""
    import difflib

    if original and not original.endswith("\n"):
        original += "\n"
    if modified and not modified.endswith("\n"):
        modified += "\n"
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def apply_change_blocks(
    oracle_files: dict[str, str],
    blocks: list[tuple[str, str, str]],
) -> dict[str, str]:
    """Apply SEARCH/REPLACE blocks. Returns modified file contents."""
    modified: dict[str, str] = {}
    for fpath, search_text, replace_text in blocks:
        # Resolve fpath against oracle keys (model may strip prefix)
        resolved = fpath
        if fpath not in oracle_files:
            for opath in oracle_files:
                if opath.endswith(fpath) or fpath.endswith(opath):
                    resolved = opath
                    break
        original = modified.get(resolved, oracle_files.get(resolved, ""))
        if not original:
            continue
        # Strict match first, then whitespace-tolerant fallback
        if search_text in original:
            modified[resolved] = original.replace(search_text, replace_text, 1)
            continue
        stripped_orig = "\n".join(l.rstrip() for l in original.split("\n"))
        stripped_search = "\n".join(l.rstrip() for l in search_text.split("\n"))
        if stripped_search in stripped_orig:
            modified[resolved] = original.replace(
                search_text.rstrip(), replace_text.rstrip(), 1
            )
    return modified


def build_diff(
    oracle_files: dict[str, str],
    modified_files: dict[str, str],
) -> str:
    """Build a unified diff covering all modified files."""
    diffs: list[str] = []
    for fpath, modified in modified_files.items():
        original = oracle_files.get(fpath, "")
        if original == modified:
            continue
        diff = make_diff(original, modified, fpath)
        if diff:
            diffs.append(diff)
    return "\n".join(diffs)


def fetch_oracle_files(repo: str, base_commit: str, file_paths: list[str]) -> dict[str, str]:
    """Read oracle files via raw.githubusercontent.com (no clone needed)."""
    import urllib.request
    import urllib.error

    out: dict[str, str] = {}
    for fpath in file_paths:
        url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{fpath}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            lines = text.split("\n")
            if len(lines) > MAX_FILE_LINES:
                text = "\n".join(lines[:MAX_FILE_LINES]) + (
                    f"\n... (truncated, {len(lines)} lines total)"
                )
            out[fpath] = text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            log.warning("  oracle fetch failed for %s: %s", fpath, exc)
    return out


def make_client(base_url: str | None) -> OpenAI:
    """Build an OpenAI-compatible client.

    `base_url=None` -> OpenRouter (requires OPENROUTER_API_KEY).
    Anything else  -> a local vLLM endpoint, no real auth needed.
    """
    if base_url is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
            "SAGE_OPENROUTER_API_KEY"
        )
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY not set in environment or .env")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return OpenAI(api_key="EMPTY", base_url=base_url)


DEFAULT_MAX_TOKENS = 4000  # focused diffs are well under this; keeps cost low


def generate_one(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    extra: dict = {}
    # Qwen3 enables thinking mode by default which inserts <think>...</think>
    # blocks ahead of the actual reply. We're doing single-shot patch
    # generation, not multi-turn reasoning, so disable it via the vLLM
    # chat_template_kwargs extra body parameter.
    if model.lower().startswith("qwen/qwen3"):
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        timeout=LLM_TIMEOUT_S,
        **extra,
    )
    return resp.choices[0].message.content or ""


def parse_raw_diff(response: str) -> str:
    """Fallback: extract a raw unified diff from the response.

    Models that don't follow the CHANGE-block instruction often emit either
    ```diff blocks or bare `--- a/... +++ b/...` headers. We try both.
    """
    fenced = re.search(r"```(?:diff|patch)?\s*\n([\s\S]*?)```", response, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
        if "+++ b/" in text or "--- a/" in text:
            if not text.endswith("\n"):
                text += "\n"
            return text
    # Bare diff: take from first '--- a/' to end-of-text
    bare = re.search(r"(--- a/[\s\S]+?)(?:\n```|\Z)", response)
    if bare:
        text = bare.group(1).rstrip()
        if not text.endswith("\n"):
            text += "\n"
        return text
    return ""


def parse_full_file_blocks(response: str) -> dict[str, str]:
    """Fallback: parse <<<FILE path ... FILE>>> blocks (full-file rewrites)."""
    blocks: dict[str, str] = {}
    pattern = re.compile(r"<<<FILE\s+(.+?)\s*\n([\s\S]*?)FILE>>>", re.MULTILINE)
    for match in pattern.finditer(response):
        fpath = match.group(1).strip()
        content = match.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        blocks[fpath] = content
    return blocks


# ---------------------------------------------------------------------------
# Phase 1: Generation
# ---------------------------------------------------------------------------

def sample_instances(seed: int, n: int) -> list[dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = indices[:n]
    return [dict(ds[i]) for i in sorted(chosen)]


def make_prompt(instance: dict, oracle_files: dict[str, str]) -> str:
    file_contents = "\n\n".join(
        f"### {fpath}\n```python\n{content}\n```"
        for fpath, content in oracle_files.items()
    ) if oracle_files else "(no files available)"
    hints = instance.get("hints_text", "")
    hints_section = f"## Hints\n{hints}" if hints else ""
    return PROMPT_TEMPLATE.format(
        repo=instance["repo"],
        problem_statement=instance["problem_statement"],
        hints_section=hints_section,
        file_contents=file_contents,
    )


def generate_for_generator(
    generator_key: str,
    generator_model: str,
    base_url: str | None,
    instances: list[dict],
    n_patches: int,
    temperature: float,
    base_seed: int,
    output_dir: Path,
    max_workers: int,
) -> Path:
    """Generate all patches for one generator and write predictions JSONL.

    Returns path to predictions JSONL (one record per (instance, patch_id)).
    """
    out_dir = output_dir / generator_key
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    raw_records_path = out_dir / "generation_records.jsonl"

    # Resume: skip records already present
    completed: set[tuple[str, int]] = set()
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                # model_name_or_path encodes the patch_id suffix
                m = re.match(r".+__p(\d+)$", rec["model_name_or_path"])
                if m:
                    completed.add((rec["instance_id"], int(m.group(1))))
            except (json.JSONDecodeError, KeyError):
                continue

    client = make_client(base_url)

    # Prepare oracle files once per instance (network fetch is expensive)
    log.info("[%s] fetching oracle files for %d instances", generator_key, len(instances))
    oracle_per_instance: dict[str, dict[str, str]] = {}
    for inst in instances:
        gold_files = get_changed_files_from_patch(inst.get("patch", ""))
        oracle_per_instance[inst["instance_id"]] = fetch_oracle_files(
            inst["repo"], inst["base_commit"], gold_files
        )

    # Build the work list
    tasks: list[tuple[dict, int, int]] = []
    for inst in instances:
        for pid in range(n_patches):
            if (inst["instance_id"], pid) in completed:
                continue
            tasks.append((inst, pid, base_seed + pid * 10_000))

    log.info(
        "[%s] %d (instance, patch_id) pairs to generate (%d already done)",
        generator_key, len(tasks), len(completed),
    )

    # Open files in append mode; use lock if needed
    import threading
    write_lock = threading.Lock()

    raw_responses_dir = out_dir / "raw_responses"
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    def _do_one(task: tuple[dict, int, int]) -> GenerationRecord:
        inst, pid, seed = task
        oracle_files = oracle_per_instance[inst["instance_id"]]
        prompt = make_prompt(inst, oracle_files)
        try:
            response = generate_one(client, generator_model, prompt, temperature, seed)
            diff = ""
            extraction_path = "none"
            # 1) preferred path: SEARCH/REPLACE blocks
            blocks = parse_change_blocks(response)
            if blocks:
                modified = apply_change_blocks(oracle_files, blocks)
                diff = build_diff(oracle_files, modified)
                if diff:
                    extraction_path = "change_blocks"
            # 2) fallback: full-file rewrites
            if not diff:
                full_blocks = parse_full_file_blocks(response)
                if full_blocks:
                    diffs: list[str] = []
                    for fpath, modified_content in full_blocks.items():
                        # resolve fpath against oracle keys
                        resolved = fpath if fpath in oracle_files else next(
                            (o for o in oracle_files if o.endswith(fpath) or fpath.endswith(o)),
                            fpath,
                        )
                        original = oracle_files.get(resolved, "")
                        if original:
                            d = make_diff(original, modified_content, resolved)
                            if d:
                                diffs.append(d)
                    if diffs:
                        diff = "\n".join(diffs)
                        extraction_path = "full_file_blocks"
            # 3) fallback: raw unified diff in the response
            if not diff:
                raw = parse_raw_diff(response)
                if raw:
                    diff = raw
                    extraction_path = "raw_diff"

            err = "" if diff else (
                f"empty diff (response chars={len(response)}, "
                f"change_blocks={len(blocks)})"
            )

            # Persist raw response for debugging when extraction fails
            if not diff:
                rp = raw_responses_dir / f"{inst['instance_id']}_p{pid}.txt"
                rp.write_text(response)

            return GenerationRecord(
                generator_key=generator_key,
                generator_model=generator_model,
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                patch_id=pid,
                diff=diff,
                response_chars=len(response),
                error=err if not diff else f"ok (via {extraction_path})",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:  # broad on purpose: we want to log every failure
            return GenerationRecord(
                generator_key=generator_key,
                generator_model=generator_model,
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                patch_id=pid,
                diff="",
                response_chars=0,
                error=f"exception: {type(exc).__name__}: {exc}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_do_one, t) for t in tasks]):
            rec = fut.result()
            log.info(
                "[%s] %s p%d -> diff=%d chars %s",
                generator_key, rec.instance_id, rec.patch_id, len(rec.diff),
                f"({rec.error})" if rec.error else "",
            )
            with write_lock:
                with open(raw_records_path, "a") as f:
                    f.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                # SWE-bench harness predictions JSONL
                pred = {
                    "instance_id": rec.instance_id,
                    # encode patch_id in model name so harness treats them as distinct
                    "model_name_or_path": f"{generator_key}__p{rec.patch_id}",
                    "model_patch": rec.diff,
                }
                with open(predictions_path, "a") as f:
                    f.write(json.dumps(pred) + "\n")

    return predictions_path


# ---------------------------------------------------------------------------
# Phase 2: Evaluation via SWE-bench Docker harness
# ---------------------------------------------------------------------------

def run_swebench_eval(
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    work_dir: Path,
) -> Path:
    """Invoke `python -m swebench.harness.run_evaluation` and return path to report."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Lite",
        "--predictions_path", str(predictions_path),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--cache_level", "env",
    ]
    log.info("eval: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=work_dir, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.warning("eval exit=%d", proc.returncode)
        log.warning("stderr tail:\n%s", proc.stderr[-2000:])
    log.info("eval stdout tail:\n%s", proc.stdout[-2000:])
    # Harness writes a report file named `<model>.<run_id>.json` in work_dir.
    candidates = sorted(work_dir.glob(f"*.{run_id}.json"))
    if not candidates:
        raise RuntimeError(
            f"No SWE-bench report file matching *.{run_id}.json in {work_dir}"
        )
    return candidates[-1]


def load_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text())


def parse_resolved(report: dict) -> set[str]:
    """Extract the set of resolved (model_patch passed) instance_ids."""
    raw = report.get("resolved_instances", report.get("resolved_ids", []))
    if isinstance(raw, dict):
        return set(raw.keys())
    return set(raw)


# ---------------------------------------------------------------------------
# Phase 3: Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    generator_key: str,
    generator_model: str,
    instances: list[dict],
    predictions_path: Path,
    eval_report_path: Path,
    n_patches: int,
) -> GeneratorSummary:
    report = load_report(eval_report_path)
    resolved = parse_resolved(report)
    log.info("[%s] resolved set size=%d", generator_key, len(resolved))

    # Build per-(instance, patch_id) outcome
    by_repo_total: dict[str, int] = {}
    by_repo_pass: dict[str, int] = {}
    n_attempted = 0
    n_nonempty = 0
    n_correct = 0
    per_instance_pass_rate: list[float] = []

    instance_repo = {inst["instance_id"]: inst["repo"] for inst in instances}

    # Reload predictions to know how many were nonempty
    nonempty_keys: set[tuple[str, int]] = set()
    for line in predictions_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r".+__p(\d+)$", rec["model_name_or_path"])
        if not m:
            continue
        pid = int(m.group(1))
        if rec.get("model_patch"):
            nonempty_keys.add((rec["instance_id"], pid))

    for inst in instances:
        inst_id = inst["instance_id"]
        repo = inst["repo"]
        passes = 0
        for pid in range(n_patches):
            n_attempted += 1
            by_repo_total[repo] = by_repo_total.get(repo, 0) + 1
            # Resolved keys in the report are the instance_id strings (harness
            # de-duplicates on (instance_id, model_name_or_path))
            # SWE-bench's report aggregates per instance_id under the model name.
            # Since we used distinct model names per patch_id, each report
            # corresponds to ONE patch_id. We loaded the per-pid report below.
            # So check resolved set for inst_id presence.
            ok = (inst_id in resolved) and ((inst_id, pid) in nonempty_keys)
            if (inst_id, pid) in nonempty_keys:
                n_nonempty += 1
            if ok:
                n_correct += 1
                passes += 1
                by_repo_pass[repo] = by_repo_pass.get(repo, 0) + 1
        per_instance_pass_rate.append(passes / n_patches if n_patches else 0.0)

    by_repo = {
        repo: {
            "total": by_repo_total[repo],
            "passed": by_repo_pass.get(repo, 0),
            "rate": by_repo_pass.get(repo, 0) / by_repo_total[repo],
        }
        for repo in sorted(by_repo_total)
    }

    return GeneratorSummary(
        generator_key=generator_key,
        generator_model=generator_model,
        n_instances=len(instances),
        n_patches_attempted=n_attempted,
        n_patches_nonempty=n_nonempty,
        n_patches_evaluated=n_attempted,  # all attempted go through harness
        n_correct=n_correct,
        base_rate=n_correct / n_attempted if n_attempted else 0.0,
        base_rate_per_instance=(
            sum(per_instance_pass_rate) / len(per_instance_pass_rate)
            if per_instance_pass_rate else 0.0
        ),
        by_repo=by_repo,
    )


def aggregate_per_pid(
    generator_key: str,
    generator_model: str,
    instances: list[dict],
    predictions_path: Path,
    work_dir: Path,
    n_patches: int,
) -> GeneratorSummary:
    """Look up one report per patch_id and aggregate.

    The harness was invoked once per patch_id with run_id `{generator_key}_p{pid}`.
    """
    by_repo_total: dict[str, int] = {}
    by_repo_pass: dict[str, int] = {}
    n_attempted = 0
    n_nonempty = 0
    n_correct = 0
    per_instance_pass_rate: list[float] = []

    # Map (instance_id, pid) -> nonempty
    nonempty_keys: set[tuple[str, int]] = set()
    for line in predictions_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r".+__p(\d+)$", rec["model_name_or_path"])
        if not m:
            continue
        pid = int(m.group(1))
        if rec.get("model_patch"):
            nonempty_keys.add((rec["instance_id"], pid))

    # Load resolved sets per pid
    resolved_per_pid: dict[int, set[str]] = {}
    for pid in range(n_patches):
        run_id = f"{generator_key}_p{pid}"
        candidates = sorted(work_dir.glob(f"*.{run_id}.json"))
        if not candidates:
            log.warning("[%s] no report for pid=%d", generator_key, pid)
            resolved_per_pid[pid] = set()
            continue
        report = load_report(candidates[-1])
        resolved_per_pid[pid] = parse_resolved(report)
        log.info(
            "[%s] pid=%d resolved=%d/%d",
            generator_key, pid, len(resolved_per_pid[pid]), len(instances),
        )

    for inst in instances:
        inst_id = inst["instance_id"]
        repo = inst["repo"]
        passes = 0
        for pid in range(n_patches):
            n_attempted += 1
            by_repo_total[repo] = by_repo_total.get(repo, 0) + 1
            if (inst_id, pid) in nonempty_keys:
                n_nonempty += 1
            ok = inst_id in resolved_per_pid.get(pid, set())
            if ok:
                n_correct += 1
                passes += 1
                by_repo_pass[repo] = by_repo_pass.get(repo, 0) + 1
        per_instance_pass_rate.append(passes / n_patches if n_patches else 0.0)

    by_repo = {
        repo: {
            "total": by_repo_total[repo],
            "passed": by_repo_pass.get(repo, 0),
            "rate": by_repo_pass.get(repo, 0) / by_repo_total[repo],
        }
        for repo in sorted(by_repo_total)
    }

    return GeneratorSummary(
        generator_key=generator_key,
        generator_model=generator_model,
        n_instances=len(instances),
        n_patches_attempted=n_attempted,
        n_patches_nonempty=n_nonempty,
        n_patches_evaluated=n_attempted,
        n_correct=n_correct,
        base_rate=n_correct / n_attempted if n_attempted else 0.0,
        base_rate_per_instance=(
            sum(per_instance_pass_rate) / len(per_instance_pass_rate)
            if per_instance_pass_rate else 0.0
        ),
        by_repo=by_repo,
    )


# ---------------------------------------------------------------------------
# Per-pid prediction file split (harness expects one row per (instance, model))
# ---------------------------------------------------------------------------

def split_predictions_by_pid(
    predictions_path: Path, n_patches: int
) -> dict[int, Path]:
    """Split a multi-pid predictions file into per-pid files.

    The harness identifies predictions by (instance_id, model_name_or_path).
    Since we encoded patch_id in the model name as `{key}__p{pid}`, splitting
    the JSONL into one file per pid lets us use distinct run_ids and report
    files (`<model>.{run_id}.json`).
    """
    out: dict[int, Path] = {}
    buckets: dict[int, list[str]] = {pid: [] for pid in range(n_patches)}
    for line in predictions_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r".+__p(\d+)$", rec.get("model_name_or_path", ""))
        if not m:
            continue
        pid = int(m.group(1))
        if pid in buckets:
            buckets[pid].append(line)
    for pid, lines in buckets.items():
        if not lines:
            continue
        path = predictions_path.with_name(f"predictions_p{pid}.jsonl")
        path.write_text("\n".join(lines) + "\n")
        out[pid] = path
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument("--n-patches", type=int, default=DEFAULT_N_PATCHES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-workers-gen", type=int, default=DEFAULT_MAX_WORKERS_GEN)
    parser.add_argument("--max-workers-eval", type=int, default=DEFAULT_MAX_WORKERS_EVAL)
    parser.add_argument("--generators", default=",".join(GENERATORS),
                        help="Comma-separated generator keys to run")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = [k.strip() for k in args.generators.split(",") if k.strip()]
    selected: dict[str, tuple[str, str | None]] = {}
    for k in selected_keys:
        if k not in GENERATORS:
            raise SystemExit(f"unknown generator key: {k}")
        selected[k] = GENERATORS[k]

    log.info("output dir: %s", output_dir)
    log.info("generators: %s", {k: f"{m} via {u or 'OpenRouter'}"
                                 for k, (m, u) in selected.items()})
    log.info("n_instances=%d n_patches=%d seed=%d", args.n_instances, args.n_patches, args.seed)

    instances = sample_instances(args.seed, args.n_instances)
    log.info(
        "sampled %d instances; repos=%s",
        len(instances),
        sorted({i["repo"] for i in instances}),
    )
    # Persist the sample so it's reproducible
    (output_dir / "sample.json").write_text(
        json.dumps(
            [{"instance_id": i["instance_id"], "repo": i["repo"]} for i in instances],
            indent=2,
        )
    )

    # ---------- Phase 1 ----------
    if not args.skip_generate:
        for key, (model, base_url) in selected.items():
            log.info("=== generating: %s (%s via %s) ===",
                     key, model, base_url or "OpenRouter")
            generate_for_generator(
                generator_key=key,
                generator_model=model,
                base_url=base_url,
                instances=instances,
                n_patches=args.n_patches,
                temperature=args.temperature,
                base_seed=args.seed,
                output_dir=output_dir,
                max_workers=args.max_workers_gen,
            )

    # ---------- Phase 2 ----------
    if not args.skip_eval:
        eval_dir = output_dir / "eval"
        for key in selected:
            predictions_path = output_dir / key / "predictions.jsonl"
            if not predictions_path.exists():
                log.warning("[%s] no predictions file at %s; skipping eval", key, predictions_path)
                continue
            per_pid_files = split_predictions_by_pid(predictions_path, args.n_patches)
            for pid, path in per_pid_files.items():
                run_id = f"{key}_p{pid}"
                report_glob = list(eval_dir.glob(f"*.{run_id}.json"))
                if report_glob:
                    log.info("[%s] pid=%d already evaluated, skipping", key, pid)
                    continue
                log.info("=== eval: %s pid=%d ===", key, pid)
                run_swebench_eval(
                    predictions_path=path,
                    run_id=run_id,
                    max_workers=args.max_workers_eval,
                    work_dir=eval_dir,
                )

    # ---------- Phase 3 ----------
    summaries: list[GeneratorSummary] = []
    eval_dir = output_dir / "eval"
    for key, (model, _base_url) in selected.items():
        predictions_path = output_dir / key / "predictions.jsonl"
        if not predictions_path.exists() or not eval_dir.exists():
            log.warning("[%s] missing data; skipping summary", key)
            continue
        summary = aggregate_per_pid(
            generator_key=key,
            generator_model=model,
            instances=instances,
            predictions_path=predictions_path,
            work_dir=eval_dir,
            n_patches=args.n_patches,
        )
        summaries.append(summary)

    summary_payload = {
        "n_instances": args.n_instances,
        "n_patches": args.n_patches,
        "temperature": args.temperature,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generators": [dataclasses.asdict(s) for s in summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    # Pretty print
    print()
    print("=" * 78)
    print("SPOT-CHECK SUMMARY")
    print("=" * 78)
    print(f"sample: n_instances={args.n_instances} n_patches={args.n_patches} seed={args.seed}")
    print(f"{'generator':<14} {'model':<32} {'attempted':>9} {'nonempty':>8} {'pass':>5} "
          f"{'rate':>6} {'inst-rate':>10}")
    for s in summaries:
        print(
            f"{s.generator_key:<14} {s.generator_model:<32} "
            f"{s.n_patches_attempted:>9} {s.n_patches_nonempty:>8} "
            f"{s.n_correct:>5} {s.base_rate:>6.2%} {s.base_rate_per_instance:>10.2%}"
        )
    print("=" * 78)
    print("regime check (PRE_REGISTRATION.md S4):")
    for s in summaries:
        in_window = 0.30 <= s.base_rate <= 0.70
        verdict = "IN  [0.30,0.70]" if in_window else "OUT [0.30,0.70]"
        print(f"  {s.generator_key:<14} base_rate={s.base_rate:.3f}  -> {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
