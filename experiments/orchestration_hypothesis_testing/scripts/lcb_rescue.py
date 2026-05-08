"""Rescue critic_results.jsonl from raw_responses/*.txt artifacts.

Used when lcb_calibrate.py crashes mid-run (it only persists results at
the END of each generator's loop — see line 429). All raw model responses
are on disk under <gen>/raw_responses/<inst_id>_p<pid>.txt; this script
recomputes L0/L1/L2 and Y deterministically from them. L3 (Haiku review)
requires re-calling the API and is OPT-IN via --rerun-l3.

Usage:
  # cheap: recover Y, L0, L1, L2 (L3 set to null)
  python3 lcb_rescue.py --output-dir data/lcb_calibration_v2 --generator gpt5_mini

  # full: also re-run Haiku on each saved patch (~$0.30 for 174 patches)
  python3 lcb_rescue.py --output-dir data/lcb_calibration_v2 --generator gpt5_mini --rerun-l3

Output: <gen>/critic_results_rescued.jsonl (does NOT overwrite the
calibrator's eventual critic_results.jsonl, so it's safe to run while
the calibration is still active on a different generator).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lcb_calibrate import (  # noqa: E402
    MAX_PRIVATE_TESTS,
    check_tests,
    critic_L0_syntax,
    critic_L1_lint,
    critic_L3_review,
    decode_private_tests,
    extract_code,
    load_lcb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lcb_rescue")


_FILENAME_RE = re.compile(r"^(?P<inst>.+)_p(?P<pid>\d+)\.txt$")


def parse_raw_filename(name: str) -> tuple[str, int] | None:
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("inst"), int(m.group("pid"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--rerun-l3", action="store_true",
                        help="Re-call Haiku to recompute L3 (costs ~$0.0017/patch)")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--platform", default="leetcode")
    args = parser.parse_args()

    gen_dir = (args.output_dir / args.generator).resolve()
    raw_dir = gen_dir / "raw_responses"
    if not raw_dir.exists():
        log.error("no raw_responses dir at %s", raw_dir)
        sys.exit(1)

    # Index raw files by instance_id
    raw_files: dict[str, list[tuple[int, Path]]] = {}
    for p in raw_dir.glob("*.txt"):
        parsed = parse_raw_filename(p.name)
        if parsed is None:
            continue
        inst, pid = parsed
        raw_files.setdefault(inst, []).append((pid, p))
    log.info("found %d instances, %d raw responses",
             len(raw_files), sum(len(v) for v in raw_files.values()))

    # Load LCB problems for tests / starter_code (need a fresh map by question_id)
    problems = load_lcb(difficulty=args.difficulty, platform=args.platform)
    by_qid = {str(p["question_id"]): p for p in problems}

    # Optional client for L3 re-run
    client = None
    if args.rerun_l3:
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            log.error("--rerun-l3 needs OPENROUTER_API_KEY in env")
            sys.exit(1)
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    records: list[dict] = []
    skipped: list[tuple[str, str]] = []
    total_l3_cost = 0.0
    for inst_id, files in sorted(raw_files.items()):
        problem = by_qid.get(str(inst_id))
        if problem is None:
            skipped.append((inst_id, "no matching LCB problem"))
            continue
        starter = problem.get("starter_code", "") or ""
        public_tests = problem.get("public_test_cases") or []
        if isinstance(public_tests, str):
            try:
                public_tests = json.loads(public_tests)
            except Exception:
                public_tests = []
        private_tests = decode_private_tests(problem.get("private_test_cases", "") or "")

        for pid, path in sorted(files):
            text = path.read_text()
            code = extract_code(text)
            l0 = critic_L0_syntax(code)
            l1 = critic_L1_lint(code)
            l2_pass, l2_total = check_tests(code, public_tests, starter_code=starter)
            l2_ok = (l2_pass == l2_total) and l2_total > 0
            y_pass, y_total = check_tests(code, private_tests[:MAX_PRIVATE_TESTS], starter_code=starter)
            Y = 1 if (y_pass == y_total) and y_total > 0 else 0
            l3 = None
            if args.rerun_l3 and client is not None:
                l3_ok, l3_cost = critic_L3_review(
                    problem.get("question_content", "")[:3000], code, client,
                )
                l3 = bool(l3_ok)
                total_l3_cost += l3_cost
            records.append({
                "generator": args.generator,
                "instance_id": inst_id,
                "patch_id": pid,
                "Y": Y,
                "L0_syntax": bool(l0),
                "L1_lint": bool(l1),
                "L2_public_tests": bool(l2_ok),
                "L3_llm_review": l3,
                "diff_chars": len(code),
                "y_pass_rate": (y_pass / max(y_total, 1)),
                "l2_pass_rate": (l2_pass / max(l2_total, 1)),
                "rescued": True,
            })
            if len(records) % 10 == 0:
                log.info("[%s] rescued %d records (L3 cost so far: $%.4f)",
                         args.generator, len(records), total_l3_cost)

    out_path = gen_dir / "critic_results_rescued.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    log.info("wrote %d records to %s (L3 total cost: $%.4f, skipped: %d)",
             len(records), out_path, total_l3_cost, len(skipped))
    if skipped:
        for inst, reason in skipped[:5]:
            log.info("  skip %s: %s", inst, reason)


if __name__ == "__main__":
    main()
