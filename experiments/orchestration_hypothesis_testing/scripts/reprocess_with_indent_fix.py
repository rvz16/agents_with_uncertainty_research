"""Re-extract diffs from existing raw_responses using the fixed parser.

Walks each generator's `raw_responses/*.txt` (which only contains the
failed extractions), re-runs `parse_change_blocks` -> `apply_change_blocks`
-> `build_diff` with the new indentation-tolerant matcher, and updates
`predictions.jsonl` (and per-pid splits) in place for the pairs that
flipped from empty to non-empty.

Tracks the flipped (instance_id, patch_id) pairs per generator so the
caller can re-run the harness only on the deltas.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
from typing import Iterable

# Allow running from any cwd
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from spot_check_generators import (  # noqa: E402
    apply_change_blocks,
    build_diff,
    fetch_oracle_files,
    get_changed_files_from_patch,
    make_diff,
    parse_change_blocks,
    parse_full_file_blocks,
    parse_raw_diff,
    split_predictions_by_pid,
    strip_think_blocks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("reprocess")


def _re_extract(response: str, oracle_files: dict[str, str]) -> tuple[str, str]:
    """Run the same three-tier extraction as generate_for_generator.

    Returns (diff, extraction_path).
    """
    response = strip_think_blocks(response)
    blocks = parse_change_blocks(response)
    if blocks:
        modified = apply_change_blocks(oracle_files, blocks)
        diff = build_diff(oracle_files, modified)
        if diff:
            return diff, "change_blocks"

    full_blocks = parse_full_file_blocks(response)
    if full_blocks:
        diffs: list[str] = []
        for fpath, modified_content in full_blocks.items():
            resolved = (
                fpath if fpath in oracle_files
                else next(
                    (o for o in oracle_files
                     if o.endswith(fpath) or fpath.endswith(o)),
                    fpath,
                )
            )
            original = oracle_files.get(resolved, "")
            if original:
                d = make_diff(original, modified_content, resolved)
                if d:
                    diffs.append(d)
        if diffs:
            return "\n".join(diffs), "full_file_blocks"

    raw = parse_raw_diff(response)
    if raw:
        return raw, "raw_diff"

    return "", "none"


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def reprocess_generator(
    generator_dir: pathlib.Path,
    instances_by_id: dict[str, dict],
    n_patches: int,
) -> dict:
    """Re-extract diffs for one generator dir. Returns a per-generator report."""
    gen_key = generator_dir.name
    raw_dir = generator_dir / "raw_responses"
    pred_path = generator_dir / "predictions.jsonl"
    if not raw_dir.exists() or not pred_path.exists():
        return {"generator": gen_key, "skipped": True, "flipped": []}

    preds = _load_jsonl(pred_path)
    pred_index: dict[tuple[str, int], int] = {}
    for idx, rec in enumerate(preds):
        m = re.match(r".+__p(\d+)$", rec.get("model_name_or_path", ""))
        if m:
            pred_index[(rec["instance_id"], int(m.group(1)))] = idx

    oracle_cache: dict[str, dict[str, str]] = {}
    flipped: list[tuple[str, int]] = []
    n_attempted = 0
    n_no_blocks = 0
    n_blocks_no_match = 0

    files = sorted(raw_dir.glob("*.txt"))
    for fp in files:
        m = re.match(r"(.+)_p(\d+)\.txt", fp.name)
        if not m:
            continue
        inst_id, pid = m.group(1), int(m.group(2))
        n_attempted += 1
        key = (inst_id, pid)
        if key not in pred_index:
            continue
        # Only process pairs that were previously empty
        existing = preds[pred_index[key]].get("model_patch", "")
        if existing:
            continue

        if inst_id not in oracle_cache:
            inst = instances_by_id.get(inst_id)
            if inst is None:
                continue
            files_to_fetch = get_changed_files_from_patch(inst.get("patch", ""))
            try:
                oracle_cache[inst_id] = fetch_oracle_files(
                    inst["repo"], inst["base_commit"], files_to_fetch
                )
            except Exception as exc:
                log.warning("oracle fetch failed for %s: %s", inst_id, exc)
                continue
        oracle = oracle_cache[inst_id]

        response = fp.read_text()
        diff, path_used = _re_extract(response, oracle)
        if diff:
            preds[pred_index[key]]["model_patch"] = diff
            flipped.append(key)
            log.info(
                "[%s] %s p%d FLIPPED -> %d chars (via %s)",
                gen_key, inst_id, pid, len(diff), path_used,
            )
        else:
            # diagnose
            blocks = parse_change_blocks(strip_think_blocks(response))
            if not blocks:
                n_no_blocks += 1
            else:
                n_blocks_no_match += 1

    # Write back if any flips
    if flipped:
        _write_jsonl(pred_path, preds)
        # Refresh per-pid splits
        split_paths = split_predictions_by_pid(pred_path, n_patches)
        log.info(
            "[%s] wrote updated predictions.jsonl + %d per-pid splits",
            gen_key, len(split_paths),
        )

    # Update generation_records.jsonl too (for transparency)
    rec_path = generator_dir / "generation_records.jsonl"
    if rec_path.exists() and flipped:
        recs = _load_jsonl(rec_path)
        flipped_set = set(flipped)
        # iterate by appearance order; keep last entry per (inst, pid)
        # mark records whose (inst, pid) flipped
        for rec in recs:
            key = (rec.get("instance_id", ""), rec.get("patch_id", -1))
            if key in flipped_set:
                rec["error"] = "ok (via change_blocks; reprocessed with indent fix)"
                # backfill the new diff length so downstream tools agree
                # (we don't rewrite the diff text into the record because
                # generation_records is append-only; predictions.jsonl is the
                # source of truth for the harness).
        _write_jsonl(rec_path, recs)

    return {
        "generator": gen_key,
        "skipped": False,
        "n_failed_responses_seen": n_attempted,
        "n_flipped": len(flipped),
        "n_no_blocks": n_no_blocks,
        "n_blocks_still_no_match": n_blocks_no_match,
        "flipped": [{"instance_id": i, "patch_id": p} for i, p in flipped],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data/spot_check",
        help="Spot-check data root (contains <generator>/ subdirs)",
    )
    parser.add_argument("--n-patches", type=int, default=3)
    parser.add_argument(
        "--generators",
        default="qwen25_7b,qwen3_8b,qwen3_8b_thinking",
        help="Comma-separated list of generator subdirs to process",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional path to write the JSON report (default: <data-dir>/reprocess_report.json)",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    from datasets import load_dataset
    log.info("loading SWE-bench Lite dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instances_by_id = {d["instance_id"]: dict(d) for d in ds}

    reports = []
    for key in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = data_dir / key
        log.info("=== reprocessing %s ===", key)
        rep = reprocess_generator(gen_dir, instances_by_id, args.n_patches)
        reports.append(rep)
        log.info(
            "[%s] flipped=%d no_blocks=%d blocks_still_no_match=%d (of %d failed responses)",
            key,
            rep.get("n_flipped", 0),
            rep.get("n_no_blocks", 0),
            rep.get("n_blocks_still_no_match", 0),
            rep.get("n_failed_responses_seen", 0),
        )

    out_path = pathlib.Path(args.report_out) if args.report_out else (
        data_dir / "reprocess_report.json"
    )
    out_path.write_text(json.dumps({"generators": reports}, indent=2))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
