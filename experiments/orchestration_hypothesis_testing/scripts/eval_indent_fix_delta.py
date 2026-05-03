"""Run the SWE-bench harness on only the (instance, pid) pairs that
flipped from empty -> non-empty after the indent-tolerant fix.

For each generator with flips, we group by patch_id, write a delta
predictions JSONL, and invoke the harness with run_id
`{key}_p{pid}_indentfix`. This produces report files alongside the
existing `{key}_p{pid}` reports; aggregation merges the two.

This avoids re-running harness on the ~150 already-evaluated pairs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from spot_check_generators import run_swebench_eval  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("delta-eval")


def load_predictions(pred_path: pathlib.Path) -> list[dict]:
    out = []
    for line in pred_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/spot_check")
    ap.add_argument("--report-path", default=None,
                    help="reprocess_report.json path (default: <data-dir>/reprocess_report.json)")
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir).resolve()
    rep_path = pathlib.Path(args.report_path) if args.report_path else (
        data_dir / "reprocess_report.json"
    )
    if not rep_path.exists():
        raise SystemExit(f"reprocess report not found: {rep_path}")
    rep = json.loads(rep_path.read_text())

    eval_dir = data_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if "podman" in (os.environ.get("DOCKER_HOST", "").lower()):
        log.info("DOCKER_HOST=%s -> harness will use podman + compat shim",
                 os.environ["DOCKER_HOST"])
    else:
        log.warning(
            "DOCKER_HOST does not look like a podman socket (%r). "
            "Make sure DOCKER_HOST and SWEBENCH_PODMAN_COMPAT are set if needed.",
            os.environ.get("DOCKER_HOST", ""),
        )

    n_evals_run = 0
    for gen_rep in rep.get("generators", []):
        if gen_rep.get("skipped"):
            continue
        gen_key = gen_rep["generator"]
        flipped = gen_rep.get("flipped", [])
        if not flipped:
            log.info("[%s] no flipped pairs; skipping", gen_key)
            continue

        gen_dir = data_dir / gen_key
        full_pred_path = gen_dir / "predictions.jsonl"
        if not full_pred_path.exists():
            log.warning("[%s] no predictions.jsonl; skipping", gen_key)
            continue
        all_preds = load_predictions(full_pred_path)

        # group flipped by pid
        by_pid: dict[int, set[str]] = {}
        for f in flipped:
            by_pid.setdefault(f["patch_id"], set()).add(f["instance_id"])

        for pid, inst_set in sorted(by_pid.items()):
            tag = f"{gen_key}_p{pid}_indentfix"
            existing = list(eval_dir.glob(f"*.{tag}.json"))
            if existing:
                log.info("[%s] pid=%d already has indent-fix report (%s); skipping",
                         gen_key, pid, existing[-1].name)
                continue
            # Build delta predictions: only the flipped instances for this pid
            delta = []
            for rec in all_preds:
                if rec.get("instance_id") not in inst_set:
                    continue
                mname = rec.get("model_name_or_path", "")
                if not mname.endswith(f"__p{pid}"):
                    continue
                if not rec.get("model_patch"):
                    continue
                delta.append(rec)
            if not delta:
                log.info("[%s] pid=%d no non-empty delta predictions; skipping",
                         gen_key, pid)
                continue
            delta_path = gen_dir / f"predictions_p{pid}_indentfix.jsonl"
            with open(delta_path, "w") as f:
                for r in delta:
                    f.write(json.dumps(r) + "\n")
            log.info("[%s] pid=%d running harness on %d delta predictions (run_id=%s)",
                     gen_key, pid, len(delta), tag)
            try:
                report_path = run_swebench_eval(
                    predictions_path=delta_path,
                    run_id=tag,
                    max_workers=args.max_workers,
                    work_dir=eval_dir,
                )
                log.info("[%s] pid=%d delta report -> %s",
                         gen_key, pid, report_path.name)
                n_evals_run += 1
            except Exception as exc:
                log.warning("[%s] pid=%d harness failed: %s", gen_key, pid, exc)

    log.info("ran %d delta evaluations", n_evals_run)


if __name__ == "__main__":
    main()
