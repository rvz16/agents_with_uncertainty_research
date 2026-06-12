"""Run SWE-bench harness on the iterative refinement predictions.

Wraps spot_check_generators.run_swebench_eval (which includes the podman
compat shim) and runs it for steps 1..N-1 of one generator under a chosen
predictions directory.

Originally hard-coded to a single Linux/podman setup and to the
`data/spot_check_n50/` tree; now parameterised so the same script works on:

  * Mac / Docker Desktop (no podman) — for local SWE-Bench reruns
  * Linux cluster with podman 3.4.4 — opt in via --podman or env

Output naming convention (unchanged): per-step report.json under
<work-dir>/<run-id>/, where run-id = "<gen>_iter_step{k}".

Usage:
  # Mac, Docker, lite split, Self-Refine outputs
  python iter/eval_steps.py \\
      --gen gpt5_mini \\
      --data-dir data/swebench_lite_realbaselines_selfrefine_rerun \\
      --dataset princeton-nlp/SWE-bench_Lite

  # Verified split, Reflexion outputs
  python iter/eval_steps.py \\
      --gen haiku45 \\
      --data-dir data/swebench_verified_realbaselines_reflexion_rerun \\
      --dataset princeton-nlp/SWE-bench_Verified

  # Linux cluster with podman
  python iter/eval_steps.py --gen qwen3_coder \\
      --data-dir data/swebench_lite_realbaselines_selfrefine \\
      --podman

Resume-safe: existing report.json files for a (gen, step) pair are
overwritten on rerun, but already-passed instances are skipped by the
harness itself.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This file lives at iter/eval_steps.py. parents[1] = orchestration_hypothesis_testing/
ROOT = Path(__file__).resolve().parents[1]
# Both paths needed: ROOT for `_common.*` imports, ROOT/scripts for
# `spot_check_generators` (mirrors refine_swe.py / harness.py setup).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _maybe_setup_podman_shim() -> None:
    """Set up Podman compat for Linux clusters. No-op on Mac (Docker Desktop)."""
    # Only attempt the podman shim when:
    #   * --podman CLI flag was given (sets SWEBENCH_PODMAN_COMPAT below), OR
    #   * the calling shell already exported SWEBENCH_PODMAN_COMPAT=1, OR
    #   * DOCKER_HOST is already pointing at a podman socket
    docker_host = os.environ.get("DOCKER_HOST", "")
    podman_flag = os.environ.get("SWEBENCH_PODMAN_COMPAT", "")
    if not (podman_flag or "podman" in docker_host.lower()):
        return
    # Heuristic for the standard rootless podman socket path. Skip the
    # setdefault if the user already gave us an explicit DOCKER_HOST.
    if not docker_host:
        os.environ["DOCKER_HOST"] = (
            f"unix:///run/user/{os.geteuid()}/podman/podman.sock"
        )
    os.environ.setdefault("SWEBENCH_PODMAN_COMPAT", "1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen", required=True,
                   help="Generator slug, e.g. gpt5_mini / haiku45 / sonnet45 / "
                        "qwen3_coder / qwen25_coder_32b / gpt_oss_20b. Must "
                        "match the subdirectory name under --data-dir.")
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing <gen>/predictions_iter_step{1..N}"
                        ".jsonl. Typically the --output-dir from a refine_swe."
                        "py run, e.g. data/swebench_lite_realbaselines_"
                        "selfrefine_rerun.")
    p.add_argument("--n-steps", type=int, default=5,
                   help="Iterate step in 1..N-1 (matches refine_swe.py default).")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   choices=["princeton-nlp/SWE-bench_Lite",
                            "princeton-nlp/SWE-bench_Verified"],
                   help="HuggingFace dataset id forwarded to the harness. Must "
                        "match the dataset used by the refine run.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Where the harness writes its report files. "
                        "Default: <data-dir>/eval")
    p.add_argument("--max-workers", type=int,
                   default=int(os.environ.get("SWE_EVAL_MAX_WORKERS", "1")),
                   help="Concurrent harness workers. Default 1 to avoid "
                        "containerd metadata race (see iter/harness.py). "
                        "Override via SWE_EVAL_MAX_WORKERS env or flag.")
    p.add_argument("--podman", action="store_true",
                   help="Use podman-compat shim instead of regular Docker. "
                        "Required on Linux clusters where DOCKER_HOST points at "
                        "a podman.sock. No-op on Mac / Docker Desktop.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.podman:
        os.environ["SWEBENCH_PODMAN_COMPAT"] = "1"
    _maybe_setup_podman_shim()

    # Late import — has to come AFTER any podman env wiring because
    # spot_check_generators imports docker SDK at module level.
    import spot_check_generators as scg  # noqa: E402

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"ERROR: --data-dir does not exist: {data_dir}", file=sys.stderr)
        return 1
    work_dir = (args.work_dir or (data_dir / "eval")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"eval config: gen={args.gen}  data-dir={data_dir}  "
          f"n-steps={args.n_steps}  max-workers={args.max_workers}  "
          f"dataset={args.dataset}  work-dir={work_dir}")

    n_ok = n_skip = n_err = 0
    for step in range(1, args.n_steps):
        pred_path = data_dir / args.gen / f"predictions_iter_step{step}.jsonl"
        if not pred_path.exists():
            print(f"skip step {step}: {pred_path} not found")
            n_skip += 1
            continue
        run_id = f"{args.gen}_iter_step{step}"
        print(f"\n==== eval {args.gen} step {step} (run_id={run_id}) ====")
        try:
            report_path = scg.run_swebench_eval(
                predictions_path=pred_path,
                run_id=run_id,
                max_workers=args.max_workers,
                work_dir=work_dir,
                dataset_name=args.dataset,
            )
            print(f"  -> {report_path.name}")
            n_ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            n_err += 1

    print(f"\nDone: ok={n_ok}  skipped={n_skip}  errored={n_err}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
