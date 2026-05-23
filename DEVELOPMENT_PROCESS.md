# Development process

This file tracks source-code changes so reviewers and future runners can
reconstruct what changed and why. Append a dated entry for every source
edit. Keep entries terse: file, what, why.

---

## 2026-05-20 — Colleague-runnable smoke path for function-level + SWE-Bench

Goal: a teammate cloning the repo for the first time should be able to
run all five benchmarks (LCB-hard/medium/easy, MBPP+, HumanEval+,
SWE-Bench Lite/Verified) end-to-end without per-host source patches.
Found two off-cluster papercuts and fixed them.

- **New doc:**
  `experiments/orchestration_hypothesis_testing/COLLEAGUE_RUNBOOK.md` —
  step-by-step, OS-aware (API vs local-vLLM, Mac vs cluster), with smoke
  tests, gotchas, and the recommended order for filling the empty cells
  in `tab:full_results`. Sister to `PLAYBOOK.md` (which is about
  *extending* the matrix); this one is about *running* it.

- **`experiments/orchestration_hypothesis_testing/calibration/mbpp.py`,
  `humaneval_calibrate.py`:** two fixes that were blocking first-time
  off-cluster runs.
  1. Dropped `os.environ.setdefault("HF_HOME", "/mnt/data/users/vlad.smirnov/hf_cache")`
     (a cluster-only path). HF now falls back to the OS default
     `~/.cache/huggingface`; users can still override via env var.
     Off-cluster the old default raised `OSError: [Errno 30]
     Read-only file system: '/mnt'`.
  2. Added `.env` autoload at the top of `main()`, mirroring
     `lcb_calibrate.py`'s walk-up-the-tree chain. Without this, MBPP+ /
     HumanEval+ fail with `OPENROUTER_API_KEY not set` even when the
     repo root has a `.env`. LCB worked because it already loaded
     `.env`; MBPP+ / HumanEval+ did not.

- **No change to `spot_check_generators.py`.** It already carries the
  `--dataset` and `--language-filter` flags upstream; the runbook uses
  them as documented.

### Verification (function-level synthesis, n=5, on Mac)

| Benchmark | cost | prior_Y1 | notes |
|---|---|---|---|
| LCB-hard | $0.063 | 0.143 | matches paper ≈0.17 |
| LCB-medium | $0.044 | 0.571 | strong L2 signal |
| LCB-easy | $0.020 | 0.571 | strong L2 + L3 |
| MBPP+ | ~$0.05 | 0.571 | saturation regime as expected |
| HumanEval+ | ~$0.05 | 0.714 | highest prior, regime C |

Re-ran MBPP+ at n=3 with **no** `OPENROUTER_API_KEY` exported and **no**
`HF_HOME` set, to confirm both fixes work cold. Pass.

### Verification (SWE-Bench Phase 1+2, n=5, on MBZUAI-Artem-1)

x86_64 + podman 3.4.4 + swebench 4.1.0;
`DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock` and
`SWEBENCH_PODMAN_COMPAT=1`:

| Benchmark | sample | Phase 1 | Phase 2 (podman harness) |
|---|---|---|---|
| SWE-Bench Lite | django×4 + sympy | $0.026, 36 s, 5/5 non-empty | ~3 min, 5/5 resolved, 0 errors |
| SWE-Bench Verified | django×4 + sphinx | similar | ~4 min, 5/5 resolved, 0 errors |

Eval reports written under each `<output-dir>/eval/`; no stray
containers or images after run.

### Out of scope (deliberate non-changes)

- `lcb_calibrate.py`'s GENERATORS dict still lacks `Qwen/Qwen2.5-Coder-7B-Instruct`
  and `gpt-oss-20b`. Adding them is `PLAYBOOK.md`'s "adding a new
  generator" workflow, not a plumbing fix.
- `spot_check_generators.py` GENERATORS likewise.
- No `requirements.txt` edit. `evalplus` is needed for HumanEval+; it is
  flagged in the runbook §0.1.

---

(append future entries below this line, newest on top)
