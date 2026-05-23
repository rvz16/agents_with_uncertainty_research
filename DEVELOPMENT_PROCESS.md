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

---

## 2026-05-23 — Shared online + post-hoc transition-kernel module

Adopted PR #5's first follow-up request ("Online Kernel Calibration ported
to a shared library"). Unified five copies of the post-hoc Beta(1,1)-smoothed
kernel computation plus the live Beta-Binomial estimator into one module.

- **New: `experiments/orchestration_hypothesis_testing/_common/kernel.py`.**
  Exports `DEFAULT_KERNEL`, `kernel_update`, `compute_transition_kernel_from_pairs`,
  `pairs_from_trajectories`, `OnlineKernelCalibration`, `resolve_kernel`. The
  online estimator is a thread-safe Beta-Binomial running posterior with
  per-regime fallback to the seed kernel. `resolve_kernel(gen_dir, mode)`
  is a three-way switch: `measured` (load file or default), `online` (same
  + return an estimator), `hardcoded` (always literature default).

- **New: `tests/test_kernel.py`** — 24 unit tests covering kernel_update
  with both lowercase/uppercase schemas, post-hoc Laplace smoothing, custom
  Beta priors, online posterior convergence to ground truth (2000 samples),
  thread-safety under 8 concurrent workers, per-regime fallback, and
  resolve_kernel mode dispatch. All passing.

- **Migrated callers** (4 of 5 — see exclusion below):
  - `scripts/run_synthesis_live.py`: deleted the inline `DEFAULT_KERNEL`,
    `kernel_update`, `load_kernel`, `_KernelCounts`, `OnlineKernelCalibration`
    (~75 lines); now imports from `_common.kernel`. The 8-line load + mode
    dispatch block collapses to `kernel, src, _ = resolve_kernel(...)`.
  - `iter/refine.py:compute_kernel`: now a thin wrapper over the shared
    helper. Preserves iter's legacy contract (returns `None` for an empty
    regime rather than Laplace-uniform 0.5) for back-compat.
  - `iter/kernel.py`: standalone CLI now uses the shared helper; output
    schema (with `P_stay_*` fields under a `kernel_all` wrapper) unchanged.
  - `analysis/compute_transition_kernel.py:_kernel_from_pairs`: shared
    helper + legacy alias keys (`n_broken_transitions`, etc.) kept so
    existing `transition_kernel.json` jq queries don't break.
  - `scripts/synthesis_transition_kernel.py`: shared helper + literature-
    prior fallback (0.5 / 0.05) for empty regimes overrides the helper's
    Laplace-uniform default.

- **Deliberately NOT migrated: `iter/swe_kernel.py:compute_kernel`.** Its
  `transition_kernel.json` output is a flat unsmoothed schema (no
  `kernel_all` wrapper, raw counts as `fix_count` etc.), consumed by the
  SWE sections of `analysis.ipynb` with that exact shape. Migrating it
  would be a downstream-visible JSON shape change and is scoped out. Note
  is in `_common/kernel.py`'s module docstring.

- **New `iter/refine.py` flag: `--kernel-mode {measured,online,hardcoded}`**
  (default `measured`, no behavior change). When set to `online` or
  `hardcoded`, the post-iter block also streams this run's (Y_t, Y_{t+1})
  transitions through an `OnlineKernelCalibration` (seeded from the
  src-dir calibration kernel for `online`, or `DEFAULT_KERNEL` for
  `hardcoded`) and writes `transition_kernel_online_final.json` with the
  Beta-Binomial posterior summary. This file is informational only — the
  existing post-hoc `transition_kernel.json` is still produced, and
  `compute_policy_comparison` continues to use the static kernel (a
  methodology change for a future PR).

### Verification

- `python -m pytest tests/ -q` — **94 passed** (60 pre-existing + 34
  kernel tests), no warnings beyond the existing urllib3 dependency warn.
- Round-trip test on each migrated caller confirms output schemas match
  pre-migration behavior bit-for-bit on hand-built fixtures.
- All 4 migrated CLIs launch via `python -m <pkg>.<mod> --help` without
  import errors.
- The `--kernel-mode online` end-to-end path is exercised by a synthesized
  iter_records.jsonl fixture; produces the expected `n_broken_observed`,
  `k_fix`, etc. counts and `current_estimate`.

### Post-review fixups (from self-review of PR #6)

- `OnlineKernelCalibration.update()` now validates that y_before and
  y_after are both in {0, 1}; previously, `update(None, 1)` silently
  recorded a (correct → correct) transition because anything `!= 0`
  fell through to the "else" branch. Both current callers
  (iter/refine.py online block, scripts/run_synthesis_live.py per-variant
  branches) already filter non-binary externally, so this is a tightening
  of the contract, not a behavior break for production callers.
- `OnlineKernelCalibration.__init__` now validates that init_kernel
  carries the required keys at construction time, rather than letting a
  malformed dict silently propagate and KeyError later inside `.get()`
  on the first empty-regime fallback.
- `scripts/run_synthesis_live.py` migration comment no longer claims
  the abbo bridge is wired (it isn't — explicitly scoped out, see below).
- `_common/kernel.py` drops a string-quoted forward ref now that
  `from __future__ import annotations` is in scope.
- Strengthened `test_online_kernel_thread_safe` to assert the per-regime
  counts (`n_broken`, `k_fix`, `n_correct`, `k_break`) rather than just
  the total, so a race that miscategorizes transitions can no longer
  pass silently.
- New tests: malformed kernel.json, missing required keys, init_kernel
  validation, parameterized non-binary update rejection, asymmetric Beta
  prior on a populated regime (pins the "P_fix + P_stay_broken ≠ 1 with
  alpha ≠ beta" property the docstring claims).

### Out of scope (deliberate non-changes)

- `iter/swe_kernel.py` keeps its private compute_kernel (different schema).
- `compute_policy_comparison` semantics unchanged — it still consumes the
  static post-hoc kernel. Wiring it to the online posterior is a separate
  methodology discussion.
- abbo's `bayesian_optimization_for_code_testing/.../run_codecontests_full.py`
  still imports the never-created `abbo.realworld.agents.kernel_helpers` +
  `abbo.realworld.telemetry` modules; deferred since abbo is being retired.

