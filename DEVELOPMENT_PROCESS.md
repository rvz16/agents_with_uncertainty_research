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

---

## 2026-05-23 — Shared per-action telemetry across calibration + iter

Adopted PR #5's second follow-up request ("Structured Telemetry promoted to
a shared library"). The `_ActionTelemetry` class that previously lived
inline in `calibration/lcb.py` (and was cross-imported by `mbpp.py` and
`humaneval.py`) moves into a proper module under `_common/`, picks up the
superset schema needed by iter / BDP-aware code paths, and is wired into
five scripts that were previously missing per-action timing.

- **New: `experiments/orchestration_hypothesis_testing/_common/telemetry.py`**
  - `TelemetryLogger(path, dataset, model_name, *, run_id=None)` —
    thread-safe append-only JSONL writer, positional signature matches
    the pre-existing `_ActionTelemetry` for back-compat
  - `.record(*, action_type, runtime_s, instance_id, patch_id=None,
    step=None, passed=None, api_cost_usd=0.0, belief_before=None,
    extra=None)` — superset of the previous record shape; new optional
    fields cover iter (`step`) and BDP-aware code paths (`belief_before`)
  - `write_action(logger, ...)` — convenience wrapper matching abbo's
    module-level `from abbo.realworld.telemetry import write_action`
    shape so code ported from abbo can switch import path only
  - `_ActionTelemetry = TelemetryLogger` — back-compat alias
  - `ACTION_TYPES` — canonical set of action-type strings downstream
    analyzers expect

- **New: `tests/test_telemetry.py`** — 12 unit tests covering
  construction, slim/rich record schemas, passed-field round-trip,
  thread-safety under 8 concurrent workers, close idempotency,
  context-manager usage, `write_action` equivalence with `record`,
  back-compat alias identity, parent-dir mkdir, ACTION_TYPES contents.
  All passing.

- **Migrated existing callers** (3):
  - `calibration/lcb.py`: deleted the inline `_ActionTelemetry` class,
    added `from _common.telemetry import TelemetryLogger` plus a
    `_ActionTelemetry = TelemetryLogger` re-export so any straggling
    `from calibration.lcb import _ActionTelemetry` keeps working
  - `calibration/mbpp.py`, `humaneval.py`: repointed their imports from
    `calibration.lcb._ActionTelemetry` to
    `_common.telemetry.TelemetryLogger as _ActionTelemetry`

- **Instrumented previously-missing scripts** (5):
  - `calibration/humanevalfix.py`: added `tele.record(...)` at 6 call
    sites (generate / critic_L0 / critic_L1 / critic_L2 / verify /
    critic_L3) mirroring `mbpp.py`'s template
  - `calibration/codecontests.py`: same 6-site pattern
  - `calibration/from_spotcheck.py`: 3 sites (critic_L0 / critic_L1 /
    critic_L3 only — this script reads pre-existing predictions, no
    generation or verify here)
  - `iter/refine.py`: added `tele` kwarg to `_run_lcb_one_instance` and
    `_run_generic_one_instance`. Records per-step `reflect` (SR critique
    + Rfx reflection share this label), `refine`, optional `critic_L3`,
    and a fused `verify` row for the L0+L1+L2+Y block. Closes
    the latency-analysis gap (calibration had step-0 timing; iter now
    has refinement-step timing). Per-critic granularity inside
    `_eval_patch` is a follow-up.
  - `iter/refine_swe.py`: same pattern; fused `critic_L0` row carries
    `L1` in `extra` (the two are computed in lockstep on diff'd files);
    separate `critic_L3` row for the LLM judge.

- **JSONL row schema** (all 8 callers produce this superset; optional
  fields omitted when not supplied so calibration JSONLs stay slim):
  ```
  ts, dataset, model_name, instance_id, action_type, runtime_seconds,
  passed, api_cost_usd,
  [patch_id], [step], [belief_before], [run_id], [extra]
  ```

### Verification

- `python -m pytest tests/ -q` — **72 passed** (60 pre-existing + 12 new
  telemetry tests), no regressions.
- All 8 entry-point CLIs (`calibration.{lcb,mbpp,humaneval,humanevalfix,
  codecontests,from_spotcheck}`, `iter.{refine,refine_swe}`) launch via
  `python -m <pkg>.<mod> --help` without import errors.
- End-to-end functional smoke: write 2 rows via the back-compat alias
  path, read back, confirm schema includes `ts` + correct field names.

### Out of scope (deliberate non-changes)

- Per-critic granularity inside `_eval_patch` (the iter generic-variant
  helper) — its L0/L1/L2/Y are fused in a single block; surfacing each
  individually is a separate restructuring.
- `scripts/run_synthesis_live.py` was already using `TelemetryLogger`
  via the imports added in PR #5 — kept as is.
- W&B upload of `action_telemetry.jsonl` files — `upload_runs.py` already
  globs `*.jsonl` files alongside `cost_log.jsonl`, so the new files are
  picked up automatically. No code change needed; verify on the next
  W&B sync.

### Post-review fixups (from self-review of PR #7)

- `iter/refine.py:main()` now wraps the per-generator work block in a
  `try / finally` with `tele.close()` in the finally clause. The previous
  code reached `tele.close()` only via the happy path — a SWE-not-
  implemented `continue`, a worker-thread exception that propagated, or
  any raise in `compute_kernel` / `compute_policy_comparison` /
  `write_combined_iter_policy` would leak the file handle for the rest
  of the run. This brings the file into line with `iter/refine_swe.py`
  and all 5 calibration scripts.
- `_common/telemetry.py:record()` now uses `if extra is not None` rather
  than `if extra:` for the `extra` field. The truthiness check silently
  dropped an explicit empty dict, which was inconsistent with the other
  optional fields (`patch_id`, `step`, `belief_before`, `run_id` — all of
  which use `is not None`). New tests pin both directions: `extra={}` is
  now written as `"extra": {}`, and `extra=None` (default) still omits
  the field for slim calibration JSONLs.
- All 5 LCB `tele.record(...)` sites in `iter/refine.py` now carry
  `extra={"variant": "lcb", ...}` to match the generic-variant and SWE
  records. Downstream analyses that want to query "refine-step latency
  by variant" can now do so via a uniform `extra.variant` field across
  all three benchmark families.

`pytest tests/ -q` → **74 passed** (60 pre-existing + 14 telemetry tests
after the +2 new extra-semantics tests).

---

## 2026-05-23 — Online-BDP analysis cell on iter trajectories

Adds a counter-factual column to the analysis notebook: **"what if the BDP
planner had been learning the transition kernel online as it walked each
iter trajectory, rather than using a fixed measured kernel?"**

This was scoped intentionally to iter data only — calibration data has at
most one Y observation per instance (simulate_policy ends at first verify),
so no within-trajectory transitions exist for an online estimator to learn
from. Iter trajectories observe Y at every step (the iter script always
backfills via the oracle), so each step→step+1 boundary provides a real
`(Y_t, Y_{t+1})` transition pair.

- **New module:
  `experiments/orchestration_hypothesis_testing/analysis/online_dp_iter.py`**
  - `simulate_online_bdp_on_iter(iter_traj, likes, prior, kernel_seed, cost)`
    — per-instance replay. Walks the trajectory step-by-step, after each
    `generate` action absorbs the observed `(Y_t, Y_{t+1})` pair into an
    `OnlineKernelCalibration` seeded from the calibration kernel, then
    re-decides at the next step using the updated posterior.
  - `_online_bdp_decide(belief, k_left, kernel, cost)` — small backward
    induction over only `{verify, give_up, generate}`. Iter-replay-specific:
    critic outcomes are pre-observed (cost is sunk in `step_cost_usd`),
    so the action space collapses to just the verify-or-continue decision.
    Faster + cleaner than rebuilding the full `BayesianController` per step.
  - `_bayes_update_belief(belief, critic_likes, rec)` — Bayes-updates
    belief on all critic outcomes available in a step record. Matches
    `BayesianController._bayes_update` arithmetic, so iter-replay belief is
    directly comparable to static-BDP belief at the same step.
  - Per-instance reset (single-task picture, no cross-instance transfer).
    Cross-instance accumulation is a different research question deserving
    its own paper section; not in this PR.

- **New: `tests/test_online_dp_iter.py`** — 21 unit tests covering:
  - Decision DP: high/low belief → verify/give_up; high-p_fix kernel →
    generate; uppercase + lowercase kernel keys; malformed kernel raises
  - Bayes-update: pass/fail directions; skips missing/None likes
  - Simulator: empty trajectory, high-prior step-0 verify, low-prior
    give-up, kernel updates equal generate count, kernel actually evolves
    away from seed, per-instance reset is independent, Y=None handled
    mid-trajectory and at verify, critics move belief, belief propagates
    via online (not seed) kernel between steps
  - All 21 tests pass.

- **New: 3 notebook cells in `experiments/orchestration/wandb/analysis.ipynb`**
  inserted at positions 16-18, right after the existing SR/Rfx iter
  analysis (`STAT_ITER`):
  - Cell 16 (markdown): methodology explanation — what online-BDP means
    here, why it's meaningful on iter data but not on calibration data,
    why per-instance reset.
  - Cell 17 (code): builds `STAT_ITER_ONLINE_DP` dataframe. For each
    (benchmark, generator, method) iter cell: downloads `iter_records.jsonl`
    from W&B, downloads `critic_results.jsonl` for likelihoods, seeds the
    online kernel from `KERN_MEAS` (same source as static-BDP), runs
    `simulate_online_bdp_on_iter` per instance, paired-bootstrap delta vs
    `always_verify`. Schema parallels `STAT_ITER`.
  - Cell 18 (code): side-by-side bar plot of static-BDP vs online-BDP
    delta-vs-always-verify, per (benchmark, generator) cell. Saves
    `static_vs_online_bdp_eval.png` and prints average Δ summary.

### Verification

- `pytest tests/ -q` → **133 passed** (60 pre-existing + 38 kernel +
  14 telemetry + 21 new online-dp-iter).
- Both new code cells `ast.parse` cleanly.
- Notebook JSON-roundtrips cleanly (cell count 86 → 89).

### Out of scope (deliberate non-changes)

- Cross-instance accumulation variant of online-BDP. The per-instance
  reset is the cleaner single-task story; accumulation models a fleet-
  learning agent and would need its own paper section to motivate.
- Live BDP-online agent (vs replay). The live version exists in
  `scripts/run_synthesis_live.py` with `--kernel-mode online --variants
  dp_fitted` and produces its own W&B runs; the notebook surfaces those
  results from a different code path. This PR is replay-only.
- Wiring online-BDP into the existing visualizations (cells 29/35/46
  bar charts, cell 70 regime maps). The new analysis sits as a
  standalone supplementary section; touching the main visualizations
  would risk regressions in Table 1's existing static-BDP figures.

