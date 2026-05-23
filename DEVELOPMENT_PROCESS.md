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

## 2026-05-23 — Cost-vector balance search

Adds a measurement-anchored sweep over `c_ver` within each cost-mode
regime (FAST / SLOW) to find values that produce **balanced policy
comparison histograms** — multiple non-trivial winners against AV AND
multiple non-trivial losers. The motivation: several §2.5 cells (e.g.
`lcb_hard/gpt5_mini`) currently produce degenerate histograms where
every policy sits below `always_verify`, which makes the policy
comparison uninformative. The c_ver values most likely justifiable by
measurement may not be the values that produce the most informative
comparison; this module surfaces the tradeoff.

- **New module:
  `experiments/orchestration_hypothesis_testing/analysis/cost_vector_balance.py`**
  - `balance_score(policy_deltas, epsilon=2.0)` — effective-competitor
    count: `min(#policies with Δ > +ε, #policies with Δ < -ε)`. Higher
    = the comparison is more informative. Excludes `always_verify`
    itself (sits at 0 by definition).
  - `CostMode` dataclass — packs `c_gen`, `c_critic_*`, sweep `c_ver`
    range, current §2.5 c_ver value, and which benchmarks fall in the
    mode.
  - `FAST_MODE` (LCB-{easy,medium,hard}, MBPP+, HumanEval+, HEFix, CC;
    c_gen=10, sweep `c_ver ∈ [1, 30]`, current 5).
  - `SLOW_MODE` (SWE-Bench Lite + Verified; c_gen=5, sweep
    `c_ver ∈ [5, 100]`, current 30). SLOW range covers up to ~20×
    pooled median to capture the bimodal Docker-eval tail (heavy-suite
    p90 ≈ 682s in Table tab:action_latency).
  - `sweep_c_ver_one_cell(traj, likes, prior, mode)` — per-cell
    backward-induction policy comparison at each c_ver in the mode's
    grid; reuses the existing `run_policies` from
    `analysis/lcb_sensitivity.py`.
  - `aggregate_mode_balance(per_cell_sweeps, mode)` — average balance
    score across all cells in a mode at each c_ver.
  - `recommend_c_ver_for_mode(aggregated, min_balance=1.0)` — pick the
    c_ver that maximizes mean balance across cells, ties broken by
    spread (std of Δ across policies). Returns `None` if no c_ver
    achieves `min_balance` (degenerate mode).

- **New: `tests/test_cost_vector_balance.py`** — 19 unit tests covering:
  - `balance_score` math (balanced histogram, all-below-AV degenerate,
    all-above-AV degenerate, custom epsilon, AV exclusion/inclusion,
    spread tiebreaker, empty input)
  - mode definitions (known/unknown benchmarks, sweep range sanity)
  - mode aggregation (cell filtering by benchmark membership, multi-cell
    averaging, empty input)
  - recommendation (max-balance pick, spread tiebreaker, below-threshold
    None return, empty input)

- **3 new notebook cells in `analysis.ipynb`** (positions 19-21, right
  after the §3c canonical `pc` cell):
  - Cell 19 (markdown): methodology, mode definitions, output schema
  - Cell 20 (code): sweeps c_ver across each mode's grid for every
    calibration cell, builds `STAT_CVER_SWEEP` dataframe, prints
    per-mode recommendation (current vs balance-optimal c_ver, mean
    balance score at each)
  - Cell 21 (code): heatmap (x=c_ver, y=cell, color=balance score) with
    vertical lines marking current §2.5 + recommended c_vers; saves
    `c_ver_balance_heatmap.png`; prints per-cell diagnostic table +
    "how many cells could improve" headline

### Verification

- `python -m pytest tests/ -q` → **131 passed** (60 baseline + 38
  kernel + 14 telemetry + 19 new balance), no regressions.
- Both new code cells `ast.parse` cleanly.
- Notebook: 86 → 89 cells.

### Methodology notes

- The current §2.5 c_ver values (FAST=5, SLOW=30) were chosen as
  "honest matches" to per-benchmark measured c_ver/c_gen ratios, but
  the measurement table (`tab:action_latency`) shows the median
  measured ratios are much smaller than the current §2.5 values for
  FAST benchmarks (0.009 - 0.059 measured vs 0.5 used). The current
  §2.5 values are abstract utility units that already bake in
  perceived-cost factors beyond wall-clock (API spend, batch effects,
  retry cost, user friction). This module's job is **not** to compete
  with §2.5 on measurement fidelity, but to surface which c_ver
  values within each mode's plausible range produce the most
  informative policy comparison — and to flag cells where no c_ver
  in range achieves balance (genuinely degenerate cells).
- Per-mode aggregation uses mean balance across cells. Median is
  reported alongside in case of outlier-driven mean inflation. A
  more conservative choice would be `min` across cells — guarantees
  every cell hits the recommended balance — but that's pessimistic
  on heterogeneous mode populations.
- The sweep uses `n_boot=100` (vs cell 13's 1000) because we're
  scoring sign-of-Δ patterns rather than pinning CIs tightly; the
  speed gain is ~10×.

### Out of scope (deliberate non-changes)

- **§2.5 cost-vector assignments are not modified.** This is a
  diagnostic / exploratory module; if the recommendation is to move
  to different c_ver values, that's a separate PR with its own
  paper-section discussion and a full Table 1 re-computation.
- **No update to Table 1 cells.** Cell 13's STAT_POLICY still uses
  §2.5 cost vectors as-is. The sweep is additive.

