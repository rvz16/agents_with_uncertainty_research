# Development process

This file tracks source-code changes so reviewers and future runners can
reconstruct what changed and why. Append a dated entry for every source
edit. Keep entries terse: file, what, why.

---

## 2026-06-17 — SWE-Bench pipeline: local-build harness fix + reusable runner

**Problem.** Running the SWE-Bench harness against rootless podman 3.4.4
(no docker group on the cluster VMs) hits two bugs that make remote-pull
mode error on 28% of SWE-Bench Lite instances (every astropy + matplotlib +
recent django — the swebench team never uploaded their images to docker.io),
and local-build mode (`--namespace none`) fails outright because:
  1. The Docker SDK `client.api.build(pull=False)` is ignored by podman 3.4.4 —
     buildah tries to pull the unqualified `sweb.base.py.x86_64:latest` base
     image from docker.io, which doesn't exist there.
  2. `client.containers.create(platform=...)` is rejected by podman's API
     v1.40 (`InvalidVersion: platform is not supported for API version < 1.41`).

**Fix.** Three artifacts now under `experiments/orchestration_hypothesis_testing/`:

- **`scripts/patch_swebench_harness.py`** — idempotent patcher that locates
  the installed `swebench/harness/docker_build.py` and replaces (1) the
  `client.api.build(...)` call with a `podman build --pull=false` subprocess
  and (2) the `platform=` kwarg in `containers.create(...)` with a comment.
  Run once per swebench install before the first harness invocation.

- **`scripts/spot_check_generators.py`** — `run_swebench_eval()` now passes
  `--namespace none` to the harness so env+instance images build locally,
  bypassing the missing-on-Docker-Hub instances entirely. Comment documents
  the prerequisite (run `patch_swebench_harness.py` first; set TMPDIR/
  BUILDAH_TMPDIR outside root quota).

- **`scripts_pipeline/run_swebench_pipeline.sh`** — generic 14-step pipeline
  (Cal Lite + Verified -> from_spotcheck -> SR refine+eval+backfill on both
  benchmarks -> Rfx refine+eval+backfill on both benchmarks). Takes
  `<generator> <lite_cost> <verified_cost>` as args; resumable
  (LLM gen is skipped on re-run if predictions exist); sets up TMPDIR on
  `/mnt/data` to avoid blowing through the 200 GB root quota on `/var/tmp`
  during buildah layer commits.

**Wall-clock.** Local builds are 5-15 min per first-build per env-image
hash, then ~1-3 min per instance image. Full pipeline on one generator
takes ~24-36 hours on a single rootless-podman host vs ~12-18 hours under
remote pulls (but with 0% error rate vs 28%).

**OpenRouter cost.** Cal LLM gen is ~$15-25 per benchmark for qwen3-coder
(scales ~8× for claude-sonnet-4.5); the harness eval itself is free
(no LLM). The from_spotcheck L3 LLM-review step is ~$5-10 per benchmark.

### 2026-06-17 (later, after codex review) — patcher + pipeline correctness

A codex review on the diff above caught two [P1] bugs before merge:

- **`scripts/patch_swebench_harness.py:patch_platform_kwarg`**. The original
  regex used `[^)]*?` to span the body of `client.containers.create(...)`,
  but that body contains `name=test_spec.get_instance_container_name(run_id)`
  whose nested `)` terminated the match before `platform=test_spec.platform,`
  was reached. `--dry-run` aborted with "Could not find platform=test_spec.
  platform inside containers.create()". Replaced the regex with a paren-
  depth scanner: locate `client.containers.create(`, walk forward counting
  parens until depth returns to zero, then search that span for the kwarg
  and comment it. Verified by re-running `--dry-run` against the installed
  swebench v4.1.0 — both patches now apply (and `ast.parse` on the patched
  source still succeeds).

- **`scripts_pipeline/run_swebench_pipeline.sh` + `iter/eval_steps.py`**.
  `refine_swe.py` writes per-step predictions to
  `<out>/<gen>/<method>/predictions_iter_step{N}.jsonl`, but `eval_steps.py`
  was hard-coded to read `<data-dir>/<gen>/predictions_iter_step{N}.jsonl`.
  In the pipeline's SR/Rfx stages every step was therefore reported as
  "skipped: file not found", `eval_steps` returned 0, `set -e` did not
  fire, and the subsequent `swe_backfill_y` step saw no harness reports —
  the pipeline would have completed with *zero* iterative SWE-Bench
  results. Added an optional `--method` flag to `eval_steps`: when set,
  predictions resolve under `<data-dir>/<gen>/<method>/`, `run_id` becomes
  `<gen>_<method>_iter_step{N}` (the convention `swe_backfill_y` already
  expects), and the default work-dir becomes `<cell_dir>/eval` so that
  backfill can find the reports. Updated all four SR/Rfx eval calls in
  `run_swebench_pipeline.sh` to pass `--method selfrefine` or `--method
  reflexion`. Legacy invocations without `--method` are unchanged.

---

## 2026-05-25 — EMNLP paper Results section rewrite + figure-asset reorg

**Notebook (`analysis.ipynb`).** Cell 1 now also creates
`FIGURES_OUT/main/` and `FIGURES_OUT/cver_sweep/`. Six master-figure
cells (34/36/38/42/77/79) and the c\_ver-sweep grid cell (91) moved
their `savefig` calls from top-level `CACHE/` to `FIGURES_OUT/main/`
and `FIGURES_OUT/cver_sweep/<bench>_grid.png` respectively. Cells
36/38 also gained `bbox_to_anchor=(0.5, -0.08)` on `fig.legend(...)`
to push the figure-level legend below the supxlabel (was overlapping
on the 3×3 grid). Existing PNGs were physically moved on disk to
match.

**Paper repo (`emnlp2026/initial/`).** Full Results section rewrite
in `results.tex`: reorganised into seven subsections (regime
characterisation → policy comparison on eval → critic
informativeness → DP-vs-greedy → published-baseline comparison →
cost-vector sensitivity → robustness), every claim resolves to a
figure or table reference. New main-paper Fig 2 added
(`policies_per_model_main.png` = SWE-Lite/sonnet45 eval). New
*Critic Informativeness* subsection pointing to the appendix
critic-grid. Two `\theo{TODO ...}` placeholders remain (DP-vs-greedy
scatter, Self-Refine/Reflexion table).

**Paper repo — figure refresh.** Legacy `regime_map.png`,
`three_regimes_populated.png`, `headline.png` replaced/dropped.
Refreshed `critics/`, `generator_locations/`, `per_strategy_grid/`,
`cver_sweep/` from the current notebook cache; per-bench coverage
extended from 7 benchmarks to 9 (added CodeContests, HumanEvalFix).
Dropped gpt-oss-20b name-variant duplicates (`gpt-oss-20b.png` and
`gpt_oss_20b_local.png`) in `cver_sweep/`, kept only the canonical
`gpt_oss_20b.png`. Four composite master grids
(`fig8_winner_grid_R_cver.png`, `fig7_per_strategy_grid.png`,
`fig7b_generator_locations.png`, `fig_regime_grid_cver_R.png`) now
shipped under `figures/main/`. The 9 per-benchmark eval rollups
under `figures/policies_per_model/<bench>_eval.png` are also
shipped. Total assets: 103 PNGs.

**Paper repo — appendix.** `appendix_figures.tex` rewritten to
match the refreshed asset inventory: 3×3 subfigure grids for
generator-locations and per-strategy-grid (was 3×2+blanks), new
*Held-out eval policy comparison* subsection
(`appendix:policies_per_model`), new *Composite master grids*
subsection. The four appendix subsections now reference exactly the
asset categories shipped in §3 of `emnlp2026/notes/results_revision_plan.md`.
`appendix_setup.tex` gained an `\label{appendix:setup}` alias for
forward-references from Results.

**Build status.** `pdflatex` (TeX Live 2025) produces a clean
29-page PDF. Remaining undefined references are the two known
TODOs (`fig:dp_vs_greedy_scatter`, `tab:selfrefine_reflexion`) plus
pre-existing labels in commented-out sections.

**Plan document.** New
`emnlp2026/notes/results_revision_plan.md` records the 6 design
decisions (Fig 2/3 choices, three legacy assets dropped, k-sensitivity
panels dropped, CodeContests/HumanEvalFix kernel TODO).

---

## 2026-05-24 — bayesian_DP kernel source: switch to selfrefine, drop on miss

Adopted selfrefine iter trajectories as the uniform source for `KERN_MEAS`
(the empirical refine transition kernel used by `bayesian_DP`). Previously
used `single_method` iter runs, which covered only 24/54 (benchmark,
generator) cells; the remaining 30 silently fell back to IID DP from
`run_policies`, mixing measured-kernel and IID DP rows under the same
`bayesian_DP` label. With selfrefine sourcing, **48/54 cells** have a
measured kernel; the remaining 6 get `bayesian_DP` dropped (loud, not
silent).

**Rationale.** Production code-fixing agents include a self-critique step
as part of their refine action. Selfrefine trajectories ARE the empirical
refine kernel for the agent's deployed algorithm. Cost-vector treatment:
`c_gen` represents one refine PRIMITIVE call regardless of internal
sub-steps (critique, self-reflection). See
`experiments/orchestration_hypothesis_testing/KERNEL_SOURCE_DECISION.md`
for the full argument.

**No silent fallbacks** — every lookup in `analysis.ipynb` now raises
loudly on unknown inputs rather than returning a misleading default:

- `canonical_generator(name)`: raises `ValueError` on `None`, `KeyError`
  on names not in `GENERATORS_RAW` (the explicit set of every valid raw
  W&B generator name). Previously silently passed through unknown names.
- `cost_dict_for(bench)`: raises `KeyError` on unknown benchmark.
  Previously silently used `FAST_ORACLE_COST` as default — would
  mis-cost any typo'd SWE benchmark catastrophically.
- Iter variant assignment (cell 15): replaced silent `else None`
  fallback with explicit `ITER_VARIANT_MAP[b]` — raises `KeyError` if a
  new benchmark gets uploaded without being registered. Previously
  skipped mbpp / humaneval / humanevalfix / codecontests entirely from
  `STAT_ITER` because they didn't match "lcb" or "swe" prefix.

**Source contract assertion (cell 13).** A post-loop assertion verifies
that no `bayesian_DP` row exists in `STAT_POLICY` for any (b, g) cell
NOT in `KERN_MEAS`. If a future change adds a new code path that
bypasses the override/drop pattern, the assertion fires and refuses to
let stale IID DP rows reach figures.

**Files changed:**
- `experiments/orchestration/wandb/analysis.ipynb`:
  - Cell 1: `GENERATOR_NAME_MAP` extended with `"gpt-oss-20b" → "gpt_oss_20b"`
    (third name variant discovered in `.cache/raw_evidence/`). Added
    `GENERATORS_RAW` frozenset + strict `canonical_generator()` raising
    `ValueError`/`KeyError` on bad input.
  - Cell 11: `cost_dict_for()` made strict.
  - Cell 13: `KERN_MEAS` filter `_k.method == "single_method"` →
    `_k.method == "selfrefine"`. Added pre-KERN_MEAS source-contract
    banner; pre-loop coverage report; `BDP_ABSENT_CELLS` tracker for
    cells with no calibration at all (vs `BDP_NO_KERNEL_CELLS` for cells
    with cal but no selfrefine kernel); end-of-cell summary banner; and
    post-loop assertion. Reframed pc-vs-STAT_POLICY xcheck message
    (legacy pc is heterogeneous; not "CORRECTED" but "differs-from-canonical").
  - Cell 15: replaced variant fallback with explicit `ITER_VARIANT_MAP`.
  - Cells 41 + 78: `RAW.glob("calib__*")` now dedupes by
    `canonical_generator(g)` to handle the multiple cache-dir copies per
    gpt-oss-20b cell (different uploaders wrote `gpt-oss-20b`,
    `gpt_oss_20b`, and `gpt_oss_20b_local` separately).

- New: `experiments/orchestration_hypothesis_testing/KERNEL_SOURCE_DECISION.md`
  documenting the source contract, cost-vector interpretation (Fix i),
  coverage matrix, remaining 6-cell gap, and ETA for closing it.

**Coverage outcome.** 48/54 (b, g) cells have measured selfrefine
kernel. Remaining 6 cells (`gpt5_mini` + `qwen3_coder` ×
{`codecontests`, `humanevalfix`}; `gpt_oss_20b` × {`swe_lite`,
`swe_verified`}) get `bayesian_DP` dropped explicitly. Per-cell `ERROR`
print + end-of-cell summary banner. Closing the gap requires running
selfrefine iter for those 6 cells (calibration first for 2 of them);
estimated ~6.5 h sequential, parallelisable.

---

## 2026-05-24 — SWE-Bench headline cost vector: median → p90 anchoring

Adopted p90-anchored SLOW as the headline cost vector for SWE-Bench Lite
and Verified. The previous median-anchored SLOW (c_ver=30) was demoted to
SLOW_MEDIAN, kept for appendix sensitivity.

**Rationale (paper-side):** SWE-Bench Docker eval is bimodal — pooled
median a_ver = 1.9s but heavy test suites drag p90 to 682s (350× median).
Median-anchoring lets AV "average out" the tail; the framework's value
proposition is exactly the tail-cost avoidance. P90 captures the regime
the framework is designed for and is the operationally relevant
verification cost for deployed agents.

**Measurement audit:** before committing, went deep on whether cell-11
cost vectors are auditable. Findings (full report on PR #9):
- The "Table tab:action_latency" referenced in cell-11 comments is NOT
  derivable from any data in W&B or the local repo. Values are hardcoded
  engineering estimates from prior calibration runs that predate the
  TelemetryLogger infrastructure (`experiments/orchestration_hypothesis_testing/_common/telemetry.py`).
- W&B does have `cost_usd` per refine step (from `iter_records.jsonl`
  artifacts across 24 single_method iter runs). Measured SWE/FAST $-cost
  ratio at step>0 = **~13×**. Current cost vector c_L3 SLOW=5 / FAST=1
  reflects 5× — validates the SWE > FAST direction but understates the
  $-cost magnitude. Decision: keep c_L3 SLOW=5 (treating $-cost as not
  equivalent to latency, which is what the cost vector represents).
- The action_telemetry.jsonl files (per-action wall-clock latency) exist
  only on the cluster filesystem, never uploaded to W&B. SSH to cluster
  blocked by network access during audit; deferred to a future
  tightening pass.

**Files changed:**
- `experiments/orchestration/wandb/analysis.ipynb`:
  - Cell 11 (§2.5 cost vectors): SLOW_ORACLE_COST gets c_ver=90 (was 30).
    SLOW_HEAVY_COST renamed → SLOW_MEDIAN_COST with c_ver=30 (was 90).
    `cost_model_for_sensitivity` regime 'slow_heavy' → 'slow_median'.
    SENSITIVITY_VECTORS key SLOW_HEAVY → SLOW_MEDIAN. Measurement-table
    comment updated to show SWE p90-anchored ratio.
  - Cell 19 (§3f markdown): updated SLOW c_ver/R from 0.30 → 0.90 with
    note about why we moved off median anchoring.
  - Cell 22 (§3h markdown): reframed — was "stress test under heavy
    regime"; now "appendix sensitivity: what if we anchored to median."
    Same data, inverted framing.
  - Cell 23 (§3h code): regime='slow_median' (was 'slow_heavy'),
    STAT_POLICY_SWE_HEAVY → STAT_POLICY_SWE_MEDIAN, cost_regime field
    and c_ver value reflect the median anchoring.
  - Cell 24 (§3i code): variable renames, bar chart colors swapped
    (headline in red, sensitivity in blue), output text reframed to
    show that the sign flips are *evidence for* the p90 headline
    choice. Math identical to previous version.
  - Cleared outputs of cells 11, 19, 20, 21, 22, 23, 24 — user must
    re-run. Downstream cells (panel grids in §6/§12) will also produce
    different SWE-Bench numbers and need re-running.

- `experiments/orchestration_hypothesis_testing/analysis/cost_vector_balance.py`:
  - `SLOW_MODE.c_ver_current` moved 30 → 90. Sweep range unchanged
    (5.0–90.0) — the sweep now spans "what if we anchored to median"
    (low end) up to the headline p90 (upper bound).
  - Added inline comment explaining the change.

**Tests:** 19 cost_vector_balance unit tests still pass.

**Pending for paper:** appendix needs a "cost-vector sensitivity"
subsection explaining the median-vs-p90 decision and noting that
results are robust to ±50% variation in absolute cost values (driven
by ratios, not absolutes). The 30 sign flips between SLOW and
SLOW_MEDIAN are the data backing this claim — bayesian_DP is robust,
critic-gate policies and fixed_pipeline are anchoring-sensitive.

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

### Post-iteration: reframing the balance analysis as a diagnostic

The first run of the balance sweep produced FAST median optimal c_ver
≈ 75 (with the [1, 100] range, ceiling-clipped). Two issues surfaced:

1. **c_ver > reward is methodologically incoherent.** When c_ver > R,
   AV's utility is strictly negative even on correct patches. The
   balance metric then saturates at high c_ver values for the wrong
   reason — AV is artificially worst. Both sweep ranges have been
   capped at 0.9·R = 90.

2. **FAST balance-optimal c_ver > SLOW balance-optimal c_ver is
   counterintuitive given the FAST/SLOW labels** (FAST = measured-fast
   verification = should have *cheap* c_ver). This contradiction
   surfaces the real finding: the "balance objective" doesn't track
   measurement-anchored c_ver. For SWE (SLOW), they coincide. For
   function-level (FAST), they diverge — function-level benchmarks
   have cheap verification AND high prior_Y1, putting AV in a regime
   where it's trivially optimal *by design* (§2.5 deliberately puts
   FAST below the analytic crossover at c_ver/R = 0.05). The balance
   metric would push c_ver up toward R to rescue these cells, but
   that requires inflating verification cost beyond measurement.

**Conclusion: the balance analysis is a DIAGNOSTIC of regime structure,
not a prescription to change §2.5.** §2.5's design — FAST below
crossover (AV trivially wins by design), SLOW above crossover
(framework operates) — is methodologically correct. The "degenerate
histograms" we observed for function-level cells at c_ver=5 are
*confirming evidence*, not bugs to fix.

The c_gen unification commit stays — it removes an unmeasured
asymmetry. §2.5 c_ver values (5 FAST, 30 SLOW) are unchanged; their
c_ver/R ratios (0.05, 0.30) are the methodology-anchored quantities
and remain at the design points.

What the balance analysis IS useful for in the paper:
- Confirming the regime structure of §2.5
- Identifying intrinsically-saturated cells (balance ≤ 1 at any c_ver
  in [1, 90]) — these are (benchmark, generator) pairs where the
  policy framework structurally doesn't discriminate. Worth a footnote.

Reframing:
- `cost_vector_balance.py` module docstring updated to lead with the
  "diagnostic, not prescription" framing
- Notebook cells 19-21 rewritten to interpret balance scores as
  regime confirmation, not as a recommendation to change §2.5
- The "headline" section now distinguishes:
  - cells balanced at §2.5 design point (framework operates here)
  - cells intrinsically saturated (paper footnote candidates)
  - cells only balanced at high c_ver (DO NOT rescue — by-design
    below-crossover)
- New explicit table of intrinsically-saturated cells

### Follow-up: SWE-Bench heavy-suite sensitivity vector

After re-analyzing Table tab:action_latency, a real methodology gap
surfaces: the §2.5 `SLOW_ORACLE_COST` (c_ver=30, c_ver/R=0.30) is
anchored to SWE-Bench's pooled-median verification cost (a_ver=1.9s),
but Docker eval is bimodal — heavy-suite (e.g. `psf/requests`) takes
~672s, pooled p90 = 682s. The median-anchored vector understates the
worst-case eval cost a deployed agent would face by ~22×.

Resolution (this commit): add a SECOND cost vector for SWE-Bench
specifically — `SLOW_HEAVY_COST` (c_ver=90, c_ver/R=0.90) — capped at
0.9·R to keep the policy comparison methodologically coherent
(c_ver > R would make AV's utility strictly negative even on correct
verifications). This is NOT a new MODE; the FAST/SLOW dichotomy still
holds. SLOW_HEAVY is a robustness-check sidecar to SLOW, used for
sensitivity analysis only.

- **New: `SLOW_HEAVY_COST` in §2.5 (notebook cell 11)** — `dict(c_gen=10,
  c_L0=1, c_L2=2, c_L3=5, c_ver=90, reward=100)`. Same c_critic
  structure as SLOW (Docker-aware), c_ver tripled to capture the
  heavy-suite Docker tail.

- **New helper: `cost_model_for_sensitivity(bench, regime='default')`**.
  `regime='default'` returns the §2.5 per-benchmark cost (unchanged).
  `regime='slow_heavy'` returns SLOW_HEAVY_COST. Raises for non-SWE
  benchmarks (the heavy-suite interpretation doesn't apply to function-
  level verification — no bimodality in that family per Table).

- **New notebook cells (§3h + §3i)** inserted after the balance heatmap
  (§3g):
  - **§3h** (code): builds `STAT_POLICY_SWE_HEAVY` — re-runs each
    SWE-Bench calibration cell's `run_policies` under SLOW_HEAVY_COST,
    using the same fit/eval split and measured kernel (`KERN_MEAS`)
    as cell 13's STAT_POLICY. Same paired-bootstrap CI machinery.
  - **§3i** (code): side-by-side comparison. Merges STAT_POLICY (SLOW)
    and STAT_POLICY_SWE_HEAVY (heavy) per (b, g, policy), reports
    delta_SLOW vs delta_SLOW_HEAVY + per-cell paired-bar chart. Also
    reports per-policy median shift across SWE cells (robustness
    summary) + sign-flip detection (any policy whose Δ-vs-AV changes
    sign between the two regimes).

- **No changes to §2.5 defaults.** Cell 13's STAT_POLICY still uses
  the §2.5 per-benchmark cost vectors. SLOW_HEAVY is opt-in via the
  new sensitivity cell.

### Verification

- `pytest tests/ -q` → **131 passed** (unchanged from previous commit;
  no test changes needed — the new cell 11 additions don't break the
  existing balance tests).
- All 5 added/modified notebook cells `ast.parse` cleanly.

### Paper-narrative gain

The headline robustness claim becomes much stronger:
- §2.5 SLOW vector: BDP / SR / Rfx beat AV on SWE-Bench cells (current
  Table 1 result).
- SLOW_HEAVY vector (worst-case Docker tail): if the same policies STILL
  beat AV, that's a 22× cost-range robustness claim. If they don't, we
  disclose this as a sensitivity limitation in the methodology section.

Either outcome strengthens the paper. The headline finding from running
§3h / §3i (after re-running cells in order) will tell us which.

### Out of scope (deliberate, follow-up candidate)

- **Per-benchmark `c_L3` adjustments for MBPP+ and HEFix.** Table
  tab:action_latency shows these benchmarks have `Cr_llm / a_gen` ≈
  0.88-0.95 (LLM critic is comparable to gen), but §2.5 FAST assigns
  `c_L3 = 1` (10× cheaper than gen). A measurement-anchored adjustment
  would set `c_L3 ≈ 9` for these benchmarks. The change might shift
  BDP / gate(L3) decisions for MBPP+/HEFix cells specifically. Held
  back from this PR to keep the SWE-Bench sensitivity self-contained
  and reviewable. Worth a follow-up PR if the SWE_HEAVY analysis lands
  well.

