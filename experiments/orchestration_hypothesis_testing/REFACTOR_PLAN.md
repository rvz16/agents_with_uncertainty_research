# Scripts directory refactor — COMPLETE

**Status:** all 6 phases landed on `refactor/scripts-cleanup`, ready to merge.

## Final layout

```
experiments/orchestration_hypothesis_testing/
├── _common/           shared library (4 modules, single source of truth)
│   ├── generators.py    GENERATORS table + _make_client
│   ├── critics.py       critic_L0_syntax, critic_L1_lint, critic_L3_review
│   ├── extract.py       extract_code
│   └── cost.py          cost_for_call + CostTracker
├── calibration/       one CLI per benchmark (no shims left in scripts/)
│   ├── lcb.py / mbpp.py / humaneval.py / humanevalfix.py
│   ├── codecontests.py / from_spotcheck.py
│   └── (legacy files left untouched: livecodebench_calibration.py,
│        iterative_calibration.py, generate_calibration_data.py,
│        compute_transition_kernel.py)
├── iter/              Self-Refine + Reflexion + harness eval
│   ├── refine.py / refine_swe.py / harness.py
│   ├── replay_baselines.py / kernel.py / swe_kernel.py
│   ├── eval_steps.py / split_predictions.py / swe_backfill_y.py
│   └── _legacy/       superseded iter pipelines (4 files)
├── analysis/          sensitivity / regime / statistical tests
│   ├── lcb_compare.py / lcb_sensitivity.py / controller.py
│   ├── lcb_baseline_kernel.py / lcb_regime_map.py / lcb_summarize_paper.py
│   ├── cver_sensitivity_sweep.py / critic_gap_sweep.py / regime_map_sweep.py
│   ├── critic_independence_test.py / cluster_bootstrap_swe.py
│   ├── mde_power.py / wilson_ci_priors.py / loo_cv_lcb.py
│   ├── apply_fdr_correction.py / compute_transition_kernel.py
│   └── l3_reviewer/   analyze.py, sweep.py, swap_reviewer.py
├── figures/           paper figures (11 scripts, fig_ prefix dropped)
├── paper/             paper-output aggregation (release.py + aggregate_with_deltas.py)
├── tools/             one-off rescue / re-eval utility (1 file currently)
├── tests/             pytest tests (60 pass)
└── scripts/           7 residual files NOT in refactor scope:
    ├── bugfix_calibrate.py        (older bug-fix calibration, kept for reference)
    ├── bugfix_table4_common.py    (helper for the bug-fix Table 4 in paper)
    ├── run_synthesis_endtoend.py  (synthesis pipeline runner)
    ├── run_synthesis_live.py      (live synthesis runner)
    ├── synthesis_train_test_split.py
    ├── synthesis_transition_kernel.py
    └── spot_check_generators.py   (1914-line generator runner, used by many scripts)
```

## Per-phase summary

| Phase | Commit | What changed |
|---|---|---|
| 0 | scaffolding | created folder skeletons + __init__.py + plan |
| 1 | 16 moves | figures/tests/tools/paper migrated, no consumers affected |
| 2 | 4 NEW modules | shared helpers extracted from lcb_calibrate into _common/ |
| 3 | 6 moves + 6 shims | calibration scripts consolidated |
| 4 | 13 moves + 1 shim | iter scripts moved (current + legacy split) |
| 5 | 18 moves + 4 shims | analysis scripts moved (incl. l3_reviewer/ subgroup) |
| 6 | -12 shims, +26 import rewrites, +1 notebook update | every consumer migrated to new paths; shims deleted |

## Total impact

- **scripts/ shrunk from 65 to 7 files** (89% reduction)
- **75 files total moved or created**
- **27 consumers updated** (26 .py + 1 .ipynb with 15 import rewrites)
- **0 behavior change** — `pytest tests/` 60/60 pass at every phase

## Migration pattern for new contributors

The notebook (`analysis.ipynb`) sets up sys.path like this:

```python
SCRIPTS  = (Path("../../orchestration_hypothesis_testing/scripts")).resolve()
PKG_ROOT = SCRIPTS.parent   # orchestration_hypothesis_testing/
if str(PKG_ROOT) not in sys.path: sys.path.insert(0, str(PKG_ROOT))
if str(SCRIPTS)  not in sys.path: sys.path.insert(0, str(SCRIPTS))
```

Then it imports via the new canonical paths:

```python
from calibration.lcb        import critic_L0_syntax, extract_code, GENERATORS
from analysis.controller    import CostModel, BayesianController
from analysis.lcb_compare   import load_lcb_trajectories
from analysis.lcb_sensitivity import run_policies
from iter.replay_baselines  import load_iter_trajectories, utility_*
```

CLI scripts in subdirs (calibration/, iter/, analysis/, figures/, paper/, tools/) do the same — `ROOT = parents[1]`, then add ROOT and ROOT/"scripts" to sys.path.

## Residual cleanup (out of scope for this PR)

The 7 files left in scripts/ form a "synthesis + bugfix" subgroup. A
follow-up could:
- Create `synthesis/` package; move synthesis_* into it
- Move `bugfix_calibrate.py` to calibration/ (or merge into humanevalfix.py)
- Move `bugfix_table4_common.py` to paper/
- Decide if `spot_check_generators.py` belongs in iter/ or _common/

The existing `calibration/` legacy files (livecodebench_calibration.py,
iterative_calibration.py, generate_calibration_data.py) are duplicate/
superseded versions of files now in their canonical locations; they were
left untouched in this PR. A follow-up could archive them to
calibration/_legacy/ for clarity.
