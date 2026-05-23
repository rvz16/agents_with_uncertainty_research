# Scripts directory refactor

**Status:** in progress on `refactor/scripts-cleanup` branch.
**Goal:** split the 65-file `scripts/` dump into purpose-organized subpackages.
**Constraint:** zero behavior change. Each phase is independently verifiable.

## Target layout

```
experiments/orchestration_hypothesis_testing/
├── REFACTOR_PLAN.md             # this file (delete on phase 6)
├── _common/                     # shared library — imported by everything
│   ├── generators.py            # GENERATORS table + _make_client
│   ├── critics.py               # critic_L0_syntax / L1_lint / L3_review
│   ├── extract.py               # extract_code
│   ├── cost.py                  # cost_for_call + CostTracker
│   └── controller.py            # CostModel + BayesianController
├── calibration/                 # one CLI per benchmark (3-patch generation + critic eval)
│   ├── lcb.py
│   ├── mbpp.py
│   ├── humaneval.py
│   ├── humanevalfix.py
│   ├── codecontests.py
│   ├── swe.py
│   ├── from_spotcheck.py
│   └── _archive/                # superseded scripts kept for provenance
├── iter/                        # Self-Refine + Reflexion + harness eval
│   ├── refine.py                # LCB / MBPP+ / HumanEval+ / HEFix / CC (variant-dispatched)
│   ├── refine_swe.py            # SWE-Bench (separate flow due to Docker)
│   ├── harness.py               # post-iter Y backfill for SWE
│   ├── replay_baselines.py      # scoring (notebook imports this)
│   ├── kernel.py
│   ├── backfill_y.py
│   └── _archive/
├── analysis/                    # sensitivity / regime / statistical tests
│   ├── compare.py               # notebook imports this
│   ├── sensitivity.py           # notebook imports this
│   ├── regime_map.py
│   ├── critic_gap_sweep.py
│   ├── critic_independence.py
│   ├── cluster_bootstrap.py
│   ├── mde_power.py
│   ├── wilson_ci.py
│   ├── loo_cv.py
│   ├── fdr.py
│   └── l3_reviewer/
├── figures/                     # one CLI per chart in the paper
├── paper/                       # paper-output aggregation (renamed from "build" — gitignore collision)
├── tools/                       # one-off rescue / surgical retry / backfill
└── tests/                       # unit tests
```

## Phased migration

| Phase | Scope | Risk |
|---|---|---|
| **0** ✅ | Create folders + `__init__.py` + this README; nothing moved | None |
| **1** | Move LOW-RISK groups (`figures/`, `tests/`, `tools/`, `paper/`) | Low |
| **2** | Extract shared helpers from `lcb_calibrate.py` into `_common/`; `lcb_calibrate.py` becomes a thin re-export | Med — 10 sibling files need import updates |
| **3** | Consolidate `calibration/` (merge existing subdir + new `*_calibrate.py` scripts); resolve `compute_transition_kernel.py` duplicate | Med |
| **4** | Move iter scripts into `iter/`; update notebook's `SCRIPTS` path | Med |
| **5** | Move analysis scripts into `analysis/`; update notebook imports | High |
| **6** | Delete compat shims; verify `scripts/` is empty; remove this file | Low |

## Risk mitigation

- **Branch policy:** all work on `refactor/scripts-cleanup`. Main stays usable for Artem.
- **Compat shims:** during phases 2–5, each renamed file gets a 1-line shim in `scripts/` so external callers keep working. Removed in phase 6.
- **Notebook:** Cells 1, 13, 15 import from `scripts/`. We update notebook imports ONLY in phase 4–5 commits that move the imported files; previous phases leave notebook untouched.
- **Verification per phase:** run `python -m py_compile <changed-files>` + spot-test one moved script before committing.

## How to follow along

```bash
# Pull the refactor branch
git fetch origin refactor/scripts-cleanup
git checkout refactor/scripts-cleanup

# After each phase commit, current state of each phase is testable in isolation
git log --oneline refactor/scripts-cleanup
```
