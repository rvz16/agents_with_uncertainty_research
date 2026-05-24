# `single_method` iter coverage audit

**Audit date:** 2026-05-24
**Source:** W&B project `nlpresearch.group/orchestration-hypothesis-testing`
**Status:** **30 of 54 (b, g) cells missing `single_method` iter runs**

## TL;DR

`bayesian_DP` in the analysis notebook relies on a measured per-step refine
transition kernel (`KERN_MEAS`) built from `single_method` iter runs. Of the
54 possible (benchmark, generator) cells across our 9 benchmarks and 6
generators, **only 24 have measured kernels**. The remaining 30 either fell
back to IID DP silently (old behaviour, masquerades as measured) or are
dropped from `bayesian_DP` (new behaviour, per PR #9 commit `ee72f332`).

## Coverage matrix (✅ = `single_method` iter exists; ❌ = missing)

| Benchmark | gpt5_mini | qwen3_coder | qwen25_32b | haiku45 | sonnet45 | gpt_oss_20b |
|---|---|---|---|---|---|---|
| lcb_easy | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| lcb_medium | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| lcb_hard | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **mbpp** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **humaneval** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| humanevalfix | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| codecontests | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| swe_lite | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| swe_verified | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |

## Per-generator summary

| Generator | single_method coverage | Missing benchmarks |
|---|---|---|
| haiku45 | 7/9 | mbpp, humaneval |
| sonnet45 | 7/9 | mbpp, humaneval |
| qwen3_coder | 6/9 | mbpp, humaneval, swe_lite |
| gpt5_mini | 4/9 | mbpp, humaneval, humanevalfix, codecontests, swe_lite |
| **qwen25_32b** | **0/9** | (all) |
| **gpt_oss_20b** | **0/9** | (all) |

## Two distinct gap patterns

**Pattern 1 — benchmark-wide gaps.** `mbpp` and `humaneval` have zero
`single_method` runs across all 6 generators (12 missing cells = 40% of the
total gap). These benchmarks were apparently never included in the original
single_method measurement campaign.

**Pattern 2 — generator-wide gaps.** `qwen25_32b` and `gpt_oss_20b` have zero
`single_method` runs on any benchmark (18 missing cells = 60% of the gap).
Both generators were added to the project **after** the single_method
campaign ran — they were never measured.

The remaining 5 cells (gpt5_mini on humanevalfix/codecontests/swe_lite;
qwen3_coder on swe_lite) are individual partial-campaign holes.

## What "single_method" actually means

`single_method` is a **layout marker from `upload_runs.py`**, not an algorithm
name. From `experiments/orchestration/wandb/upload_runs.py:362-366`:

```python
def upload_iter_orch(args, existing):
    # Single-method iter (old layout: one transition_kernel.json per cell)
    for bench, subdir in ORCH_ITER_BENCHMARKS:
        ...
        cell = DATA_ROOT / subdir / gen   # NO method subdirectory
        tk = cell / "transition_kernel.json"
        ir = cell / "iter_records.jsonl"
```

vs the newer per-method layout (`upload_runs.py:418-427`):

```python
# Per-method realbaselines (newer layout: selfrefine/ + reflexion/ subdirs)
for bench, subdir in ORCH_REALBASELINES_BENCHMARKS:
    ...
    for method in ["selfrefine", "reflexion"]:
        cell = DATA_ROOT / subdir / gen / method   # method-subdirectory
```

Algorithmically `single_method` was **vanilla generate→refine→refine→…**
with no critique or reflection stages — just the simplest iterative refine
loop, used purely to measure the empirical refine transition kernel
P(Y_{t+1} | Y_t, refine). That kernel feeds `bayesian_DP`'s dynamic-
programming planner.

**Selfrefine and reflexion trajectories are NOT a drop-in substitute** —
their refine step is conditioned on a prior critique/reflect LLM call, so
the transition dynamics differ from vanilla refine.

## Why no new single_method runs can be produced today

Both current iter scripts only accept selfrefine or reflexion:

```python
# refine.py:1330 and refine_swe.py:515:
parser.add_argument("--method", required=True, choices=["selfrefine", "reflexion"])
```

There is no CLI to produce `single_method` data. The legacy script that
produced it is not in the codebase any more. To fill the 30 missing cells,
one of these has to happen:

1. **Add `--method single_method` to `refine.py` / `refine_swe.py`** — the
   simplest path. Add a third branch that runs only the refine call, skipping
   the Stage-1 critique/reflect LLM call. See `SINGLE_METHOD_RUNBOOK.md`.
2. Re-implement the legacy vanilla-refine script from scratch.
3. Compute the kernel from selfrefine or reflexion trajectories instead
   (methodologically dirty — see "Three honest paths" below).

## What the loud-failure patch changed (PR #9 commit `ee72f332`)

Previously, cells without a measured kernel silently fell back to the IID
DP branch in `run_policies`, mixing measured-kernel and IID `bayesian_DP`
rows in `STAT_POLICY` under the same label. This made the `bayesian_DP`
comparison apples-to-oranges across generators.

The patch removes the silent fallback: cells without a measured kernel get
`bayesian_DP` REMOVED from the result dict entirely. Downstream plots show
gaps for those cells. Cell 13 prints a banner summary listing affected
(b, g) cells; cells 23/41/52/78 print per-cell `ERROR:` lines.

Touched cells: 13, 23, 41, 52, 78.

## Implications for the paper

1. **Headline `bayesian_DP`-vs-AV win claims** are anchored on 24 measured
   cells, not the 54 total. The paper currently doesn't qualify which
   (benchmark, generator) the bayesian_DP measurement is anchored on.

2. **mbpp and humaneval bayesian_DP figures** (cells 33, 35, 40, 41, 52,
   etc.) will go entirely blank for the bayesian_DP policy under the
   loud-failure patch — these entire benchmark columns lose bayesian_DP.

3. **The cross-benchmark robustness claim** "BDP is robust across
   benchmarks" loses 2 benchmarks (mbpp, humaneval) entirely.

4. **qwen25_32b and gpt_oss_20b appear nowhere** in any bayesian_DP plot
   (zero cells out of 9 for each).

## Three honest paths

### (A) Run single_method to fill the 30 missing cells

- Add `--method single_method` to `refine.py` / `refine_swe.py` (small
  change — see `SINGLE_METHOD_RUNBOOK.md`).
- Run 30 iter campaigns × ~30-60 min each = **~15-30 h wall-clock compute**,
  parallelisable across generators / GPUs.
- Upload via `upload_runs.py --track orchestration`.
- After: uniformly measured `bayesian_DP` across all 54 cells. Methodologically
  cleanest; matches what was originally intended.

### (B) Revert the loud-failure patch — accept mixed-source DP

- Restore the silent IID fallback. `bayesian_DP` becomes a mix of measured
  (24 cells) + IID (30 cells), labeled identically.
- Visually complete plots but methodologically dirtier; the bayesian_DP
  column hides which cells are measured vs not.
- Disclose the mix in the paper's methodology section: "bayesian_DP uses a
  measured single_method refine kernel where available (24/54 cells) and
  falls back to the IID DP variant otherwise."

### (C) Keep the loud-drop patch — show gaps explicitly

- Current state. bayesian_DP appears only where a measured kernel exists
  (24 cells). The other 30 cells show empty bayesian_DP in plots.
- Most defensible methodologically: every bayesian_DP datapoint in the
  paper traces to actual measurement.
- Visual cost: the bayesian_DP column has obvious gaps in mbpp, humaneval,
  qwen25_32b, and gpt_oss_20b.

## Recommendation

**(C) for now, (A) as the followup.** Ship the loud-drop version with a
clear methodological caveat in the paper appendix; in parallel, do the
single_method measurement campaign (Path A) and back-fill the gaps before
final submission. Path B is a temptation to avoid — mixing measured and
IID under the same label is the kind of methodological corner-cut that
gets flagged at review.

## Outstanding work tracked on this branch

- [ ] Decide path (A / B / C) — see `SINGLE_METHOD_RUNBOOK.md` for Path A
  implementation details.
- [ ] If A: implement `--method single_method` in `refine.py` and
  `refine_swe.py`.
- [ ] If A: run the 30 missing cells, upload, re-verify.
- [ ] If B: revert commit `ee72f332` on PR #9 branch.
- [ ] Paper methodology section: document the bayesian_DP kernel-coverage
  story however path resolves.
