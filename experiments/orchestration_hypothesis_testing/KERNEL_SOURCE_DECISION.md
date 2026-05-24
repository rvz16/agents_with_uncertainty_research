# bayesian_DP kernel source: decision and rationale

**Decision date:** 2026-05-24
**Status:** committed on branch `single-method-coverage`

## Decision

`KERN_MEAS` in `experiments/orchestration/wandb/analysis.ipynb` (cell 13)
sources from **SELFREFINE iter runs uniformly**. Single_method iter runs
are NOT used for `KERN_MEAS`. Cells without a selfrefine kernel get
`bayesian_DP` DROPPED — no silent IID fallback.

## Why selfrefine

`bayesian_DP` plans against an empirical refine transition kernel:

> P(Y_{t+1} | Y_t, refine_action)

Selfrefine trajectories ARE the empirical refine kernel for the agent's
deployed algorithm. A production code-fixing agent naturally includes a
self-critique step in its refine action; that's how good agents work, not
an extra layer to strip away. The kernel measured from selfrefine
trajectories captures the actual refine dynamics, including the
agent's self-assessment signal.

The previous design used `single_method` (vanilla refine, one LLM call per
step, no critique) for the kernel. That measured a stripped-down algorithm
the agent doesn't actually run at deployment, creating a mismatch between
the planner's model and runtime behaviour.

## Cost-vector interpretation (Fix i)

`c_gen = 10` represents the cost of **one refine primitive call**, treated
as a black box. Selfrefine's internal critique+refine is the agent's
refine action, charged uniformly with `c_gen`. We don't double-charge for
the internal critique step because the cost vector models the agent's
refine primitive at its natural abstraction level.

Paper-side methodology note (to add):

> bayesian_DP plans against the empirical refine kernel measured from the
> agent's deployed refine algorithm (selfrefine). The cost vector treats
> one refine primitive call as a single c_gen unit, abstracting over any
> internal sub-steps (critique, self-reflection, chain-of-thought) that
> the refine algorithm performs. This matches the "refine = one improvement
> attempt" abstraction implicit in the policy specification.

## Selfrefine kernel coverage matrix

As of 2026-05-24, **48 of 54 cells** have a selfrefine iter run in W&B:

| Benchmark | gpt5_mini | qwen3_coder | qwen25_32b | haiku45 | sonnet45 | gpt_oss_20b |
|---|---|---|---|---|---|---|
| lcb_easy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| lcb_medium | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| lcb_hard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| mbpp | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| humaneval | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| humanevalfix | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| codecontests | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| swe_lite | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| swe_verified | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

### Remaining 6-cell gap

| Generator | Missing benchmarks |
|---|---|
| gpt5_mini | codecontests, humanevalfix |
| qwen3_coder | codecontests, humanevalfix |
| gpt_oss_20b | swe_lite, swe_verified |

For these 6 cells, `bayesian_DP` is dropped explicitly (loud `ERROR:` log
in cell 13 + per-cell prints in cells 35 / 46 / 72; end-of-cell summary
banner listing affected cells).

### Path to 54/54 coverage

Re-run **selfrefine iter** for the 6 missing cells (NOT single_method).
Two of them (gpt5_mini, qwen3_coder × codecontests/humanevalfix) need
calibration first; the other 4 are calibration-ready.

ETA (similar arithmetic to the earlier single_method estimate):
- gpt5_mini + qwen3_coder × humanevalfix + codecontests = 4 cells × ~25 min = ~1.5 h
- gpt_oss_20b × swe_lite + swe_verified = 2 cells × ~150 min = ~5 h
- **Total: ~6.5 h sequential or ~2 h with two parallel hosts.**

Selfrefine has 2 LLM calls per step (critique + refine) so wall-clock is
roughly 2× a single_method run.

## What changed in the notebook (single-method-coverage branch)

| Cell | What |
|---|---|
| 13 | `KERN_MEAS` filter `_k.method == "single_method"` → `_k.method == "selfrefine"`. bayesian_DP `else` branch drops the policy with a loud per-cell `ERROR:` print + tracking in `BDP_NO_KERNEL_CELLS`. End-of-cell summary banner if any cells were dropped. |
| 35 | §7 winner grid: drops `bayesian_DP` when KERN_MEAS lacks the cell (winner is computed over remaining policies). |
| 46 | All-policies-per-model plotter: drops `bayesian_DP` when KERN_MEAS lacks the cell. |
| 72 | Regime grid helper: same drop-on-missing pattern. |

## What this supersedes

This decision supersedes the original `SINGLE_METHOD_AUDIT.md` Path A
recommendation (run 30 new single_method campaigns). `SINGLE_METHOD_RUNBOOK.md`
is removed since we're not pursuing that work — its content is preserved
in git history if ever needed.

## Outstanding follow-up

- [ ] Run selfrefine for the 6 missing cells (calibration first for 2 of
  them). See ETAs above.
- [ ] Paper methodology section: state that bayesian_DP's kernel is
  derived from selfrefine trajectories under the "refine primitive"
  cost-accounting choice (Fix i).
- [ ] If PR #9 (cost-vector-balance-search) merges, port these cell-13
  and cell-35/46/72 patches into the new cell numbering (cells 13/41/52/78
  on that branch).
