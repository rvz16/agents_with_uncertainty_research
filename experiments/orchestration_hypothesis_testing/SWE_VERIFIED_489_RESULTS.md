# SWE-Bench Verified expansion: 200 → 489 (sonnet45, qwen3_coder)

Result of the Verified-cell expansion from the stratified 200-instance subset to
the full **489-instance working pool** (500 − 11 build-errored). Companion to
PR #13 (`feat/swe-verified-500-expansion`). Run on Artem-1 at `baa076f`,
Aug 23–27 2026 (survived an API-key rotation, a host reboot, and several VPN drops).

## Coverage integrity (verified, not just counted)

For **both** generators, the 489 is exactly the intended pool — no gaps, no dupes:

| Check | sonnet45 | qwen3_coder |
|---|---|---|
| `full` == stratified 200 subset | ✓ exact | ✓ exact |
| `exp` == missing 289 | ✓ exact | ✓ exact |
| overlap (duplicate instances) | 0 | 0 |
| union == 489 working pool | ✓ exact | ✓ exact |
| critic rows == 489 × 3 | 1467 | 1467 |
| **base predictions with harness `Y`** | **489 / 489** | **489 / 489** |
| Self-Refine / Reflexion cover 489 | ✓ / ✓ | ✓ / ✓ |

All 489 base predictions carry a real harness `Y`, so the calibration is fully
evaluated. (Note: qwen3's *old* `_full` cell has ~300 leftover, never-critiqued
base predictions in `predictions_p0.jsonl` — harmless, since the analysis keys off
`critic_results` = the clean 489. To be cleaned during the in-place `_full` merge.)

## Calibration: critic likelihoods are stable under 2.4× more data

| Generator | Prior (pass@p0) | L0-syntax gap | **L3-critic gap** |
|---|---|---|---|
| sonnet45 | 0.390 → **0.485** | +0.08 → +0.06 | +0.26 → **+0.25** |
| qwen3_coder | 0.495 → **0.462** | +0.20 → +0.19 | +0.41 → **+0.40** |

The informativeness gaps — the load-bearing likelihood parameters for the Bayesian
controller — barely move. This directly answers reviewer **hUB2-R1** (likelihood
sensitivity): the estimates are not an artifact of the 200-subset.

## Headline: Bayes-DP advantage holds at 489

Δ utility vs `always_verify`, SWE slow-oracle cost (`c_ver=90`), measured
Self-Refine kernel, 1000-sample paired-bootstrap 95% CI:

| Generator | N=200 | **N=489** |
|---|---|---|
| sonnet45 | +51.0 [44.0, 57.5] | **+41.5 [37.2, 46.0]** |
| qwen3_coder | +40.5 [34.0, 47.5] | **+43.8 [39.3, 48.3]** |

Both stay large and strongly positive (CIs far above 0) and the **CIs tighten**.
sonnet's narrows (+51→+41.5) because its prior rose 0.39→0.49, so `always_verify`
does a bit better at the higher pass rate; qwen3's is essentially unchanged.

**Caveat:** these Δ's are **in-sample** (likelihoods + kernel fit and evaluated on
the same N) — a clean apples-to-apples 200-vs-489 check, *not* the paper's 25%
held-out protocol, so absolute values differ slightly from Table 1's held-out
numbers. The camera-ready number comes from merging into `_full` and running the
notebook's held-out pipeline.

## Cost / speed (qwen3 far cheaper & faster than sonnet)

Generation (LLM) spend — harness eval is generator-agnostic and dominates wall-clock:

| Stage (LLM $) | sonnet45 | qwen3_coder |
|---|---|---|
| Cal (generation) | ~$48.8 | $4.4 |
| Self-Refine | $26.3 | $3.1 |
| Reflexion | $22.5 | $3.7 |
| **Total generation** | **~$97.6** | **~$11.3** |

`qwen/qwen3-coder` (fast/cheap) vs `claude-sonnet-4.5` (slow/expensive) → **~8.6×**.
No further cost to finish: the recovery after the reboot was pure harness eval.

## Evidence / reproduction

- **wandb runs** (critic_results + SR/Rfx iter_records attached as
  `expansion_evidence` artifacts, metrics in the run summary):
  - sonnet45: https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing/runs/1lf30b3f
  - qwen3_coder: https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing/runs/drfjrmzw
  - project: https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing
- **Data (cluster, Artem-1):** `…/data/swebench_verified_calibration_{full,exp}/<gen>/`
  and `…_realbaselines_{selfrefine,reflexion}_{full,exp}/<gen>/`.
- **Δ computation:** `analysis/{controller,compute_transition_kernel,lcb_compare}` +
  `scripts_pipeline/run_verified_expand.sh` (PR #13).
