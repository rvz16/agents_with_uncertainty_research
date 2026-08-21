# Experiments — orchestration as sequential hypothesis testing

This directory contains the experiment-running infrastructure for the
EMNLP 2026 submission **"Bayesian Control for Coding Agents via
Sequential Hypothesis Testing"**.

If you're new here, this README is the entry point. Detailed docs are
listed at the bottom; this file links to them rather than duplicating.

---

## Quick orientation — which doc do I need?


| If you want to...                                                               | Read                                                                             | Length    |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------- |
| Run experiments end-to-end for the first time                                   | `[COLLEAGUE_RUNBOOK.md](COLLEAGUE_RUNBOOK.md)`                                   | 837 lines |
| Understand each policy's semantics (always_verify, BoN, gate, BG, BDP, SR, Rfx) | `[POLICIES.md](POLICIES.md)`                                                     | 614 lines |
| Extend the pipeline with a new benchmark / generator / critic                   | `[PLAYBOOK.md](PLAYBOOK.md)`                                                     | 250 lines |
| Audit log of what was run, when, and what it produced                           | `[EXPERIMENTAL_LOG.md](EXPERIMENTAL_LOG.md)`                                     | reference |
| Track ongoing code changes (what changed since the paper draft)                 | `[../../DEVELOPMENT_PROCESS.md](../../DEVELOPMENT_PROCESS.md)`                   | reference |
| Reproduce paper figures + Table 1 from W&B runs                                 | `[../orchestration/wandb/analysis.ipynb](../orchestration/wandb/analysis.ipynb)` | notebook  |
| Understand the W&B run schema (tags / config fields / artifacts)                | `[../orchestration/wandb/SCHEMA.md](../orchestration/wandb/SCHEMA.md)`           | 167 lines |


---

## Quick start — 5-minute smoke test

If you can run this end-to-end without error, your environment is set up
correctly:

```bash
# 1) Clone + install
git clone https://github.com/rvz16/agents_with_uncertainty_research.git
cd agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing
pip install openai datasets evalplus python-dotenv numpy scipy \
            matplotlib pyarrow wandb pytest

# 2) Put OPENROUTER_API_KEY in <repo-root>/.env
echo "OPENROUTER_API_KEY=<OPENROUTER_API_KEY>" > ../../.env

# 3) MBPP+ smoke calibration (~30s, ~$0.02)
python -m calibration.mbpp \
    --output-dir /tmp/smoke_mbpp \
    --generators haiku45 \
    --n-instances 2 --n-patches 3 --seed 42 \
    --max-cost-usd-per-model 0.5

# 4) MBPP+ iter Self-Refine smoke (~10s, ~$0.01)
python -m iter.refine \
    --method selfrefine --variant mbpp \
    --src-dir /tmp/smoke_mbpp \
    --output-dir /tmp/smoke_mbpp_iter \
    --generators haiku45 \
    --n-instances 2 --steps 2 --seed 42 \
    --max-cost-usd-per-model 0.5

# 5) Unit tests
python -m pytest tests/ -q
```

For SWE-Bench / Qwen32B / open-weight setups, see the relevant doc above.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────┐
│  calibration/<bench>.py                         │
│    K patches per instance × N instances         │
│    Critics: L0 syntax · L1 lint · L2 tests · L3 │
│    Oracle Y                                     │
│    → critic_results.jsonl + likelihood_tables   │
└───────────────────────┬─────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   iter/refine.py   iter/refine_swe.py  (no iter)
   (LCB, MBPP+,     (SWE-Bench Lite +     analysis straight
    HumanEval+,      Verified — needs     from calibration
    HEFix, CC)       Docker harness)
        │               │
        │      ┌────────┘
        │      ▼
        │   iter/harness.py    (Docker eval, backfills Y)
        │      │
        └──────┴──────► W&B ──► analysis.ipynb ──► paper figures + Table 1
```

---

## Directory layout

```
experiments/orchestration_hypothesis_testing/
├── _common/           Shared library — single source of truth
│   ├── generators.py    GENERATORS table + _make_client
│   ├── critics.py       critic_L0/L1/L3
│   ├── extract.py       extract_code (from LLM response)
│   └── cost.py          cost_for_call + CostTracker
├── calibration/       One CLI per benchmark — step-0 patches + critics
│   ├── lcb.py           LiveCodeBench (hard/medium/easy via --difficulty)
│   ├── mbpp.py          MBPP+
│   ├── humaneval.py     HumanEval+
│   ├── humanevalfix.py  HumanEvalFix (bigcode/humanevalpack)
│   ├── codecontests.py  CodeContests (parquet-only loader, 63 MB)
│   └── from_spotcheck.py
├── iter/              Iterative refinement (Self-Refine + Reflexion)
│   ├── refine.py        LCB / MBPP+ / HumanEval+ / HEFix / CC
│   ├── refine_swe.py    SWE-Bench (separate due to Docker)
│   ├── harness.py       Post-iter Docker eval (SWE only)
│   ├── replay_baselines.py
│   ├── kernel.py / swe_kernel.py / swe_backfill_y.py
│   └── _legacy/         Single-method iter, kept for kernel computation
├── analysis/          Sensitivity / regime / statistical tests
│   ├── lcb_compare.py / lcb_sensitivity.py / controller.py    (notebook imports these)
│   ├── compute_transition_kernel.py
│   ├── critic_gap_sweep.py, cver_sensitivity_sweep.py, ...
│   └── l3_reviewer/   Subgroup: L3-reviewer-specific analyses
├── figures/           Paper-figure-generating CLI scripts
├── paper/             Paper-output aggregation
├── tools/             One-off rescue / re-eval utilities
├── tests/             pytest tests (60 currently pass)
├── data/              Processed statistics (PAPER_TABLE.csv, etc.)
├── scripts/           Residual files not yet migrated
│   ├── spot_check_generators.py    (1914-line generator runner)
│   ├── run_synthesis_endtoend.py
│   ├── run_synthesis_live.py
│   ├── synthesis_train_test_split.py
│   ├── synthesis_transition_kernel.py
│   ├── bugfix_calibrate.py
│   └── bugfix_table4_common.py
└── calibration/       (Pre-refactor "calibration/" subdir; some legacy
                       scripts here haven't been merged yet)
```

The data + W&B + notebook side lives in a sibling directory:

```
experiments/orchestration/
├── data/              Calibration outputs (humaneval_calibration/,
│                      lcb_calibration_*/, mbpp_calibration/, ...)
├── wandb/
│   ├── analysis.ipynb     THE analysis notebook (figures + Table 1)
│   ├── upload_runs.py     Uploads local data/ outputs to W&B
│   ├── README.md
│   └── SCHEMA.md          Run-tag / config-field / artifact convention
└── src/abbo/          Python package (agent-bugfix-bayes), being retired
```

---

## What's already on W&B (don't re-run these)

Project: `[nlpresearch.group/orchestration-hypothesis-testing](https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing)`

```bash
pip install wandb && wandb login
python3 -c "
import wandb
api = wandb.Api()
runs = list(api.runs('nlpresearch.group/orchestration-hypothesis-testing'))
print(f'Total runs: {len(runs)}')"
```

Approximate inventory (2026-05-23):


| Experiment type     | # runs | Coverage                                                |
| ------------------- | ------ | ------------------------------------------------------- |
| `calibration`       | 35     | 5 generators × 7 benchmarks                             |
| `iter`              | ~25    | LCB×3 + SWE×2 tiers × 5 gens × {single_method, SR, Rfx} |
| `policy_comparison` | 140+   | Each calibration × kernel variant                       |
| Sensitivity sweeps  | ~50    | c_ver, theta, R, methodology                            |


What's still missing (the focus for new runs):

- **`Qwen2.5-Coder-32B`** on HEFix + CC (cal + iter); on MBPP+ / HumanEval+ (iter only).
- **`gpt-oss-20b`** on most benchmarks.
- **`Qwen2.5-Coder-7B`** on most benchmarks. (Add to `_common/generators.py` first.)
- **`gpt-5-mini`** on HumanEvalFix + CodeContests.

For the live "what's missing" picture, run `analysis.ipynb` cell 21
(data inventory) — it renders a green/red matrix per (benchmark,
generator, policy) cell against the paper's `tab:full_results` panel.

---

## Two cost vectors (used by analysis.ipynb)

The paper reports two cost regimes. The notebook's `STAT_POLICY` cell
(cell 13) evaluates every policy under whichever vector the cell's
benchmark is assigned to in `cell 11 §2.5`:


| Regime          | $c_\text{ver}$ | $c_\text{gen}$ | $c_{L_0/L_2/L_3}$ | Benchmarks                                               |
| --------------- | -------------- | -------------- | ----------------- | -------------------------------------------------------- |
| **SLOW_ORACLE** | 30             | 5              | 1 / 2 / 5         | SWE-Bench Lite + Verified                                |
| **FAST_ORACLE** | 5              | 10             | 1 / 1 / 1         | LCB tiers, MBPP+, HumanEval+, HumanEvalFix, CodeContests |


Assignment is "honest matching" — each benchmark goes into the regime
where its measured median `c_ver / c_gen` ratio lives. The two regimes
bracket the analytic c_ver/R crossover identified in the sensitivity
section.

---

## Common gotchas

### Post-hoc UHead scores for SAGE trajectories

`scripts/score_sage_uhead.py` reconstructs the exact prompt and completion token
IDs saved by `different_agents/v4/lcb_llm_tool_agent.py`, captures hidden states
with vLLM, and writes one UHead score per generation to
`<benchmark>__<generator>.uhead.jsonl`. Validate trajectory/logprob alignment
without loading the model first:

```bash
python scripts/score_sage_uhead.py \
    --run-root /path/to/run \
    --benchmark lcb_hard \
    --generator gpt_oss_20b_local \
    --dry-run
```

The full scorer must run in the project UHead environment with `torch`,
`transformers`, `vllm`, `lm-polygraph`, `luh`, and
`utils.hook_hs_extension.HookHiddenStatesExtension` available. GPT-OSS uses the
default compatible head; pass `--uhead` explicitly for Qwen or another head.
The analysis script automatically consumes the sibling `.uhead.jsonl` file.

- **Path note (post-refactor):** Old `python scripts/X.py` invocations
are now `python -m <pkg>.<X>` (e.g. `python -m calibration.lcb`,
`python -m iter.refine`). CLI args are unchanged.
- **vLLM endpoints:** Open-weight generators (`qwen25_32b`, `qwen25_7b`,
`gpt_oss_20b`) need a local vLLM serving on the port baked into
`_common/generators.py`. See `COLLEAGUE_RUNBOOK.md §0.4`.
- **SWE-Bench harness:** Requires Docker/Podman + `x86_64` images.
Mac users: do Phase 1 (generation) locally, Phase 2 (harness eval)
on a Linux box. See `COLLEAGUE_RUNBOOK.md §2`.
- **CodeContests dataset:** uses a parquet-only loader (63 MB) instead
of `load_dataset()` which would pull 13 GB.
- `**.env` auto-loading:** All entry scripts (`calibration/*.py`,
`iter/refine.py`) walk up 5 parent dirs from the script looking for
`.env`. You don't need to `export OPENROUTER_API_KEY`.
- **Seed convention:** `--seed 42` everywhere. The notebook's 75/25
train/eval split is deterministic per cell at seed=42, so paired
comparisons across generators line up.

---

## Where to ask for help

- File issues at [github.com/rvz16/agents_with_uncertainty_research/issues](https://github.com/rvz16/agents_with_uncertainty_research/issues).
- Slack: ask Karim or Vlad.
- For paper-side questions: see `../../emnlp2026/initial/` for the
current draft and `EXPERIMENTAL_LOG.md` for what each Table 1 cell
was produced from.
