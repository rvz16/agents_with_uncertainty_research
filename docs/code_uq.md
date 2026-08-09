# Code and SWE trajectory UQ

`code_uq` is the source-only extraction of the `uq_exps` live runner. It
collects SAGE-controlled trajectories for LCB, MBPP+, HumanEval+,
HumanEvalFix, CodeContests, SWE-Bench Lite, and SWE-Bench Verified.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[code,test]'
```

Install the SWE harness only when running SWE-Bench:

```bash
.venv/bin/pip install -e '.[swe]'
```

## Generate and save trajectories

Point the generator key at an OpenAI-compatible endpoint. L3 review uses
`OPENROUTER_API_KEY` unless the generator configuration says otherwise.

```bash
export GENERATOR_KEY=gpt_oss_20b_local
export GPT_OSS_20B_BASE_URL=http://127.0.0.1:8000/v1
export OPENROUTER_API_KEY=...
export BENCHMARKS=lcb_hard,lcb_medium,mbpp,humanevalfix,codecontests
export RUN_ROOT=runs/code_uq/gpt_oss_20b

bash scripts/run_code_uq.sh
```

The runner saves final records, flat actions, generation logprobs, train/test
split, prior-calibration rows, and optional Verbalized2S scores. `runs/` is
ignored by Git.

## Analyze historical code/SWE logs

`run_code_uq.sh` runs the baseline analyzer automatically. Additional methods
are available as modules:

```bash
python -m code_uq.analysis.aggregate_trajectory_uq --help
python -m code_uq.analysis.experiment2_uq_bayes_critic --help
python -m code_uq.analysis.experiment2c_multicritic --help
python -m code_uq.analysis.verbalized_trajectory --help
python -m code_uq.analysis.entropy_kl_trajectory --help
python -m code_uq.analysis.kl_between_gens --help
python -m code_uq.analysis.ntokens_baseline --help
python -m code_uq.analysis.paired_bootstrap_uq --help
```

The compatibility analyzer covers the historical `uq_exps` schema. New
environments should emit the compact schema consumed by
`trajectory-uq-analyze` instead of copying these benchmark-specific modules.

## Data policy

Do not commit raw JSONL, raw token distributions, benchmark datasets, Docker
artifacts, `.env`, or API credentials. Commit source, configs, small fixtures,
aggregate tables, and reports only.
