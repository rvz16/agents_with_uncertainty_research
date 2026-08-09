# Migration manifest

This repository is a source-only extraction from the `uq_exps` and
`new_env_test` branches of the research monorepository.

## Included from `uq_exps`

- SAGE-controlled code/SWE trajectory runner;
- LCB, MBPP+, HumanEval+, HumanEvalFix, CodeContests, SWE-Bench Lite, and
  SWE-Bench Verified adapters;
- generator, critic, cost, extraction, kernel, and logprob helpers;
- benchmark calibration helpers required by the live adapters;
- historical code/SWE log analysis;
- trajectory aggregation, Verbalized2S, entropy, self-certainty,
  KL-between-generations, token-count, multi-critic, paired bootstrap, and
  binary Bayes threshold experiments.

## Included from `new_env_test`

- ALFWorld environment, ReAct agent, sharded runner, repair utility, and tests;
- critic-only, binary, Continuous, and Tempered Continuous Bayes methods;
- compact ALFWorld conversion into the common episode schema;
- optional LLM-judge verdict support through the `llm_judge_pass` critic.

## Common portable layer

`trajectory_uq_toolkit` provides the environment-independent contract:

- resumable parallel collection through an adapter;
- compact versioned episode JSONL;
- online token-logprob reduction;
- one analyzer for UQ baselines, critic states, Bayes fusion, calibration, and
  ranking metrics.

## Deliberately excluded

- all research `runs/`, `sim_results/`, raw JSONL, raw completions, and raw
  token distributions;
- benchmark datasets, Hugging Face caches, Docker artifacts, and model files;
- `.env`, API credentials, virtual environments, Python caches, and build
  products;
- presentation sources, unrelated agents, online-kernel experiments, and
  historical notebooks;
- UHead, because no implemented and validated UHead exists in the source
  branches.

Generated data belongs under an ignored `runs/` directory. Small synthetic
fixtures and aggregate reports may be committed when they are needed for
reproducibility.
