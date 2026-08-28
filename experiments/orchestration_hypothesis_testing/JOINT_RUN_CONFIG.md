# Shared run configuration

One branch, one config, so runs from either side are comparable. Two people
producing numbers under quietly different flags is how the leak below survived
as long as it did.

## The flags

```bash
--n-instances 0            # whole benchmark
--train-fraction 0.25      # agent default is 0.5 -> a different, smaller test set
--split-seed 42
--platform leetcode
--lcb-version all
--prior-patches 1
--private-test-cap 0       # 0 means ALL private tests, not zero of them
--max-verifications 0      # see below
--max-steps 20
--max-generations 5
--max-tokens-decision 4096
--max-tokens-generation 32768
--agent-backend sage
--final-verify
--top-logprobs 20
```

Check the split before comparing anything: `n_total`, `n_test` and a hash of
`test_ids` from `<stem>.split.json` must match on both sides. With the flags
above, gpt-oss-20b gives:

| benchmark    | n_total | train | test | sha256(test_ids)[:12] |
|--------------|--------:|------:|-----:|-----------------------|
| lcb_hard     |     102 |    26 |   76 | `9687899430cc`        |
| lcb_medium   |     207 |    52 |  155 | `adb6b0c4f57a`        |
| codecontests |     165 |    41 |  124 | `b36ba82b14fd`        |

```bash
python3 -c "
import json,hashlib,sys
s=json.load(open(sys.argv[1]))
print(s['n_total'], s['n_train'], s['n_test'],
      hashlib.sha256('\n'.join(sorted(map(str,s['test_ids']))).encode()).hexdigest()[:12])
" <run_root>/lcb_hard__gpt_oss_20b_local.split.json
```

## Why `--max-verifications 0`

An intermediate `verify` runs the private tests — the same tests that produce
the label — and `analyze_lcb_llm_tool_agent_logs` folds its outcome into the
belief *before* the final label: a failed verify collapses the belief to 0.05,
and that value propagates through every later update. `bayes_state` is read off
just before the final verify, so it has already seen the oracle.

Measured on gpt-oss-20b / lcb_hard, same split, same seed, one flag apart:

| signal            | 1 intermediate verify | 0 |
|-------------------|----------------------:|--:|
| `bayes_state`     | **0.995**             | 0.241 |
| `tool_success`    | **0.947**             | 0.251 |
| `verbalized`      | 0.466                 | 0.465 |
| `seqprob:mean`    | 0.620                 | 0.563 |

Only the two signals that touch the verifier move; the logprob-based ones stay
put. That is the signature of a leak rather than of a harder configuration.

`initial_state` clamps the flag to at most 1, so passing 2 was never different
from passing 1 — and 1 is already the leaking case. Raising
`--private-test-cap` to 0 made it worse, not better: the intermediate check now
runs the *full* label suite instead of a 12-test subset.

If the agent needs `verify` as a tool, the alternative is to keep it but stop
folding it into the belief (drop the `belief = 0.05` branch and exclude
intermediate verifies from `bayes_state`). Then `tool_success` needs the same
treatment, since it is computed from the same verdicts.

## Known bug that does not affect this branch

`critic_L1_lint` is broken in the `src/code_uq/` extraction used by
`student_askutsakov_setup` and `clearml_clean_run`: `subprocess` unimported, an
undefined `path` passed to ruff, `RUFF_TIMEOUT_S` never defined, and a timeout
branch testing `result.timed_out`, which `CompletedProcess` lacks. Every verdict
came back `None` — 504 / 980 / 736 of them across three runs, none reaching the
belief, while the agent kept spending steps out of a 20-step budget. The copy on
this branch is correct; runs made on those two branches are not comparable to
runs made here on the L1 channel.

## Analysis

`scripts/uq_analysis/` holds the post-hoc toolkit. It was written against the
`src/code_uq` package layout, so `_compat.py` registers the aliases; every
script imports it and then runs unchanged on this branch.

```bash
cd experiments/orchestration_hypothesis_testing/scripts/uq_analysis
python3 prr_table.py                       # all signals x all runs
python3 nested_compare.py --run-root <root> --benchmark codecontests
python3 kernel_ablation.py                 # placeholder vs measured vs no kernel
python3 binary_bayes_uq_only.py            # belief from UQ alone
```

The first run on a given run_root parses the multi-GB logprob sidecar and caches
it under `uq_analysis/cache/`; later runs take seconds.
