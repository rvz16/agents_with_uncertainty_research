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
--max-verifications 1      # see below
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

## Why `--max-verifications 1`, and why 0 is wrong here

On this branch `verify` is not in `ACTION_SPACE`:

```python
ACTION_SPACE = ("generate", "critic_L0", "critic_L2", "critic_L3", "finish")
VALID_ACTIONS = set(ACTION_SPACE)
```

and the router only accepts `raw in VALID_ACTIONS`, so it cannot call `verify`
mid-episode. There is no intermediate-verify leak to close — it is closed
structurally, by the action space.

What `max_verifications` gates here is the *terminal* verification. The guard in
`maybe_final_verify` reads:

```python
if state.get("fixed") or n_verifications >= max_verifications:
    return state
```

At 0 that is `0 >= 0` — true — so the terminal verify is skipped, **no label is
produced at all**, and every episode records `fixed=False`. A smoke run with 0
came back 0/4 solved, `final_action` = `finish` / `max_steps`, and no
`final_verify` in any trajectory. The failures were an artifact, not results.

`initial_state` clamps the value to at most 1, so 1 is both the floor and the
ceiling: exactly one terminal verification, none before it.

**This differs from the `src/code_uq` extraction** (`student_askutsakov_setup`,
`clearml_clean_run`), where `verify` *is* an agent action and the analyzer folds
its outcome into the belief before the final label. There 0 is required, and
1 leaks: measured on gpt-oss-20b / lcb_hard, same split and seed, one flag apart,
`bayes_state` scored 0.995 vs 0.241 and `tool_success` 0.947 vs 0.251, while
`verbalized` (0.466 vs 0.465) and `seqprob:mean` (0.620 vs 0.563) barely moved.
Runs from the two lineages are not comparable on the belief signals.

## `--max-tokens-generation` must be passed, not defaulted

The agent's own default is 4000. For gpt-oss-20b that is a budget rather than a
circuit breaker: the model spends 12k-30k tokens reasoning before it emits the
answer channel, so at 4000 the answer never arrives, `raw` comes back empty, and
the generation is recorded as `skipped: generation returned no answer content`.
Skipped generations do not count against `--max-generations`, so the agent
retries until `--max-steps` and the episode ends with no candidate and no label.

A smoke run hit this on 1 of 4 instances: all 20 steps of instance 3674 were
empty generations at exactly 4000 completion tokens. The same instance on the
extraction branch produced 5 candidates at 6.9k-30.8k tokens. Across the same
four instances, old branch solved 3 and this configuration solved 1.

The extraction branch had already raised its default to 32768 with the rationale
"at 4000 about 10 percent of steps were truncated ... at 32768 truncation is nil
and the average cost barely moves — only the tail pays". The run script now
passes the value explicitly so it does not depend on which lineage the agent
came from.

## L3 needs a working key and its own calibration pass

Two independent things bite here, and both did on the first smoke run.

The reviewer is called through OpenRouter. If the key is dead the calls come back
`403 Forbidden`, every L3 verdict is `None`, and the run still finishes — the
console shows `critic_L3 0/10 success_rate=0.000` and `cost=$0.0000`. Check those
two numbers before trusting an L3 channel.

Separately, the analysis step needs L3 verdicts for the *train* split, which the
generation pass does not produce:

```
RuntimeError: missing saved L3 train-calibration results for 2 instances
(2921, 3682); resume the agent with --calibrate-l3 first
```

So a run that reaches analysis needs `--calibrate-l3` in the generation command;
the run script passes it whenever `CALIBRATE_L3=1`, which is the default.

Note the key is passed as a docker `-e` argument and the agent echoes the full
docker command into the task console, where it is readable by anyone with access
to the task.

Before a long run, ping the reviewer yourself — a dead or blocked key otherwise
surfaces only as `critic_L3 0/N success_rate=0.000` with `cost=$0.0000` hours
later, because httpx logs the status line without the body:

```bash
curl -s -w '\nHTTP %{http_code}\n' https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"anthropic/claude-haiku-4.5","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

A key that works from a laptop can still fail from a cluster. From the CSCS
agents the same key, endpoint and model return

```json
{ "success": false, "error": "Access denied by security policy." }
```

which is not OpenRouter's error shape — that is an egress filter answering, not
the API, so no key or account change fixes it. Three ways out, in order of how
much they cost the experiment:

1. have the cluster whitelist `openrouter.ai` — keeps L3 an independent judge;
2. `REVIEWER_BASE_URL=http://127.0.0.1:8000/v1` with `L3_REVIEW_MODEL` set to the
   served model — the reviewer becomes whatever endpoint you are already running.
   Pointing it at the generator makes L3 a **self-review**: a weaker and
   differently-biased signal, so label such runs and never pool them with
   OpenRouter-reviewed ones;
3. `CALIBRATE_L3=0` — no L3 at all; the belief is built from L0 and L2, which is
   what our three earlier runs used.

`L3_REVIEW_MODEL` also swaps the reviewer model without touching the code.

A reviewer that reasons before answering needs a bigger budget than the built-in
4096: gpt-oss-20b reviewing its own output spent the whole budget deliberating
and never reached the JSON verdict, so 4 of 11 reviews came back
`review_error_or_unparseable` and the train calibration lost an instance — which
is fatal, since the analyzer refuses to fit likelihoods when any train instance
lacks a verdict. `L3_MAX_TOKENS=32768` gives it room. This is the same failure
as the generation ceiling above, in a different place.

## L1 is not part of this branch's belief

`ACTION_SPACE` has no `critic_L1`, so the lint channel never enters the belief
here — the belief is built from L0, L2 and L3 only. A smoke run confirms it:
`critic_L0` 11, `critic_L2` 10, `critic_L3` 10, `critic_L1` 0.

For context, in the `src/code_uq` extraction L1 *is* an agent action, but
`critic_L1_lint` there is broken four ways (`subprocess` unimported, an undefined
`path` passed to ruff, `RUFF_TIMEOUT_S` never defined, and a timeout branch
testing `result.timed_out`, which `CompletedProcess` lacks). Every verdict came
back `None` — 504 / 980 / 736 of them across three runs — while the agent kept
spending steps on it. The copy of `critic_L1_lint` on this branch is correct; it
is simply not wired into the action space.

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
