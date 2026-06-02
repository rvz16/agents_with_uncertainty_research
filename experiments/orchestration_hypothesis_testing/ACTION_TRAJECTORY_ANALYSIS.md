# Action Trajectory Analysis

This note documents how we analyzed controller action trajectories in the
LCB confidence experiments.

## Runs

The original runs analyzed in `analysis.ipynb` are:

```python
base_path1 = "/capstor/store/cscs/swissai/a0142/agents_uq/final_confidence_lcb_medium_gpt_oss_120b_slurm_2354303/lcb_medium/gpt_oss_120b_local/"
base_path2 = "/capstor/store/cscs/swissai/a0142/agents_uq/final_confidence_lcb_hard_gpt_oss_120b_slurm_2355104/lcb_hard/gpt_oss_120b_local/"
base_path3 = "/capstor/store/cscs/swissai/a0142/agents_uq/final_confidence_lcb_hard_gpt_oss_120b_slurm_2355996/lcb_hard/gpt_oss_120b_local/"
```

Each run contains:

- `final_logprob_bayes_quality.csv`: one final row per instance/policy with
  log-probability scores, Bayesian states, final quality, and final action.
- `final_logprob_bayes_quality.jsonl`: metadata aligned with the final CSV.
- `controller_actions.jsonl`: chronological controller/tool actions.
- `generation_trajectory_scores.csv`: per-generation scores, including
  `bayes_state_after_generation`.

## Action Path Construction

For each `(instance_id, policy)`, we sort `controller_actions.jsonl` by
`instance_id`, `policy`, `step`, and `ts`, then convert each raw action into a
compact token:

- `generate` for code generation.
- `L0+`, `L0-`, `L2+`, `L2-`, `L3+`, `L3-` for critic outcomes.
- `verify+`, `verify-` for non-final verifier outcomes.
- `label_verifier+`, `label_verifier-` for final labeling/verifier outcomes.

The ordered tokens are joined with ` -> ` to form an `action_path`, for example:

```text
generate -> L2+ -> verify+
generate -> L2- -> generate -> verify-
```

These paths are used only to summarize and interpret what the controller did.
For uncertainty metrics, the terminal label/verifier outcome must not be used as
evidence, because it reveals the target label.

## Bayesian States

We track two Bayesian confidence states:

- `bayes_state`: posterior probability of correctness before the final
  correctness label is revealed. It includes available non-final evidence such
  as critic outcomes, non-terminal verifier outcomes, and generation dynamics.
- `bayes_state_after_generation`: posterior immediately after the last real
  generation step, before later critic/verifier evidence is incorporated.

When joining trajectory-level values back to the final table, use explicit keys
such as `instance_id` and `policy`. Avoid assigning grouped values positionally
with `.values`, because row order mismatches can silently corrupt the analysis.

## Aggregation Table

After building `action_path`, we merge paths into the final table and group by
path. For each unique path, we report:

- `n`: number of instances following the path.
- `quality_mean`: average final correctness for that path.
- `final_actions`: counts of final outcomes such as `verify_pass`,
  `verify_fail`, or `label_verifier_fail`.
- `bayes_states`: unique final Bayesian states seen on that path.
- `bayes_states_after_generation`: unique post-generation Bayesian states.
- `examples`: a few representative `instance_id` values.

This table answers questions such as:

- Which action trajectories are most common?
- Which trajectories usually succeed or fail?
- Whether final confidence comes mostly from generation quality or from later
  tool evidence.

## Interpretation

The key comparison is between confidence before and after tool evidence:

- If `bayes_state_after_generation` is weak, the generator alone is not enough
  to identify correctness.
- If `bayes_state` is strong, the controller's observed tool trajectory carries
  useful information about final quality.
- If a path ends in a successful ordinary verifier call, final correctness is
  usually already known; this is why final verifier labels are excluded from
  uncertainty evaluation and used only for descriptive path analysis.

