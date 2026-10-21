# OSWorld as a third domain: plan for fine-tuning the prefix UQ model

Status: proposal, nothing run yet. Written 2026-10-22.

Source data: `ArtemVazhentsev21/osworld-qwen-trajectories` on the Hugging Face
Hub (CC-BY-4.0), a single 8.8 GB tar. Everything in "What the data actually is"
below was measured from the first 80 MB of that archive, not taken from the
dataset card.

## Why this domain, and why now

Every cross-domain claim we currently make rests on **one pair**, ALFWorld
against DeepSWE: hold out the environment and PRR goes to -0.147 in one
direction and +0.077 in the other. With two domains it is not possible to
separate "the model learns an environment" from "these two environments happen
to look nothing alike". A third domain turns a single pair into three
leave-one-domain-out points.

Two properties make this particular dataset cheap:

1. **The executor is `Qwen/Qwen3.6-35B-A3B`** -- the same model that drives our
   RQ, SQ and DQ cohorts. Our present cross-domain comparison confounds the
   environment with the mix of executor LLMs on each side; here the executor is
   held fixed, so the domain axis is clean.
2. **The episodes are already in our schema.** `uq_episode.json` carries
   `schema_version: trajectory-uq/v1`, with per-step `signals`, per-step
   `critics` and per-step `metadata`. No collection run, no vLLM time: the
   inputs our Bayes tool-critic teacher consumes are already on disk.

## What the data actually is

Measured on the main run (`full-2gpu-max100-20260911/osworld-qwen-100/`):

| quantity | value |
|---|---|
| episodes | 360, one per OSWorld-Verified task (`environment: osworld_verified`) |
| steps | 14 132 total; median 24, mean 39, min 3, max 100 (the step cap) |
| outcome | `success` 162 / 360 = 0.45; `osworld_score` > 0 on 172; mean score 0.472 |
| partial credit | 10 episodes strictly between 0 and 1 |
| apps | chrome 61, impress 51, calc 46+18+1, os 44+1, writer 34, gimp 32, vs_code 26+3, thunderbird 23, vlc 19 |
| per-step signals | `sum_logprob`, `mean_token_logprob`, `sequence_probability`, `perplexity`, `num_tokens`, `mean_token_entropy`, `kl_to_uniform`, `self_certainty`, `verb_action`, `state_change_score` |
| per-step critics | `action_format_valid`, `ui_execution_success`, `state_change`, `no_stagnation` |
| per-step metadata | `num_actions`, `num_executed_actions`, `num_controller_calls`, `requested_llm_verification`, `repeated_action`, `action_cycle`, `no_effect_loop`, `cycle_length` |
| also present | `uq_verbalized.json` (per-action and final verbalized confidence), `traj.jsonl` (action, response, reward, screenshot file), step screenshots |

`summary/results.json` and `uq/episodes.jsonl` agree exactly: 360 task ids,
1:1, every row `status: success` (meaning "the harness ran to completion", not
task success). Two recovery runs are in the archive and were not inspected --
they sit past the 80 MB slice.

Three things to settle with the author before any number is quoted:

- the card says "6 674 generations across 3 runs (main 6 315)", while the main
  run's own `episodes.jsonl` holds 360 episodes and 14 132 step-level
  generations. The counting unit does not match either figure; ask.
- for one episode, `traj.jsonl` has 133 rows while `episodes.jsonl` caps that
  episode at 100 generations. Step indexing has to be reconciled before
  prefixes are cut, or prefix *k* will not mean the same thing in the text and
  in the signals.
- the recovery runs re-ran affected tasks, so the same `task_id` can appear
  twice. Decide the dedupe rule (latest wins, presumably) before pooling.

## The two things the data does not give us

**No text observation.** The agent is screenshot-driven: the environment's reply
to an action is a PNG, and `traj.jsonl` carries only the agent's own `action`
and `response`. Our prompt template is
`[agent action] -> [environment result]`, so the right-hand slot has to come
from somewhere. Three options, in the order I would try them:

- **A (start here): synthesize the observation line from the critics.** Each
  step already carries `ui_execution_success`, `state_change`,
  `no_stagnation`, `state_change_score`, `repeated_action`, `action_cycle`,
  `no_effect_loop` -- render them as one short line, e.g.
  `executed; screen changed (0.011); no stagnation`. This keeps the modality
  identical to the existing seven cohorts, which is the whole point: if transfer
  still fails, the failure is attributable to the domain rather than to pixels.
  It also keeps the 2048-token window intact.
- **B: the vision route.** `Qwen/Qwen3.5-2B` is natively multimodal (the HF card
  lists a 24-layer vision encoder), so the backbone can take screenshots.
  `experiments/train_decision.py` cannot: it builds the model with `AutoModel` +
  `AutoTokenizer` and pools `last_hidden_state`, with no `AutoProcessor` and no
  `pixel_values`. Ten screenshots per prefix would also blow the 2048-token
  window, so this means one image (the last step) plus text for the rest. Worth
  doing as an ablation *after* A, never instead of it.
- **C: OCR the screenshots.** Noisy, new dependency, and it answers no question
  that A does not. Skip.

While on this: `target_modules=["q_proj","k_proj","v_proj","o_proj"]` matches by
name suffix across the whole module tree, so on a multimodal checkpoint the
adapters are probably also attached to the vision tower's attention -- dead
weights that never see a gradient because no image is ever fed. Check
`print_trainable_parameters()` in the next run's log either way.

**No per-step verified labels.** `reward` is `0` on every step of every episode
inspected, and `done` is true only on the last; OSWorld's validators need VM
state, and the archive ships screenshots, not snapshots. So every prefix
inherits the episode's terminal score -- exactly the `aug-term` control arm.
That is not a loss worth mourning: on DeepSWE, per-step verified labels bought
about +0.01, and on ALFWorld augmentation was negative. Do not promise
per-step labels for OSWorld; if the author still has VM snapshots, that
changes, and it is worth one question.

## Plan

**P0 -- loader (half a day, no GPU).** Download the tar once into the durable
worktree path, *not* `/private/tmp`, which purges after three days and has
already eaten code twice. Write `osworld_uq/load_osworld.py` emitting our
episode dict (`id`, `score`, `steps`, `signals`, `critics`), with
`score = features.osworld_score` (continuous; `is_continuous` will pick it up)
rather than the binary `success` flag -- the two disagree on exactly the 10
partial-credit episodes. Normalize the app names (`libreoffice calc` /
`libreoffice_calc` / `calc`, `vscode` / `vs_code`, `OS` / `os`) if folds are to
be stratified by app. Assert the loaded count against `summary/results.json`.

**P1 -- the signal table, before any training (half a day, no GPU).** Run the
existing method table on the new cohort: Logprob, Perplexity, MTE,
Self-certainty, verbalized, Bayes tool-critic, Fused H+L -- and the `-N steps`
baseline in the same table. Two reasons to do this first: it is the paper's own
table and costs nothing, and it says whether the teacher posterior is worth
distilling here *before* we spend GPU distilling it.

The length baseline is not optional here, it is the main threat. Failures run
into the 100-step cap while successes finish in 10-35 steps, so step count
alone will rank episodes well for reasons that have nothing to do with
uncertainty -- the same leak that inflates our ALFWorld column, only worse. No
OSWorld number gets quoted without the `-N steps` column beside it.

**P2 -- in-domain model (a day, ~2 h GPU).** Cut prefixes at `--stride 4`
(~3.5k prefixes from 14k steps), render the text with template A, distil
targets from the teacher, train OSWorld-only. Deliverable: the in-domain PRR,
plus Brier and ECE against the base-rate constant, since the new domain is the
cleanest place to see whether anything about our calibration is domain-specific.

**P3 -- the actual claim (~8-10 h GPU).** Leave-one-domain-out across three
domains, reusing `--holdout` / `--drop` (commit `08061e4b`): train {ALF, SWE} →
score OSW; {ALF, OSW} → SWE; {SWE, OSW} → ALF; plus the joint 10-cohort model
for the in-domain reference. Note that the four existing transfer runs were
fitted with `W_ACTION=0`, i.e. with the controller term switched off -- keep
that setting for comparability, or rerun the old pair too.

**P4 -- cohort holdout inside OSWorld (optional).** Hold out one app family
(say gimp, 32 episodes) to ask whether the "agent-agnostic, domain-specific"
story has a finer grain: is an unseen *application* as free as an unseen
scaffold was?

Compute: the 7-cohort controller run took about 21 h. OSWorld adds roughly 3.5k
prefixes to the existing 25.9k, so a joint 10-cohort run lands near 24-28 h; the
LODO runs are each smaller than that.

## Out of scope here

The `c_step` / `c_ask` sweep, saving controller weights, and the permuted-label
control are separate threads. Adding a domain does not fix the degenerate
controller, and a third domain measured with the same uncalibrated head will
reproduce the same gap between ranking and prediction.

---

## P0 / P1 results (2026-10-22, no GPU)

Loader `load_osworld.py` and the two reports live in
`~/.claude/jobs/7fd0e6ed/tmp/osworld/`, against a fresh worktree of
`decision-models-audit` at `~/.claude/jobs/7fd0e6ed/tmp/gw2` (the old `/tmp/gw`
had been purged of `belief/` and `experiments/deepswe_table.py`). Loading is a
rename, not a reconstruction: the dataset is already `trajectory-uq/v1`. Target
is `features.osworld_score`, the continuous one.

The cohort, main run only (the two recovery runs are still downloading):
360 episodes, mean score 0.472, 171 solved, 10 with partial credit, median 24
steps against a 100-step cap. **Solved episodes take a median of 14 steps,
failed ones 55.**

`experiments.prr_report_v2.method_table`, unmodified, 3 seeds x 5 folds,
PRR@0.5 on the native score. 89 of 164 methods scored (the rest want the LLM
judge, SAUP distances, UProp or HTC tokens, none of which exist here):

| rank | method | PRR |
|---|---|---|
| 1 | TemporalBelief (B4), final checkpoint | +0.693 |
| **2** | **`-N steps`** | **+0.679** |
| 3 | Logistic regression, pinned | +0.590 |
| 5 | Verb actions -- Bayes UQ-only / SEP | +0.576 |
| 7 | Logprob / mean | +0.560 |
| 85 | Tool success rate | +0.026 |

So at face value the whole cohort is episode length, and no genuine UQ signal
reaches the baseline. The mechanism is the step cap: 64 episodes stop at 100
steps and 62 of them score 0, so the single bit "did not hit the cap" is worth
+0.393 on its own.

**Under length control the picture reverses**, which is the finding that
matters:

| subset | n | `-N` | Verb SEP | Logprob | Self-cert. | B4 |
|---|---|---|---|---|---|---|
| all episodes | 360 | **+0.674** | +0.569 | +0.543 | +0.544 | +0.679 |
| capped episodes dropped | 296 | +0.393 | +0.325 | +0.398 | +0.367 | **+0.439** |
| 10-40 steps only | 167 | +0.144 | +0.237 | +0.272 | +0.207 | **+0.386** |

Inside a length-homogeneous band the baseline is nearly dead (+0.144) and the
teacher still reads +0.386, with the raw signals between +0.21 and +0.27. That
is a better-behaved domain than ALFWorld's SG cohort, where the signal *was*
the length. The teacher posterior is therefore worth distilling -- P2 stands --
but **no OSWorld number is ever quoted without the length-controlled column**,
and the controlled subset is what P3's transfer comparison should use.

One methodological trap found on the way: these subsets must be scored against
a binary target. `with_labels` splits a continuous score at the training fold's
median, and once the capped failures are removed the median score is 1.0, every
training label becomes 0, and the Bayes fits collapse to a constant. The tell
was four different methods returning one identical PRR (-0.058). The control
binarizes at 0.5, as ALFWorld does.

Still open from P0: the two recovery runs (dedupe by `task_id`, latest wins),
and the three counting questions for the author above.

### All three runs merged, and the domain control (same day)

The archive finished downloading; only the JSON was kept (56 MB of 8.8 GB), in
`~/.claude/jobs/7fd0e6ed/tmp/osworld/raw/`. Three runs: main 360 episodes /
14 132 steps, `full-recovery-infra` **1** episode / 22 steps, and
`affected-recovery` 23 episodes / 1 837 steps which are all re-runs of main-run
tasks. Merged cohort: **361 episodes**.

The counting question is half answered: the card's "+22" for the infra recovery
is exactly that run's 22 *steps*, so a "generation" is a step -- but then the
main run is 14 132, not 6 315, and the affected recovery is 1 837, not 337. One
of the three figures matches. Still worth asking.

The re-runs are not a free upgrade, and the dedupe rule is a real choice: only
3 of 23 change the label (0 -> 1), while many change the length a lot and
mostly upward (21 -> 100 steps, 20 -> 100, 31 -> 100; six land exactly on the
cap). Taking the recovery copy adds 7 capped episodes to the cohort. Both rules
were scored; the conclusions do not move:

| subset | rule | n | `-N` | B4 | Logprob SEP | Verb SEP |
|---|---|---|---|---|---|---|
| whole cohort | latest | 361 | **+0.661** | +0.649 | +0.489 | +0.592 |
| whole cohort | main | 361 | **+0.674** | +0.688 | +0.545 | +0.572 |
| capped dropped | latest | 290 | +0.325 | **+0.370** | +0.313 | +0.309 |
| capped dropped | main | 297 | +0.394 | **+0.451** | +0.341 | +0.330 |
| 10-40 steps | latest | 169 | +0.103 | **+0.377** | +0.189 | +0.244 |
| 10-40 steps | main | 168 | +0.139 | **+0.396** | +0.196 | +0.245 |

**The domain control fails, and that is informative.** OSWorld's own domain for
a task is not `metadata.related_apps[0]`: 92 of our episodes are filed under
`multi_apps`, which that field names as chrome, calc and so on. `multi_apps`
scores 0.29 and runs a median 56 steps against 13-26 elsewhere, so the obvious
hypothesis was that "long" and "failed" both just mean "multi_apps". They do
not. Scoring inside each domain (latest-wins):

| domain | n | base | `-N` | B4 | Verb SEP |
|---|---|---|---|---|---|
| multi_apps | 93 | 0.29 | +0.774 | +0.759 | +0.651 |
| libreoffice_calc | 47 | 0.36 | **+0.960** | +0.908 | +0.892 |
| libreoffice_impress | 47 | 0.45 | +0.593 | +0.582 | +0.641 |
| chrome | 46 | 0.57 | +0.315 | +0.293 | +0.284 |
| mean of the four | | | **+0.661** | +0.635 | +0.617 |

The mean over domains is identical to the whole-cohort number (+0.661), and
inside `libreoffice_calc` episode length ranks outcomes almost perfectly
(+0.96 on the merged cohort, +1.00 on the main-run copies). So the confound is
per-episode length itself, not cross-task difficulty, and the length band stays
the only control that separates anything. The six smaller domains (15-24
episodes) are single-class or too small to score.

### Ready for P2

- `load_osworld.py` now takes the canonical domain from OSWorld's
  `evaluation_examples/test_all.json` (cached as `osworld_index.json`); all 361
  ids resolve.
- **The task instruction is in none of the dataset's files** -- not
  `episodes.jsonl`, not `traj.jsonl`, not `runtime.log`, which holds only the
  model's output and the generated pyautogui code. It has to come from OSWorld
  itself, `evaluation_examples/examples/<domain>/<task_id>.json`, which
  resolves for every id (spot-checked two).
- Remaining before a training launch: fetch the 361 instructions, cut prefixes
  at stride 4 against `traj.jsonl` (reconciling its step numbering with
  `episodes.jsonl`), render the observation line from the critics, and distil
  the teacher's targets.
