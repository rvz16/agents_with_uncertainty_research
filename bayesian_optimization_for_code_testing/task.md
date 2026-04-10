Build a reproducible research artifact called `agent-bugfix-bayes`.

Goal
Implement and evaluate a comparison between:
1. Bayesian decision-theoretic orchestration for bug fixing and testing
2. Fixed heuristic orchestration of the same agents/tools

Important naming rule
In code, docs, and paper notes, call the method:
- "Bayesian decision-theoretic orchestration"
- "Bayesian value-of-information controller"
- "Bayesian controller"
Do not describe it as generic Gaussian-process Bayesian optimization.
Add one note in the README explaining that the Bayesian component here is sequential decision-making / hypothesis testing over bug-fixing actions.

Main deliverables
1. Exact reproduction of the synthetic benchmark from the paper.
2. A toy executable codebase version of the same benchmark.
3. A real-bug pilot on BugsInPy using offline replay and agent-generated candidate patches.
4. Optional Defects4J adapter after BugsInPy is stable.
5. Pytest + Allure experiment logging.
6. Article-ready CSV tables, JSON metrics, plots, and Markdown summary.

Working style
- Implement in phases and keep the repo runnable after every phase.
- After each phase, run tests and fix failures before moving on.
- Prefer small, well-named modules over one big script.
- Use deterministic seeds everywhere.
- Never make the evaluation stage depend on live API calls.
- Live LLM/API calls are allowed only in candidate-bank creation.
- Add docstrings and type hints to all public functions.
- Add a top-level README with exact run commands.

Tech stack
- Python 3.11+
- numpy, pandas, scipy, matplotlib, pydantic, typer or argparse
- pytest
- allure-pytest
- optionally joblib for parallel runs
- optionally docker integration for benchmark adapters
- a provider abstraction for LLM patch generation so I can plug in Claude, OpenAI, or local models later

Repository layout
agent-bugfix-bayes/
  README.md
  pyproject.toml
  requirements.txt
  .env.example
  configs/
    synthetic.yaml
    toy.yaml
    bugsinpy_pilot.yaml
    defects4j_pilot.yaml
    policies.yaml
    cost_models.yaml
  src/abbo/

  
    __init__.py
    cli.py
    config.py
    utils/
      seed.py
      io.py
      hashing.py
      timing.py
    core/
      types.py
      costs.py
      metrics.py
      stats.py
      splits.py
    synthetic/
      env.py
      exact_math.py
      policies.py
      simulate.py
    toy/
      env.py
      diagnostics.py
      patches.py
      verifier.py
      router_app/
        __init__.py
        routing.py
        tests/
    realworld/
      adapters/
        base.py
        bugsinpy.py
        defects4j.py
      candidate_bank/
        schema.py
        build.py
        replay.py
      agents/
        generator.py
        critics.py
        verifier.py
        prompts.py
      calibration/
        likelihoods.py
        transitions.py
        fit.py
      policies/
        bayes_controller.py
        heuristics.py
      evaluation/
        run_episode.py
        run_benchmark.py
        summarize.py
        plots.py
      reporting/
        allure_logging.py
        artifacts.py
  tests/
    test_synthetic_exact.py
    test_synthetic_mc.py
    test_toy_mapping.py
    test_policy_consistency.py
    test_candidate_bank_schema.py
    test_bugsinpy_adapter_smoke.py
  scripts/
    setup_env.sh
    run_synthetic.sh
    build_bugsinpy_bank.sh
    run_bugsinpy_eval.sh
    build_report.sh
  data/
    raw/
    processed/
    candidate_bank/
    splits/
  results/
    synthetic/
    toy/
    bugsinpy/
    defects4j/
  allure-results/

Phase 1: exact synthetic benchmark reproduction
Implement the paper’s synthetic benchmark exactly.

Environment
- Hidden class H in {A, B, C}
- Prior uniform by default
- Diagnostic tests T1, T2, T3
- Binary observation z in {fail, pass}
- Conditional failure probabilities:
  T1: P(fail|A)=0.9, P(fail|B)=0.2, P(fail|C)=0.2
  T2: P(fail|A)=0.2, P(fail|B)=0.9, P(fail|C)=0.2
  T3: P(fail|A)=0.2, P(fail|B)=0.2, P(fail|C)=0.9
- Patch actions PA, PB, PC
- Patch succeeds iff its label matches H
- Then one expensive verifier V
- Max 2 diagnostics, then exactly 1 patch, then exactly 1 verifier
- Costs:
  Ctest = 1
  Cpatch = 3
  Cver = 20
  Reward R = 200

Implement:
- posterior_update(belief, test, obs)
- terminal_value(belief) = R * max_k belief[k] - Cpatch - Cver
- dynamic program:
  V0(b) = terminal_value(b)
  Vr(b) = max_i [ -Ctest + sum_z P(z|b,Ti) * Vr-1(update(b,Ti,z)) ]
- generic policy solver that works for any confusion matrix and test budget
- tree extraction method that returns the optimal decision tree
- baseline policies:
  1. one_test_map: run one fixed test, update posterior, choose MAP patch, verify
  2. one_test_if_fail_else_rule: the fixed hand-written baseline from the paper style
  3. tuned_threshold_heuristic: for later comparison

Implementation detail
- Use fractions.Fraction in exact_math.py for the closed-form symbolic result so the exact values are not lost to floating-point noise.
- Also implement Monte Carlo simulation in simulate.py.
- Expose a CLI:
  python -m abbo.cli run-synthetic --episodes 100000 --seed 7

Acceptance criteria for Phase 1
- Exact Bayes expected utility equals 391/3 within numerical tolerance.
- Exact one-test MAP expected utility equals 268/3 within numerical tolerance.
- Monte Carlo estimates over 100000 episodes are close to the exact values.
- The extracted optimal tree should be:
  first T1
  if fail -> T1 again
  if pass -> T2
  then MAP patch and verify

Phase 2: toy executable codebase benchmark
Implement a tiny real codebase that preserves the same information structure as the synthetic benchmark.

Toy app idea
- A ticket-routing or request-routing package
- Three injected bug families A, B, C
- Each bug family breaks a different subset of inputs
- Diagnostics are lightweight tests or randomized probes
- Patches are deterministic code edits associated with A, B, C
- Full verifier is the complete test suite

Requirements
- The toy code should actually execute, not just simulate.
- The three diagnostics should empirically approximate the target failure matrix.
- Add a calibration script that runs each diagnostic many times per hidden bug and reports the empirical matrix.
- Keep the toy benchmark fully ground-truth controlled.

CLI
- python -m abbo.cli calibrate-toy --trials 10000
- python -m abbo.cli run-toy --episodes 10000 --seed 7

Acceptance criteria for Phase 2
- Empirical diagnostic probabilities are close to target matrix.
- Bayes still beats one-test heuristic by a meaningful margin.
- Pytest suite passes.
- Allure can capture toy-run artifacts.

Phase 3: real-world pilot on BugsInPy
Implement a real-bug workflow on BugsInPy first.

Important design choice
Use a two-stage real-world architecture:
A. candidate-bank build stage with live agent calls
B. offline replay evaluation stage with no live calls

Why
This isolates orchestration from model randomness and lets us compare Bayes vs heuristics on the same generated candidates.

3A. BugsInPy adapter
Create src/abbo/realworld/adapters/bugsinpy.py with functions to:
- checkout_bug(project, bug_id, fixed_or_buggy, workdir)
- compile(workdir)
- run_trigger_tests(workdir)
- run_all_tests(workdir)
- list_trigger_tests(workdir) if available from metadata or a project manifest
- restore_clean_copy(workdir)

Shell commands should be wrapped in Python subprocess code.
Store stdout, stderr, return code, duration, and command in structured logs.

3B. Candidate bank schema
For each bug, create a tree of candidate patches.
Store under:
data/candidate_bank/bugsinpy/<project>/<bug_id>/<root_or_parent>/<generator_arm>/<seed>/

Each candidate manifest.json must include:
- benchmark
- project
- bug_id
- parent_candidate_id
- candidate_id
- generator_arm
- generator_seed
- provider/model metadata
- prompt hash
- prompt text path
- diff path
- changed files
- patch apply success boolean
- wall clock time
- API token counts / dollar cost if available
- critic outputs:
  * trigger_tests_pass
  * smoke_tests_pass
  * static_checks_pass
  * optional cheap_judge_label
- verifier outputs:
  * full_suite_pass
  * failing_tests_before
  * failing_tests_after
  * regression_failures
- artifact paths:
  * raw model response
  * git diff
  * pytest logs
  * stack traces
  * timing json

3C. Generator arms
Implement at least 4 patch-generation arms:
- g1_direct_fix:
  single prompt, minimal patch, use issue/failing test context
- g2_localized_fix:
  first localize suspicious files/functions, then patch only top-k files
- g3_test_guided_fix:
  emphasize triggering tests and stack trace, forbid test edits
- g4_minimal_diff_fix:
  ask for smallest production-code-only change likely to fix the bug

Each arm should:
- take the current candidate/repo state as input
- create a patch diff
- apply the patch in an isolated worktree
- return a candidate package

Add prompt templates in prompts.py and keep them versioned and hashed.

3D. Candidate-bank build process
For each selected bug:
- start from the buggy version as root candidate c0
- root candidate has known Y=0 for calibration
- run all generator arms with N seeds each from root to produce depth-1 candidates
- optionally run one more generation step from each depth-1 candidate to produce depth-2 candidates
- for every candidate, run all critics and the full verifier once and save everything
- do not rerun candidates during replay; replay must consume stored artifacts only

Default pilot setting
- 20 to 50 BugsInPy bugs
- 4 generator arms
- 2 or 3 seeds per arm
- depth 1 first, then depth 2 if stable

Phase 4: calibration of the Bayesian model
Implement the real-world Bayes controller using the paper’s general generator / critic / verifier formulation.

Latent variable
Use Y in {0,1} where Y=1 means the current candidate patch is fully correct.

Belief
b_t = P(Y_t = 1 | history)

Critic likelihoods
For each critic d with binary output z:
- estimate P(z | Y=1, d)
- estimate P(z | Y=0, d)
Use Laplace smoothing.

If a critic is multi-class, estimate the categorical likelihood table.

Generator transition model
For each generator arm g, estimate:
- p01[g] = P(Y_{t+1}=1 | Y_t=0, action=g)
- p10[g] = P(Y_{t+1}=0 | Y_t=1, action=g)

Start simple:
- fit global rates from the calibration split
- optionally later fit repo-conditioned or bug-cluster-conditioned rates with hierarchical smoothing

Belief dynamics
- after critic output z: Bayes update using likelihood tables
- after generator action g: transition update
  T_g(b) = b * (1 - p10[g]) + (1 - b) * p01[g]

Value function
Finite horizon controller with small action budget.
Implement:
- Q_verify(b) = lambda_success * b - Cver
- Q_critic(d, b) = -Cd + sum_z P(z|b,d) * V(update(b,d,z))
- Q_gen(g, b) = -Cg + V(T_g(b))
Then:
- V(b, budget_state) = max over allowed actions

Budget state should include:
- remaining steps
- remaining generator actions
- remaining critic actions if you cap them
- whether verification has already occurred

Use a discretized belief grid first, e.g. 0.00 to 1.00 in steps of 0.01.
Keep the solver generic.

Cost model
Create configs/cost_models.yaml.
Store both raw and normalized costs.
Default main utility should be:
U = reward_success * 1{full_suite_pass}
    - measured_generator_costs
    - measured_critic_costs
    - measured_verifier_costs

For measured costs, start with wall-clock seconds.
Also record API dollar cost and expose an alternative utility that includes it.
Freeze the chosen utility before final test evaluation.

Phase 5: heuristic baselines
Implement several baselines using the exact same generator arms, critics, and verifier.

Required heuristics
- h1_one_shot:
  direct_fix once, then full verify
- h2_fixed_workflow:
  trigger_tests -> localized_fix -> full verify
- h3_two_stage_fixed:
  direct_fix -> trigger_tests -> if fail then test_guided_fix -> verify
- h4_threshold:
  maintain a hand-built score or posterior estimate and use thresholds to decide:
  verify now vs run critic vs regenerate
- h5_single_critic_then_map:
  run one cheap critic, choose the generator arm with the highest empirical success rate under that critic outcome, then verify

Tune heuristic thresholds only on validation split, never on test.

Phase 6: data splits and evaluation protocol
Use bug-level splits, not candidate-level splits.
Recommended:
- train/calibration split: fit likelihoods and transition models
- validation split: choose heuristic thresholds, cost scaling, optional ablations
- test split: final locked comparison

Rules
- No bug may appear in more than one split.
- Stratify by repository when possible.
- Replay the same candidate-bank artifacts for all policies.
- Use paired evaluation per bug.

Metrics to report
Always report raw metrics and utility.
Required:
- resolved rate
- mean utility
- mean verifier calls
- mean critic calls
- mean generator calls
- wall-clock seconds
- API dollar cost if available
- regression failure count
- per-repo breakdown
- per-policy confusion table on success/failure

Statistics
- paired bootstrap over bugs for mean utility difference and resolved-rate difference
- 95% confidence intervals
- save bootstrap samples or summaries to JSON
- include significance section in Markdown report

Phase 7: Allure integration
Use Allure only for experiment auditability.

For each episode/policy/bug:
- create steps:
  setup bug
  generate candidate
  run critic
  update posterior
  choose action
  verify
  summarize outcome
- attach:
  config yaml
  episode manifest json
  posterior timeline json
  chosen action sequence
  git diff
  failing test logs
  full verifier logs
  cost breakdown
  plot of posterior over time if available

Use suite hierarchy like:
- parent suite: benchmark name
- suite: policy name
- sub-suite: project/repository

Add a pytest-based report test that:
- runs a tiny smoke benchmark
- writes to allure-results
- proves attachments are present

Phase 8: article-ready outputs
Generate:
- results/synthetic/main_table.csv
- results/toy/main_table.csv
- results/bugsinpy/main_table.csv
- results/bugsinpy/per_bug_results.jsonl
- results/bugsinpy/bootstrap_summary.json
- results/bugsinpy/fig_utility.png
- results/bugsinpy/fig_success_rate.png
- results/bugsinpy/report.md

The report.md must include:
- setup
- benchmark description
- policies
- calibration method
- utility definition
- final tables
- bootstrap CIs
- top wins and losses for Bayes vs heuristics
- limitations
- notes on reproducibility

Optional Phase 9: Defects4J adapter
After BugsInPy is stable, add defects4j.py with the same adapter API.
Use:
- checkout
- compile
- test
and preserve timezone/config requirements in the adapter config.
Do not start here; do this only after BugsInPy passes smoke tests.

CLI requirements
Implement these commands:
- python -m abbo.cli run-synthetic --episodes 100000 --seed 7
- python -m abbo.cli run-synthetic-exact
- python -m abbo.cli calibrate-toy --trials 10000
- python -m abbo.cli run-toy --episodes 10000 --seed 7
- python -m abbo.cli build-candidate-bank --benchmark bugsinpy --config configs/bugsinpy_pilot.yaml
- python -m abbo.cli fit-bayes-model --benchmark bugsinpy --config configs/bugsinpy_pilot.yaml
- python -m abbo.cli run-benchmark --benchmark bugsinpy --policy bayes
- python -m abbo.cli run-benchmark --benchmark bugsinpy --policy all
- python -m abbo.cli summarize-results --benchmark bugsinpy
- pytest -q --alluredir 
-results --clean-alluredir

Testing requirements
Write tests for:
- exact posterior update
- exact DP values
- synthetic Monte Carlo close to exact
- toy empirical confusion matrix shape
- candidate manifest schema validation
- cost aggregation
- belief update monotonicity sanity checks
- Bayes policy never exceeds budget
- BugsInPy adapter smoke test using mocked subprocess if full benchmark unavailable

Acceptance criteria
Do not stop until all are true:
1. Synthetic exact numbers match the paper.
2. Monte Carlo synthetic run is stable and reproducible.
3. Toy benchmark works and preserves the Bayes advantage.
4. Candidate bank can be built and replayed without live model calls.
5. BugsInPy pilot runs end-to-end on a nontrivial bug set.
6. Bayes, fixed heuristics, and threshold heuristics all produce comparable result files.
7. Allure report contains diffs, logs, and posterior artifacts.
8. README contains exact setup and run commands.
9. Final report.md is article-ready.

README content requirements
Include:
- one-paragraph problem statement
- explanation that Bayes here means sequential decision-theoretic control
- environment setup
- how to run synthetic
- how to run toy
- how to build candidate bank
- how to fit models
- how to run evaluation
- how to open Allure report
- known limitations
- reproducibility notes

Implementation order
Checkpoint 1: scaffold repo, CLI, config system
Checkpoint 2: exact synthetic benchmark + tests
Checkpoint 3: toy benchmark + tests
Checkpoint 4: candidate-bank schema and build pipeline
Checkpoint 5: BugsInPy adapter and smoke run
Checkpoint 6: calibration + Bayes controller
Checkpoint 7: heuristic baselines + evaluation
Checkpoint 8: Allure + reports + README

Deliver code, tests, configs, and generated example results.
Do not leave placeholder TODOs in the final artifact.