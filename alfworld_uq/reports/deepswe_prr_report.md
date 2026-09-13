# PRR report — DeepSWE, submitted-only, pre-terminal

## Evaluation protocol

Same protocol as the ALFWorld report: Sections 1–4 use 5-fold out-of-fold evaluation (`numpy.random.RandomState(0)`, fold *i* = `order[i::5]`, one PRR on the pooled predictions); Section 7 averages OOD over five source-train → target-test holdouts (seeds 0–4, all parameters fitted on a random half of the source).

**Scores are continuous.** Neither model resolves a DeepSWE task outright (binary reward 0/113 for both), so PRR uses the verifier's native `partial` score (passed / all hidden tests, F2P + P2P) without binarisation, as the OSWorld/WebArena report does: the oracle ranks by the true score, the random reference is the mean score. Bayes and regression need a binary training label; every training fold is split at its own median score.

Cohort: 113 DeepSWE tasks, mini-swe-agent, 200-command budget, working-tree grading; **submitted-only** (the agent issued the submit command itself; context-window deaths and budget exhaustion excluded) and **pre-terminal** (scored before the submit command). Signals per command: Logprob = mean token log-probability of the generation, Perplexity = exp(−Logprob), MTE = mean token entropy (top-20). Verbalised confidence was requested in the system prompt and ignored by both models (Qwen 0/1806 commands, gpt-oss 49/6365), so no Verb rows. Tool critics are read from the agent's own test runs and commits: ran the test suite (`pytest`, `go test`, `npm test`, …), ran it at least twice, last run passed, a run that failed then passed, committed at least twice, no format errors. Our first critic set (submitted, committed, no repeated command, and "ran tests" matched by the substring `test`, which counts `ls tests/`) was saturated at ~100% in both classes and carried no signal; this is reported in the paper as a negative finding about critic design. What these critics can see: the agent runs the repository's existing suite, while the verifier's F2P tests are new, so they measure "kept the repository working" (P2P) rather than "built the feature". Tool success rate = share of commands with return code 0. **Bayes tool-only** is the critic posterior.

**Two things that look like copy-paste errors and are not.** (1) Perplexity rows repeat the Logprob rows wherever the method is rank-based (raw `last`, and every binarised Bayes variant): on DeepSWE Perplexity is exp(−Logprob) per command, a monotone transform, and PRR only sees ranks; they differ only where the value enters a Gaussian (Continuous / Tempered / Last only) or a mean over steps. (2) UQ-only Continuous (λ=1) and Tempered (λ=0.25) coincide: λ rescales the summed evidence, which does not change the ranking; they separate only once fused with the tool posterior.

**Headline.** All values are low (best Avg ≈ .24 in-domain, ≈ .30 OOD, against .5–.9 on ALFWorld): the label is partial credit rather than success, the outcome is largely a task property, and the environment offers no progress signal (process critics saturate, failing test runs are how work is done, so the tool success rate is *negative*). Within that, Bayes UQ + tools (Double) is top-1 for every signal in Section 2, the tempered/multiplied tool-only posterior beats the regression and every raw baseline in Sections 3–4, and the fused binary variants (LR+ / SEP) transfer between the two models with a drop of ≈ .05.

### Cohorts and folds

- **gpt-oss-20b:** 90 submitted episodes; mean partial score 0.487; median 0.548. Per fold: train 72–72, test 18–18.
- **Qwen3.6-35B:** 65 submitted episodes; mean partial score 0.784; median 0.902. Per fold: train 52–52, test 13–13.

## 1. UQ baselines — last, mean, max

Confidence is the aggregated raw value for Logprob and the negative of the aggregated raw value for Perplexity and MTE. No parameters are fitted.

### Logprob

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 3 | Logprob | last | -0.0633 | -0.0851 | -0.0742 |
| 1 | Logprob | mean | -0.0333 | 0.2443 | **0.1055** |
| 2 | Logprob | max | 0.0781 | -0.2154 | -0.0687 |

### Perplexity

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 3 | Perplexity | last | -0.0633 | -0.0851 | -0.0742 |
| 1 | Perplexity | mean | -0.0581 | 0.2497 | **0.0958** |
| 2 | Perplexity | max | -0.0844 | 0.0156 | -0.0344 |

### MTE

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 2 | MTE | last | -0.0648 | 0.1465 | 0.0408 |
| 1 | MTE | mean | -0.0204 | 0.2136 | **0.0966** |
| 3 | MTE | max | -0.0757 | 0.1184 | 0.0213 |

## 2. Bayesian UQ — UQ-only and UQ + tools

Five variants per signal, as in the ALFWorld report. UQ + tools starts from the tempered posterior of the six critics fitted on the same train fold.

### Logprob

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 6 | Logprob — Bayes UQ-only | SEP | -0.0225 | 0.2828 | 0.1302 |
| 2 | Logprob — Bayes UQ-only | Double | 0.2375 | 0.1893 | 0.2134 |
| 8 | Logprob — Bayes UQ-only | Continuous (λ=1) | -0.0032 | 0.2147 | 0.1058 |
| 8 | Logprob — Bayes UQ-only | Tempered (λ=0.25) | -0.0032 | 0.2147 | 0.1058 |
| 10 | Logprob — Bayes UQ-only | Last only | -0.0106 | 0.1336 | 0.0615 |
| 3 | Logprob — Bayes UQ + tools | SEP | 0.0925 | 0.3020 | 0.1973 |
| 1 | Logprob — Bayes UQ + tools | Double | 0.2290 | 0.2659 | **0.2475** |
| 5 | Logprob — Bayes UQ + tools | Continuous (λ=1) | 0.0891 | 0.1902 | 0.1397 |
| 7 | Logprob — Bayes UQ + tools | Tempered (λ=0.25) | 0.0722 | 0.1843 | 0.1283 |
| 4 | Logprob — Bayes UQ + tools | Last only | 0.1016 | 0.2147 | 0.1581 |

### Perplexity

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 6 | Perplexity — Bayes UQ-only | SEP | -0.0216 | 0.2828 | 0.1306 |
| 2 | Perplexity — Bayes UQ-only | Double | 0.2375 | 0.1893 | 0.2134 |
| 7 | Perplexity — Bayes UQ-only | Continuous (λ=1) | 0.1128 | 0.1436 | 0.1282 |
| 7 | Perplexity — Bayes UQ-only | Tempered (λ=0.25) | 0.1128 | 0.1436 | 0.1282 |
| 10 | Perplexity — Bayes UQ-only | Last only | 0.0006 | 0.1354 | 0.0680 |
| 3 | Perplexity — Bayes UQ + tools | SEP | 0.0908 | 0.3020 | 0.1964 |
| 1 | Perplexity — Bayes UQ + tools | Double | 0.2290 | 0.2659 | **0.2475** |
| 5 | Perplexity — Bayes UQ + tools | Continuous (λ=1) | 0.1474 | 0.1651 | 0.1563 |
| 9 | Perplexity — Bayes UQ + tools | Tempered (λ=0.25) | 0.0801 | 0.1673 | 0.1237 |
| 4 | Perplexity — Bayes UQ + tools | Last only | 0.1143 | 0.2210 | 0.1677 |

### MTE

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 5 | MTE — Bayes UQ-only | SEP | 0.1274 | 0.1724 | 0.1499 |
| 3 | MTE — Bayes UQ-only | Double | 0.1598 | 0.2880 | 0.2239 |
| 8 | MTE — Bayes UQ-only | Continuous (λ=1) | -0.0626 | 0.1952 | 0.0663 |
| 8 | MTE — Bayes UQ-only | Tempered (λ=0.25) | -0.0626 | 0.1952 | 0.0663 |
| 10 | MTE — Bayes UQ-only | Last only | -0.1362 | 0.1285 | -0.0038 |
| 2 | MTE — Bayes UQ + tools | SEP | 0.2161 | 0.2671 | 0.2416 |
| 1 | MTE — Bayes UQ + tools | Double | 0.2254 | 0.3095 | **0.2675** |
| 4 | MTE — Bayes UQ + tools | Continuous (λ=1) | 0.1189 | 0.1966 | 0.1577 |
| 7 | MTE — Bayes UQ + tools | Tempered (λ=0.25) | 0.0974 | 0.1810 | 0.1392 |
| 6 | MTE — Bayes UQ + tools | Last only | 0.0769 | 0.2119 | 0.1444 |

## 3. Reference methods

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 2 | N steps | −N | 0.1613 | -0.1058 | 0.0278 |
| 3 | Tool success rate | share of commands with return code 0 | -0.1834 | -0.1288 | -0.1561 |
| 1 | Bayes tool-only | episode critics | 0.0974 | 0.1717 | **0.1346** |

N steps uses −N (shorter ranks higher); N counts commands before the submit.

## 4. Logistic regression — outer cross-fitting

The unchanged main implementation (`trajectory_uq_toolkit.regression`) fitted independently in each outer train fold on the three signals' aggregations and the six critics' pass shares; the outer training label is the fold's median split.

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 1 | Logistic regression | selected | 0.0726 | -0.0173 | **0.0277** |
| 2 | Logistic regression | pinned | 0.0570 | -0.0173 | 0.0199 |

## 7. OOD Bayes fused sweep — five-split means

Two directions: fit on a random half of one model's cohort, score the other model's whole cohort. Eight fused variants per signal on top of the tempered tool posterior. Cells are mean PRR ± sample SD over seeds 0–4.

### Directions 1–2 — Same agent, different model

| Rank | UQ | Mode | gpt-oss-20b → Qwen3.6-35B | Qwen3.6-35B → gpt-oss-20b | Avg |
|---:|---|---|---:|---:|---:|
| 12 | Logprob | Quantile | 0.2160 ± 0.0186 | 0.3037 ± 0.1443 | 0.2598 ± 0.0814 |
| 5 | Logprob | SEP | 0.2076 ± 0.0628 | 0.4017 ± 0.1248 | 0.3046 ± 0.0938 |
| 2 | Logprob | LR+ | 0.2612 ± 0.0254 | 0.4634 ± 0.0779 | 0.3623 ± 0.0517 |
| 8 | Logprob | LR− | 0.2392 ± 0.0110 | 0.3456 ± 0.0856 | 0.2924 ± 0.0483 |
| 16 | Logprob | Double | 0.1628 ± 0.1713 | 0.3378 ± 0.0987 | 0.2503 ± 0.1350 |
| 24 | Logprob | Continuous (λ=1) | 0.2122 ± 0.0346 | -0.0321 ± 0.0801 | 0.0901 ± 0.0573 |
| 19 | Logprob | Tempered (λ=0.25) | 0.2182 ± 0.0187 | 0.2116 ± 0.0677 | 0.2149 ± 0.0432 |
| 15 | Logprob | Last only | 0.2332 ± 0.0129 | 0.2701 ± 0.1144 | 0.2516 ± 0.0636 |
| 12 | Perplexity | Quantile | 0.2160 ± 0.0186 | 0.3037 ± 0.1443 | 0.2598 ± 0.0814 |
| 5 | Perplexity | SEP | 0.2076 ± 0.0628 | 0.4017 ± 0.1248 | 0.3046 ± 0.0938 |
| 2 | Perplexity | LR+ | 0.2612 ± 0.0254 | 0.4634 ± 0.0779 | 0.3623 ± 0.0517 |
| 8 | Perplexity | LR− | 0.2392 ± 0.0110 | 0.3456 ± 0.0856 | 0.2924 ± 0.0483 |
| 16 | Perplexity | Double | 0.1628 ± 0.1713 | 0.3378 ± 0.0987 | 0.2503 ± 0.1350 |
| 22 | Perplexity | Continuous (λ=1) | 0.2032 ± 0.0378 | 0.0325 ± 0.2558 | 0.1178 ± 0.1468 |
| 21 | Perplexity | Tempered (λ=0.25) | 0.2123 ± 0.0182 | 0.0381 ± 0.2482 | 0.1252 ± 0.1332 |
| 14 | Perplexity | Last only | 0.2404 ± 0.0144 | 0.2646 ± 0.1157 | 0.2525 ± 0.0650 |
| 11 | MTE | Quantile | 0.2231 ± 0.0198 | 0.3178 ± 0.1787 | 0.2705 ± 0.0993 |
| 10 | MTE | SEP | 0.2174 ± 0.0622 | 0.3637 ± 0.0952 | 0.2905 ± 0.0787 |
| 1 | MTE | LR+ | 0.2673 ± 0.0348 | 0.4743 ± 0.0709 | **0.3708 ± 0.0529** |
| 4 | MTE | LR− | 0.2556 ± 0.0384 | 0.3537 ± 0.0895 | 0.3047 ± 0.0640 |
| 7 | MTE | Double | 0.2591 ± 0.0385 | 0.3326 ± 0.0812 | 0.2959 ± 0.0598 |
| 25 | MTE | Continuous (λ=1) | 0.2037 ± 0.0402 | -0.0985 ± 0.0640 | 0.0526 ± 0.0521 |
| 18 | MTE | Tempered (λ=0.25) | 0.2166 ± 0.0159 | 0.2455 ± 0.1168 | 0.2310 ± 0.0664 |
| 20 | MTE | Last only | 0.2103 ± 0.0084 | 0.1417 ± 0.1006 | 0.1760 ± 0.0545 |
| 26 | Perplexity | last | -0.0851 ± 0.0000 | -0.0633 ± 0.0000 | -0.0742 ± 0.0000 |
| 23 | Logistic regression | pinned | 0.1095 ± 0.1324 | 0.0837 ± 0.0567 | 0.0966 ± 0.0946 |
