# PRR report — DeepSWE, submitted-only, pre-terminal

## Evaluation protocol

Same protocol as the ALFWorld report: Sections 1–4 use 5-fold out-of-fold evaluation (`numpy.random.RandomState(0)`, fold *i* = `order[i::5]`, one PRR on the pooled predictions); Section 7 averages OOD over five source-train → target-test holdouts (seeds 0–4, all parameters fitted on a random half of the source).

**Scores are continuous.** Neither model resolves a DeepSWE task outright (binary reward 0/113 for both), so PRR uses the verifier's native `partial` score (passed / all hidden tests, F2P + P2P) without binarisation, as the OSWorld/WebArena report does: the oracle ranks by the true score, the random reference is the mean score. Bayes and regression need a binary training label; every training fold is split at its own median score.

Cohort: 113 DeepSWE tasks, mini-swe-agent, 200-command budget, working-tree grading; **submitted-only** (the agent issued the submit command itself; context-window deaths and budget exhaustion excluded) and **pre-terminal** (scored before the submit command). Signals per command: Logprob = mean token log-probability of the generation, Perplexity = exp(−Logprob), MTE = mean token entropy (top-20). Verbalised confidence was requested in the system prompt and ignored by both models (Qwen 0/1806 commands, gpt-oss 49/6365), so no Verb rows. Tool critics per command: no format error, ran tests, committed, no command repeated ≥3×, and two episode-level test critics repeated per step: a test that failed then passed, last test run passed. Tool success rate = share of commands with return code 0. **Bayes tool-only** is the tempered critic posterior.

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
| 4 | Logprob — Bayes UQ-only | SEP | -0.0225 | 0.2828 | 0.1302 |
| 2 | Logprob — Bayes UQ-only | Double | 0.2375 | 0.1893 | 0.2134 |
| 5 | Logprob — Bayes UQ-only | Continuous (λ=1) | -0.0032 | 0.2147 | 0.1058 |
| 5 | Logprob — Bayes UQ-only | Tempered (λ=0.25) | -0.0032 | 0.2147 | 0.1058 |
| 9 | Logprob — Bayes UQ-only | Last only | -0.0106 | 0.1336 | 0.0615 |
| 3 | Logprob — Bayes UQ + tools | SEP | 0.0710 | 0.2257 | 0.1484 |
| 1 | Logprob — Bayes UQ + tools | Double | 0.2843 | 0.1936 | **0.2389** |
| 8 | Logprob — Bayes UQ + tools | Continuous (λ=1) | 0.0623 | 0.1265 | 0.0944 |
| 10 | Logprob — Bayes UQ + tools | Tempered (λ=0.25) | 0.0330 | 0.0719 | 0.0525 |
| 7 | Logprob — Bayes UQ + tools | Last only | 0.0968 | 0.0956 | 0.0962 |

### Perplexity

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 4 | Perplexity — Bayes UQ-only | SEP | -0.0216 | 0.2828 | 0.1306 |
| 2 | Perplexity — Bayes UQ-only | Double | 0.2375 | 0.1893 | 0.2134 |
| 5 | Perplexity — Bayes UQ-only | Continuous (λ=1) | 0.1128 | 0.1436 | 0.1282 |
| 5 | Perplexity — Bayes UQ-only | Tempered (λ=0.25) | 0.1128 | 0.1436 | 0.1282 |
| 9 | Perplexity — Bayes UQ-only | Last only | 0.0006 | 0.1354 | 0.0680 |
| 3 | Perplexity — Bayes UQ + tools | SEP | 0.0727 | 0.2257 | 0.1492 |
| 1 | Perplexity — Bayes UQ + tools | Double | 0.2843 | 0.1936 | **0.2389** |
| 7 | Perplexity — Bayes UQ + tools | Continuous (λ=1) | 0.1607 | 0.0878 | 0.1243 |
| 10 | Perplexity — Bayes UQ + tools | Tempered (λ=0.25) | 0.0786 | 0.0549 | 0.0667 |
| 8 | Perplexity — Bayes UQ + tools | Last only | 0.1442 | 0.0906 | 0.1174 |

### MTE

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 4 | MTE — Bayes UQ-only | SEP | 0.1274 | 0.1724 | 0.1499 |
| 2 | MTE — Bayes UQ-only | Double | 0.1598 | 0.2880 | 0.2239 |
| 7 | MTE — Bayes UQ-only | Continuous (λ=1) | -0.0626 | 0.1952 | 0.0663 |
| 7 | MTE — Bayes UQ-only | Tempered (λ=0.25) | -0.0626 | 0.1952 | 0.0663 |
| 10 | MTE — Bayes UQ-only | Last only | -0.1362 | 0.1285 | -0.0038 |
| 3 | MTE — Bayes UQ + tools | SEP | 0.1703 | 0.1757 | 0.1730 |
| 1 | MTE — Bayes UQ + tools | Double | 0.2094 | 0.2692 | **0.2393** |
| 6 | MTE — Bayes UQ + tools | Continuous (λ=1) | 0.0483 | 0.1075 | 0.0779 |
| 5 | MTE — Bayes UQ + tools | Tempered (λ=0.25) | 0.0847 | 0.0714 | 0.0780 |
| 9 | MTE — Bayes UQ + tools | Last only | -0.0345 | 0.0914 | 0.0285 |

## 3. Reference methods

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 3 | N steps | −N | 0.1613 | -0.1058 | 0.0278 |
| 2 | Bayes tool-only | step critics, tempered | 0.1325 | 0.0350 | 0.0837 |
| 1 | Bayes tool-only | step critics, multiplied | 0.2674 | 0.0068 | **0.1371** |
| 4 | Tool success rate | share of commands with return code 0 | -0.1834 | -0.1288 | -0.1561 |

N steps uses −N (shorter ranks higher); N counts commands before the submit.

## 4. Logistic regression — outer cross-fitting

The unchanged main implementation (`trajectory_uq_toolkit.regression`) fitted independently in each outer train fold on the three signals' aggregations and the six critics' pass shares; the outer training label is the fold's median split.

| Rank | Method | Aggregation | gpt-oss-20b | Qwen3.6-35B | Avg |
|---:|---|---|---:|---:|---:|
| 2 | Logistic regression | selected | 0.1195 | -0.0173 | 0.0511 |
| 1 | Logistic regression | pinned | 0.2847 | -0.0173 | **0.1337** |

## 7. OOD Bayes fused sweep — five-split means

Two directions: fit on a random half of one model's cohort, score the other model's whole cohort. Eight fused variants per signal on top of the tempered tool posterior. Cells are mean PRR ± sample SD over seeds 0–4.

### Directions 1–2 — Same agent, different model

| Rank | UQ | Mode | gpt-oss-20b → Qwen3.6-35B | Qwen3.6-35B → gpt-oss-20b | Avg |
|---:|---|---|---:|---:|---:|
| 12 | Logprob | Quantile | 0.1892 ± 0.0389 | 0.2164 ± 0.1742 | 0.2028 ± 0.1065 |
| 5 | Logprob | SEP | 0.1833 ± 0.0508 | 0.3109 ± 0.0630 | 0.2471 ± 0.0569 |
| 1 | Logprob | LR+ | 0.2210 ± 0.0324 | 0.3790 ± 0.0424 | **0.3000 ± 0.0374** |
| 7 | Logprob | LR− | 0.2255 ± 0.0329 | 0.2629 ± 0.0625 | 0.2442 ± 0.0477 |
| 16 | Logprob | Double | 0.1257 ± 0.1675 | 0.2577 ± 0.0615 | 0.1917 ± 0.1145 |
| 24 | Logprob | Continuous (λ=1) | 0.1907 ± 0.0376 | -0.0618 ± 0.0227 | 0.0645 ± 0.0301 |
| 14 | Logprob | Tempered (λ=0.25) | 0.1948 ± 0.0328 | 0.1892 ± 0.0905 | 0.1920 ± 0.0616 |
| 19 | Logprob | Last only | 0.2439 ± 0.0165 | 0.1164 ± 0.1070 | 0.1801 ± 0.0618 |
| 12 | Perplexity | Quantile | 0.1892 ± 0.0389 | 0.2164 ± 0.1742 | 0.2028 ± 0.1065 |
| 5 | Perplexity | SEP | 0.1833 ± 0.0508 | 0.3109 ± 0.0630 | 0.2471 ± 0.0569 |
| 1 | Perplexity | LR+ | 0.2210 ± 0.0324 | 0.3790 ± 0.0424 | 0.3000 ± 0.0374 |
| 7 | Perplexity | LR− | 0.2255 ± 0.0329 | 0.2629 ± 0.0625 | 0.2442 ± 0.0477 |
| 16 | Perplexity | Double | 0.1257 ± 0.1675 | 0.2577 ± 0.0615 | 0.1917 ± 0.1145 |
| 22 | Perplexity | Continuous (λ=1) | 0.1745 ± 0.0496 | -0.0009 ± 0.1812 | 0.0868 ± 0.1154 |
| 21 | Perplexity | Tempered (λ=0.25) | 0.1839 ± 0.0298 | 0.0212 ± 0.2237 | 0.1025 ± 0.1267 |
| 18 | Perplexity | Last only | 0.2472 ± 0.0107 | 0.1264 ± 0.1157 | 0.1868 ± 0.0632 |
| 11 | MTE | Quantile | 0.2045 ± 0.0400 | 0.2096 ± 0.1810 | 0.2071 ± 0.1105 |
| 9 | MTE | SEP | 0.1780 ± 0.0609 | 0.3055 ± 0.0538 | 0.2417 ± 0.0574 |
| 3 | MTE | LR+ | 0.1930 ± 0.0546 | 0.3698 ± 0.0724 | 0.2814 ± 0.0635 |
| 4 | MTE | LR− | 0.2351 ± 0.0058 | 0.2779 ± 0.0800 | 0.2565 ± 0.0429 |
| 10 | MTE | Double | 0.1601 ± 0.0638 | 0.2744 ± 0.0777 | 0.2172 ± 0.0708 |
| 25 | MTE | Continuous (λ=1) | 0.1956 ± 0.0359 | -0.0990 ± 0.0538 | 0.0483 ± 0.0449 |
| 15 | MTE | Tempered (λ=0.25) | 0.1945 ± 0.0293 | 0.1894 ± 0.1510 | 0.1920 ± 0.0901 |
| 20 | MTE | Last only | 0.2297 ± 0.0161 | -0.0074 ± 0.1211 | 0.1111 ± 0.0686 |
| 26 | Perplexity | last | -0.0851 ± 0.0000 | -0.0633 ± 0.0000 | -0.0742 ± 0.0000 |
| 23 | Logistic regression | pinned | 0.0700 ± 0.1274 | 0.0837 ± 0.0567 | 0.0768 ± 0.0921 |
