# PRR report — ALFWorld, finished-only, pre-terminal

## Evaluation protocol

Sections 1–4 use 5-fold out-of-fold evaluation on the complete cohort: episode indices are shuffled with `numpy.random.RandomState(0)`, fold *i* is `order[i::5]` (not stratified); every episode is predicted exactly once by a model fitted on the other four folds, and each method's PRR@0.5 is computed once on the pooled prediction vector. Raw baselines, Verb final, tool success rate and −N fit nothing.

Cohort = ALFWorld valid-seen, 50-step budget, **finished-only** (the agent ended the episode itself: success, `final_answer`, or `give up`; budget exhaustion and external errors excluded — the analogue of Answer-only), and **pre-terminal**: every method sees the episode up to, not including, its last generation, because on a finished episode the last step reveals the outcome (a success ends on the goal-satisfying action, a failure on give-up / final_answer). Labels are binary success; PRR uses them (no partial scores exist).

UQ signals are read from the `combined` response segment of each generation: Logprob = `sum_logprob`, Perplexity, MTE = mean token entropy over the top-k alternatives, Self-certainty = −mean(top-k log-probs) − log k per token (lm-polygraph's definition), Verb actions = per-step verbalised confidence. Tool critics are the five per-step checks: format valid, action admissible, no repeated action, tool success, state changed. **Bayes tool-only** in Section 2 (the base of every UQ + tools row) is the tempered step-critic posterior — per-step log-likelihood ratios averaged over steps. Section 3 shows all three ways to read the critics: one episode-level observation (`critic:all`: all formats valid, all actions admissible, no repeated action — the original formulation), the per-step observations multiplied (summed log-LR), and the per-step observations tempered (averaged over steps).

### Cohorts and folds

- **ReAct · gpt-oss:** 99 finished episodes; successes 33/99. Per fold: train 79–80, test 19–20.
- **smol · gpt-oss:** 139 finished episodes; successes 43/139. Per fold: train 111–112, test 27–28.
- **ReAct · Qwen:** 75 finished episodes; successes 67/75. Per fold: train 60–60, test 15–15.
- **smol · Qwen:** 118 finished episodes; successes 104/118. Per fold: train 94–95, test 23–24.

## 1. UQ baselines — last, mean, max

Confidence is the aggregated raw value for logprob and Verb actions and the negative of the aggregated raw value for perplexity and MTE. No parameters are fitted.

### Logprob

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Logprob | last | -0.2464 | 0.5150 | 0.5238 | 0.5630 | **0.3388** |
| 2 | Logprob | mean | -0.2505 | 0.1746 | 0.4527 | 0.3251 | 0.1755 |
| 3 | Logprob | max | -0.4259 | -0.1325 | 0.2110 | 0.2277 | -0.0299 |

### Perplexity

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Perplexity | last | -0.0582 | 0.5878 | 0.1082 | 0.5651 | **0.3007** |
| 3 | Perplexity | mean | -0.2845 | 0.4199 | -0.1354 | 0.1058 | 0.0265 |
| 2 | Perplexity | max | 0.0734 | 0.7598 | 0.0532 | 0.1542 | 0.2601 |

### MTE

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | MTE | last | -0.0827 | 0.8314 | 0.1571 | 0.5674 | **0.3683** |
| 3 | MTE | mean | -0.3173 | 0.6747 | -0.1361 | 0.0832 | 0.0761 |
| 2 | MTE | max | 0.0219 | 0.7861 | 0.0175 | 0.1389 | 0.2411 |

### Self-certainty

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Self-certainty | last | n/a | n/a | n/a | n/a | **n/a** |
| 1 | Self-certainty | mean | n/a | n/a | n/a | n/a | n/a |
| 1 | Self-certainty | max | n/a | n/a | n/a | n/a | n/a |

### Verb actions

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 3 | Verb actions | last | -0.0517 | 0.8984 | -0.2053 | -0.0918 | 0.1374 |
| 1 | Verb actions | mean | 0.2836 | 0.7538 | -0.1650 | 0.0224 | **0.2237** |
| 2 | Verb actions | max | 0.1007 | 0.7126 | -0.1905 | -0.0507 | 0.1430 |

## 2. Bayesian UQ — UQ-only and UQ + tools

Five variants per signal. SEP, Double, Continuous and Tempered consume the full UQ sequence; Last only fits and applies one ContinuousBayes (λ=1) update on the last value per trajectory. UQ-only starts from the fitted prior; UQ + tools starts from the tempered posterior of the five step critics fitted on the same train fold.

### Logprob

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 6 | Logprob — Bayes UQ-only | SEP | 0.1836 | 0.4388 | 0.4261 | 0.0992 | 0.2869 |
| 5 | Logprob — Bayes UQ-only | Double | 0.1466 | 0.6165 | 0.4503 | 0.0767 | 0.3225 |
| 9 | Logprob — Bayes UQ-only | Continuous (λ=1) | -0.1868 | 0.3377 | -0.1457 | 0.0733 | 0.0196 |
| 10 | Logprob — Bayes UQ-only | Tempered (λ=0.25) | -0.1988 | 0.3144 | -0.1678 | 0.0637 | 0.0029 |
| 4 | Logprob — Bayes UQ-only | Last only | 0.2336 | 0.4715 | 0.4992 | 0.0873 | 0.3229 |
| 2 | Logprob — Bayes UQ + tools | SEP | 0.2730 | 0.6782 | 0.5106 | 0.2908 | 0.4382 |
| 3 | Logprob — Bayes UQ + tools | Double | 0.1589 | 0.7832 | 0.4765 | 0.2865 | 0.4263 |
| 8 | Logprob — Bayes UQ + tools | Continuous (λ=1) | -0.1525 | 0.5606 | -0.0241 | 0.0842 | 0.1171 |
| 7 | Logprob — Bayes UQ + tools | Tempered (λ=0.25) | -0.0929 | 0.7779 | 0.1276 | 0.0979 | 0.2276 |
| 1 | Logprob — Bayes UQ + tools | Last only | 0.4433 | 0.9565 | 0.5748 | 0.1473 | **0.5305** |

### Perplexity

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 | Perplexity — Bayes UQ-only | SEP | 0.1651 | 0.2728 | 0.1874 | 0.0168 | 0.1605 |
| 7 | Perplexity — Bayes UQ-only | Double | -0.0125 | 0.5369 | 0.4140 | 0.1039 | 0.2606 |
| 8 | Perplexity — Bayes UQ-only | Continuous (λ=1) | 0.0854 | 0.4235 | 0.1805 | 0.1045 | 0.1985 |
| 9 | Perplexity — Bayes UQ-only | Tempered (λ=0.25) | 0.0767 | 0.4418 | 0.1267 | 0.0772 | 0.1806 |
| 6 | Perplexity — Bayes UQ-only | Last only | -0.0362 | 0.5430 | 0.2055 | 0.5377 | 0.3125 |
| 4 | Perplexity — Bayes UQ + tools | SEP | 0.3587 | 0.6805 | 0.3660 | 0.2455 | 0.4127 |
| 3 | Perplexity — Bayes UQ + tools | Double | 0.1695 | 0.7252 | 0.5750 | 0.3856 | 0.4639 |
| 5 | Perplexity — Bayes UQ + tools | Continuous (λ=1) | 0.2827 | 0.6639 | 0.2415 | 0.3092 | 0.3743 |
| 2 | Perplexity — Bayes UQ + tools | Tempered (λ=0.25) | 0.3164 | 0.8452 | 0.3610 | 0.4498 | 0.4931 |
| 1 | Perplexity — Bayes UQ + tools | Last only | 0.2183 | 0.9865 | 0.4278 | 0.6548 | **0.5719** |

### MTE

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 | MTE — Bayes UQ-only | SEP | 0.2734 | 0.3528 | 0.0030 | 0.0443 | 0.1684 |
| 9 | MTE — Bayes UQ-only | Double | -0.0553 | 0.6647 | 0.2360 | 0.1049 | 0.2376 |
| 7 | MTE — Bayes UQ-only | Continuous (λ=1) | 0.2038 | 0.6389 | 0.2228 | 0.1088 | 0.2936 |
| 8 | MTE — Bayes UQ-only | Tempered (λ=0.25) | 0.1582 | 0.6196 | 0.2125 | 0.0439 | 0.2585 |
| 3 | MTE — Bayes UQ-only | Last only | 0.1419 | 0.7211 | 0.2863 | 0.5322 | 0.4204 |
| 6 | MTE — Bayes UQ + tools | SEP | 0.4171 | 0.7384 | 0.1887 | 0.2711 | 0.4038 |
| 4 | MTE — Bayes UQ + tools | Double | 0.0618 | 0.8300 | 0.3345 | 0.4338 | 0.4150 |
| 5 | MTE — Bayes UQ + tools | Continuous (λ=1) | 0.3078 | 0.7281 | 0.2534 | 0.3289 | 0.4046 |
| 2 | MTE — Bayes UQ + tools | Tempered (λ=0.25) | 0.4016 | 0.8548 | 0.4587 | 0.4284 | 0.5359 |
| 1 | MTE — Bayes UQ + tools | Last only | 0.2907 | 0.9932 | 0.4419 | 0.6411 | **0.5917** |

### Self-certainty

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 6 | Self-certainty — Bayes UQ-only | SEP | -0.2996 | -0.1667 | -0.0456 | -0.2165 | -0.1821 |
| 6 | Self-certainty — Bayes UQ-only | Double | -0.2996 | -0.1667 | -0.0456 | -0.2165 | -0.1821 |
| 6 | Self-certainty — Bayes UQ-only | Continuous (λ=1) | -0.2996 | -0.1667 | -0.0456 | -0.2165 | -0.1821 |
| 6 | Self-certainty — Bayes UQ-only | Tempered (λ=0.25) | -0.2996 | -0.1667 | -0.0456 | -0.2165 | -0.1821 |
| 6 | Self-certainty — Bayes UQ-only | Last only | -0.2996 | -0.1667 | -0.0456 | -0.2165 | -0.1821 |
| 1 | Self-certainty — Bayes UQ + tools | SEP | 0.3321 | 0.8819 | 0.3801 | 0.2675 | **0.4654** |
| 1 | Self-certainty — Bayes UQ + tools | Double | 0.3321 | 0.8819 | 0.3801 | 0.2675 | 0.4654 |
| 1 | Self-certainty — Bayes UQ + tools | Continuous (λ=1) | 0.3321 | 0.8819 | 0.3801 | 0.2675 | 0.4654 |
| 1 | Self-certainty — Bayes UQ + tools | Tempered (λ=0.25) | 0.3321 | 0.8819 | 0.3801 | 0.2675 | 0.4654 |
| 1 | Self-certainty — Bayes UQ + tools | Last only | 0.3321 | 0.8819 | 0.3801 | 0.2675 | 0.4654 |

### Verb actions

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 6 | Verb actions — Bayes UQ-only | SEP | -0.1589 | 0.7038 | 0.4787 | 0.1744 | 0.2995 |
| 10 | Verb actions — Bayes UQ-only | Double | -0.0734 | 0.6828 | -0.1787 | -0.0377 | 0.0983 |
| 8 | Verb actions — Bayes UQ-only | Continuous (λ=1) | -0.1525 | 0.5046 | 0.4619 | -0.1038 | 0.1776 |
| 9 | Verb actions — Bayes UQ-only | Tempered (λ=0.25) | -0.1350 | 0.4539 | 0.4707 | -0.0885 | 0.1753 |
| 7 | Verb actions — Bayes UQ-only | Last only | -0.2032 | 0.8166 | -0.0934 | 0.2472 | 0.1918 |
| 3 | Verb actions — Bayes UQ + tools | SEP | 0.0102 | 0.8428 | 0.4871 | 0.3454 | 0.4214 |
| 5 | Verb actions — Bayes UQ + tools | Double | -0.0067 | 0.8269 | 0.3118 | 0.2310 | 0.3408 |
| 4 | Verb actions — Bayes UQ + tools | Continuous (λ=1) | 0.0620 | 0.7600 | 0.4747 | 0.0666 | 0.3408 |
| 2 | Verb actions — Bayes UQ + tools | Tempered (λ=0.25) | 0.2222 | 0.7877 | 0.4872 | 0.1931 | 0.4226 |
| 1 | Verb actions — Bayes UQ + tools | Last only | 0.2575 | 0.9060 | 0.3491 | 0.4333 | **0.4865** |

## 3. Reference methods

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 3 | Verb final — Bayes UQ + tools | Last only | 0.2575 | 0.9060 | 0.3491 | 0.4333 | 0.4865 |
| 7 | Verb final | final | -0.0517 | 0.8984 | -0.2053 | -0.0918 | 0.1374 |
| 2 | Tool success rate | mean of step tool critics | 0.3429 | 0.8665 | 0.4671 | 0.4437 | 0.5301 |
| 5 | N steps | −N | 0.5612 | 0.7281 | 0.2336 | -0.0364 | 0.3716 |
| 6 | Bayes tool-only | episode critics (critic:all) | 0.4976 | 0.5481 | 0.0164 | 0.2181 | 0.3201 |
| 1 | Bayes tool-only | step critics, multiplied | 0.3861 | 0.9449 | 0.4663 | 0.3899 | **0.5468** |
| 4 | Bayes tool-only | step critics, tempered | 0.3321 | 0.8819 | 0.3801 | 0.2675 | 0.4654 |

Tool success rate is the mean over steps and critics of the five step-critic booleans. N steps uses −N (shorter ranks higher); N counts generations before the terminal step. Verb final is the last verbalised confidence before the terminal step.

## 4. Logistic regression — outer cross-fitting

The unchanged main implementation (`trajectory_uq_toolkit.regression`, agentic-uq main) fitted independently in each outer train fold: forward feature selection over every (signal, aggregation) column of the four signals plus the five step critics (share of passing steps), two internal hash splits, train-only preprocessing. `selected` searches the penalty; `pinned` fixes C=0.03. No length feature is supplied.

| Rank | Method | Aggregation | ReAct · gpt-oss | smol · gpt-oss | ReAct · Qwen | smol · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Logistic regression | selected | 0.1477 | 0.9790 | 0.1216 | 0.5501 | **0.4496** |
| 2 | Logistic regression | pinned | -0.1864 | 0.9932 | 0.1016 | 0.5649 | 0.3683 |

## 7. OOD Bayes fused sweep — five-split means

Each signal has eight fused variants (Quantile, SEP, LR+, LR−, Double, Continuous, Tempered, Last only) on top of the tempered tool posterior; Verb final adds a Last-only fused row. For each direction and seed 0–4, every parameter (prior, critic likelihoods, thresholds, Gaussians, regression) is fitted on a random half of the source cohort and scored on the whole target cohort. Cells are mean PRR ± sample SD over the five seeds; Avg/Rank are local to each direction pair.

### Directions 1–2 — Same model, different agent

| Rank | UQ | Mode | ReAct · gpt-oss → smol · gpt-oss | smol · gpt-oss → ReAct · gpt-oss | ReAct · Qwen → smol · Qwen | smol · Qwen → ReAct · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 14 | Logprob | Quantile | 0.7728 ± 0.0689 | 0.1320 ± 0.5453 | 0.2244 ± 0.1809 | 0.5090 ± 0.0380 | 0.4096 ± 0.2083 |
| 39 | Logprob | SEP | 0.4771 ± 0.5968 | -0.0294 ± 0.5311 | 0.0259 ± 0.0349 | 0.3347 ± 0.0272 | 0.2021 ± 0.2975 |
| 42 | Logprob | LR+ | -0.4126 ± 0.6899 | 0.4781 ± 0.0194 | 0.0926 ± 0.0712 | 0.3750 ± 0.0646 | 0.1333 ± 0.2113 |
| 29 | Logprob | LR− | -0.0935 ± 0.7946 | 0.5535 ± 0.0327 | 0.1249 ± 0.1097 | 0.5772 ± 0.0388 | 0.2905 ± 0.2440 |
| 40 | Logprob | Double | -0.4225 ± 0.6739 | 0.5559 ± 0.0378 | 0.1011 ± 0.0715 | 0.5015 ± 0.0183 | 0.1840 ± 0.2004 |
| 36 | Logprob | Continuous (λ=1) | 0.6615 ± 0.0462 | -0.0447 ± 0.1184 | 0.1444 ± 0.1147 | 0.1535 ± 0.3082 | 0.2287 ± 0.1469 |
| 25 | Logprob | Tempered (λ=0.25) | 0.7309 ± 0.0188 | 0.0293 ± 0.1647 | 0.3156 ± 0.1874 | 0.2406 ± 0.3831 | 0.3291 ± 0.1885 |
| 28 | Logprob | Last only | 0.6558 ± 0.1291 | -0.0686 ± 0.0543 | 0.3675 ± 0.2095 | 0.2211 ± 0.2827 | 0.2939 ± 0.1689 |
| 18 | Perplexity | Quantile | 0.5663 ± 0.1542 | 0.3241 ± 0.4496 | 0.2489 ± 0.2123 | 0.3462 ± 0.0367 | 0.3714 ± 0.2132 |
| 23 | Perplexity | SEP | 0.6007 ± 0.2788 | 0.1634 ± 0.4661 | 0.1110 ± 0.1353 | 0.4729 ± 0.0271 | 0.3370 ± 0.2268 |
| 34 | Perplexity | LR+ | -0.1913 ± 0.5802 | 0.4707 ± 0.0155 | 0.1436 ± 0.0705 | 0.5176 ± 0.0397 | 0.2352 ± 0.1765 |
| 33 | Perplexity | LR− | -0.0183 ± 0.5302 | 0.5275 ± 0.0411 | 0.1669 ± 0.2031 | 0.2702 ± 0.0803 | 0.2366 ± 0.2137 |
| 41 | Perplexity | Double | -0.3275 ± 0.4305 | 0.5437 ± 0.0289 | 0.1586 ± 0.1053 | 0.3170 ± 0.0657 | 0.1729 ± 0.1576 |
| 37 | Perplexity | Continuous (λ=1) | 0.1633 ± 0.6463 | 0.2643 ± 0.1412 | 0.1777 ± 0.0617 | 0.2718 ± 0.1219 | 0.2193 ± 0.2427 |
| 24 | Perplexity | Tempered (λ=0.25) | 0.4649 ± 0.6495 | 0.3691 ± 0.0690 | 0.1638 ± 0.0562 | 0.3268 ± 0.1210 | 0.3312 ± 0.2239 |
| 17 | Perplexity | Last only | 0.5646 ± 0.2013 | 0.0577 ± 0.0428 | 0.3906 ± 0.3086 | 0.4771 ± 0.4221 | 0.3725 ± 0.2437 |
| 11 | MTE | Quantile | 0.7624 ± 0.0898 | 0.5385 ± 0.0629 | 0.2424 ± 0.3123 | 0.3442 ± 0.0272 | 0.4719 ± 0.1231 |
| 16 | MTE | SEP | 0.4465 ± 0.5834 | 0.5262 ± 0.0539 | 0.1043 ± 0.0856 | 0.4636 ± 0.0372 | 0.3852 ± 0.1900 |
| 31 | MTE | LR+ | -0.0809 ± 0.7669 | 0.4649 ± 0.0220 | 0.1161 ± 0.0351 | 0.4832 ± 0.0385 | 0.2458 ± 0.2156 |
| 21 | MTE | LR− | 0.3979 ± 0.7345 | 0.5120 ± 0.0706 | 0.1348 ± 0.1459 | 0.3165 ± 0.0537 | 0.3403 ± 0.2512 |
| 35 | MTE | Double | -0.0943 ± 0.7448 | 0.5197 ± 0.0619 | 0.1709 ± 0.1556 | 0.3405 ± 0.0581 | 0.2342 ± 0.2551 |
| 44 | MTE | Continuous (λ=1) | -0.3399 ± 0.6729 | 0.2140 ± 0.1453 | 0.2757 ± 0.0481 | 0.2484 ± 0.0872 | 0.0996 ± 0.2384 |
| 38 | MTE | Tempered (λ=0.25) | -0.0792 ± 0.7364 | 0.4015 ± 0.0286 | 0.1433 ± 0.0422 | 0.3539 ± 0.0811 | 0.2049 ± 0.2221 |
| 32 | MTE | Last only | 0.1537 ± 0.3306 | -0.0199 ± 0.0190 | 0.3429 ± 0.2777 | 0.4819 ± 0.4431 | 0.2397 ± 0.2676 |
| 1 | Self-certainty | Quantile | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | **0.5458 ± 0.0810** |
| 1 | Self-certainty | SEP | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 1 | Self-certainty | LR+ | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 1 | Self-certainty | LR− | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 1 | Self-certainty | Double | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 1 | Self-certainty | Continuous (λ=1) | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 1 | Self-certainty | Tempered (λ=0.25) | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 1 | Self-certainty | Last only | 0.8599 ± 0.0282 | 0.3717 ± 0.0054 | 0.4189 ± 0.2562 | 0.5326 ± 0.0341 | 0.5458 ± 0.0810 |
| 12 | Verb actions | Quantile | 0.8459 ± 0.0259 | 0.4078 ± 0.0311 | 0.3653 ± 0.3134 | 0.2509 ± 0.1870 | 0.4675 ± 0.1394 |
| 22 | Verb actions | SEP | 0.8053 ± 0.1595 | 0.4865 ± 0.0209 | 0.0229 ± 0.0968 | 0.0425 ± 0.5144 | 0.3393 ± 0.1979 |
| 20 | Verb actions | LR+ | 0.8361 ± 0.1091 | 0.4926 ± 0.0191 | 0.3653 ± 0.3134 | -0.3011 ± 0.1147 | 0.3482 ± 0.1391 |
| 13 | Verb actions | LR− | 0.8236 ± 0.0090 | 0.2741 ± 0.1313 | 0.3653 ± 0.3134 | 0.2106 ± 0.1274 | 0.4184 ± 0.1453 |
| 26 | Verb actions | Double | 0.8253 ± 0.0150 | 0.4566 ± 0.0388 | 0.2733 ± 0.2885 | -0.3362 ± 0.1136 | 0.3047 ± 0.1140 |
| 19 | Verb actions | Continuous (λ=1) | 0.7243 ± 0.0541 | 0.4456 ± 0.0277 | 0.0833 ± 0.0735 | 0.2023 ± 0.5333 | 0.3639 ± 0.1722 |
| 15 | Verb actions | Tempered (λ=0.25) | 0.8137 ± 0.0394 | 0.4107 ± 0.0546 | 0.1006 ± 0.1212 | 0.3123 ± 0.3940 | 0.4093 ± 0.1523 |
| 9 | Verb actions | Last only | 0.9329 ± 0.0247 | 0.2039 ± 0.0171 | 0.2022 ± 0.0407 | 0.5973 ± 0.3241 | 0.4841 ± 0.1016 |
| 9 | Verb final | Last only | 0.9329 ± 0.0247 | 0.2039 ± 0.0171 | 0.2022 ± 0.0407 | 0.5973 ± 0.3241 | 0.4841 ± 0.1016 |
| 43 | Verb final | final | 0.8264 ± 0.0000 | -0.0517 ± 0.0000 | -0.0907 ± 0.0000 | -0.2053 ± 0.0000 | 0.1197 ± 0.0000 |
| 27 | Perplexity | last | 0.5878 ± 0.0000 | -0.0582 ± 0.0000 | 0.5651 ± 0.0000 | 0.1082 ± 0.0000 | 0.3007 ± 0.0000 |
| 30 | Logistic regression | pinned | 0.5276 ± 0.4358 | 0.1342 ± 0.0989 | 0.0592 ± 0.0000 | 0.2915 ± 0.5079 | 0.2531 ± 0.2606 |

### Directions 3–4 — Different model and agent

| Rank | UQ | Mode | ReAct · gpt-oss → smol · Qwen | smol · Qwen → ReAct · gpt-oss | ReAct · Qwen → smol · gpt-oss | smol · gpt-oss → ReAct · Qwen | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 41 | Logprob | Quantile | 0.1995 ± 0.2109 | -0.1078 ± 0.4133 | -0.4923 ± 0.3826 | 0.1783 ± 0.3323 | -0.0556 ± 0.3348 |
| 44 | Logprob | SEP | 0.0980 ± 0.1500 | -0.4137 ± 0.2821 | -0.7692 ± 0.0583 | 0.1018 ± 0.3906 | -0.2458 ± 0.2203 |
| 32 | Logprob | LR+ | 0.4698 ± 0.0188 | 0.0310 ± 0.3755 | -0.6176 ± 0.2211 | 0.5063 ± 0.0495 | 0.0974 ± 0.1662 |
| 40 | Logprob | LR− | 0.3874 ± 0.0696 | -0.1916 ± 0.1602 | -0.6816 ± 0.1769 | 0.3360 ± 0.0435 | -0.0374 ± 0.1126 |
| 36 | Logprob | Double | 0.4279 ± 0.0351 | -0.2194 ± 0.2209 | -0.4995 ± 0.2740 | 0.3336 ± 0.0286 | 0.0107 ± 0.1397 |
| 43 | Logprob | Continuous (λ=1) | 0.4566 ± 0.0895 | -0.2068 ± 0.4221 | -0.6999 ± 0.1515 | -0.2550 ± 0.0000 | -0.1763 ± 0.1658 |
| 42 | Logprob | Tempered (λ=0.25) | 0.3379 ± 0.1461 | -0.1455 ± 0.5247 | -0.3634 ± 0.4498 | -0.2550 ± 0.0000 | -0.1065 ± 0.2802 |
| 24 | Logprob | Last only | 0.3438 ± 0.0512 | 0.3967 ± 0.1157 | 0.4457 ± 0.3612 | -0.2550 ± 0.0000 | 0.2328 ± 0.1320 |
| 12 | Perplexity | Quantile | 0.2894 ± 0.1396 | 0.4806 ± 0.0117 | 0.2489 ± 0.5379 | 0.3151 ± 0.3376 | 0.3335 ± 0.2567 |
| 13 | Perplexity | SEP | 0.0288 ± 0.1023 | 0.5425 ± 0.0551 | 0.3286 ± 0.5987 | 0.3199 ± 0.4075 | 0.3049 ± 0.2909 |
| 16 | Perplexity | LR+ | 0.2153 ± 0.1430 | 0.5866 ± 0.0124 | -0.2388 ± 0.3656 | 0.5761 ± 0.0517 | 0.2848 ± 0.1432 |
| 27 | Perplexity | LR− | 0.2251 ± 0.1393 | 0.4723 ± 0.0461 | -0.2547 ± 0.4837 | 0.4469 ± 0.0787 | 0.2224 ± 0.1869 |
| 23 | Perplexity | Double | 0.0945 ± 0.0355 | 0.4736 ± 0.0455 | -0.0304 ± 0.3145 | 0.4428 ± 0.0353 | 0.2451 ± 0.1077 |
| 26 | Perplexity | Continuous (λ=1) | 0.1633 ± 0.2607 | 0.4106 ± 0.0971 | 0.1666 ± 0.0776 | 0.1576 ± 0.0324 | 0.2245 ± 0.1170 |
| 37 | Perplexity | Tempered (λ=0.25) | 0.2029 ± 0.2613 | 0.4163 ± 0.0970 | -0.8800 ± 0.0233 | 0.1976 ± 0.0074 | -0.0158 ± 0.0973 |
| 30 | Perplexity | Last only | -0.0859 ± 0.2894 | 0.2951 ± 0.1374 | 0.2726 ± 0.1540 | 0.2467 ± 0.0812 | 0.1821 ± 0.1655 |
| 11 | MTE | Quantile | 0.3285 ± 0.2161 | 0.4483 ± 0.0176 | 0.2845 ± 0.5879 | 0.4006 ± 0.0119 | 0.3655 ± 0.2084 |
| 25 | MTE | SEP | 0.0122 ± 0.0636 | 0.5380 ± 0.0595 | -0.1166 ± 0.7530 | 0.4665 ± 0.0842 | 0.2250 ± 0.2401 |
| 15 | MTE | LR+ | 0.2644 ± 0.1979 | 0.5879 ± 0.0047 | -0.2942 ± 0.2963 | 0.5990 ± 0.0511 | 0.2893 ± 0.1375 |
| 28 | MTE | LR− | 0.3423 ± 0.2250 | 0.4390 ± 0.0731 | -0.3541 ± 0.3701 | 0.4070 ± 0.0199 | 0.2085 ± 0.1720 |
| 18 | MTE | Double | 0.2412 ± 0.2037 | 0.4080 ± 0.0489 | -0.0420 ± 0.4460 | 0.4227 ± 0.0260 | 0.2575 ± 0.1811 |
| 19 | MTE | Continuous (λ=1) | 0.0796 ± 0.1509 | 0.4575 ± 0.0744 | 0.3338 ± 0.0975 | 0.1577 ± 0.0613 | 0.2572 ± 0.0960 |
| 35 | MTE | Tempered (λ=0.25) | 0.1704 ± 0.1737 | 0.4303 ± 0.0561 | -0.6909 ± 0.0924 | 0.2520 ± 0.0162 | 0.0404 ± 0.0846 |
| 29 | MTE | Last only | -0.2342 ± 0.0556 | 0.3748 ± 0.0989 | 0.4303 ± 0.2680 | 0.2046 ± 0.0968 | 0.1939 ± 0.1299 |
| 1 | Self-certainty | Quantile | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | **0.4547 ± 0.1914** |
| 1 | Self-certainty | SEP | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 1 | Self-certainty | LR+ | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 1 | Self-certainty | LR− | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 1 | Self-certainty | Double | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 1 | Self-certainty | Continuous (λ=1) | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 1 | Self-certainty | Tempered (λ=0.25) | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 1 | Self-certainty | Last only | 0.4557 ± 0.0179 | 0.5080 ± 0.0217 | 0.3985 ± 0.7028 | 0.4566 ± 0.0233 | 0.4547 ± 0.1914 |
| 9 | Verb actions | Quantile | 0.3586 ± 0.0369 | 0.4318 ± 0.1463 | 0.6475 ± 0.1938 | 0.1712 ± 0.1424 | 0.4023 ± 0.1299 |
| 38 | Verb actions | SEP | 0.1987 ± 0.1722 | 0.0546 ± 0.3548 | -0.1243 ± 0.2489 | -0.2130 ± 0.0160 | -0.0210 ± 0.1980 |
| 22 | Verb actions | LR+ | 0.2684 ± 0.1285 | 0.2875 ± 0.3678 | 0.6475 ± 0.1938 | -0.1950 ± 0.0102 | 0.2521 ± 0.1751 |
| 10 | Verb actions | LR− | 0.2864 ± 0.1805 | 0.4957 ± 0.0078 | 0.6475 ± 0.1938 | 0.0422 ± 0.1862 | 0.3680 ± 0.1421 |
| 17 | Verb actions | Double | 0.2184 ± 0.1515 | 0.4293 ± 0.1541 | 0.6502 ± 0.1814 | -0.1828 ± 0.0487 | 0.2788 ± 0.1339 |
| 39 | Verb actions | Continuous (λ=1) | 0.3410 ± 0.1126 | 0.3618 ± 0.2991 | -0.4976 ± 0.1403 | -0.3306 ± 0.0826 | -0.0314 ± 0.1587 |
| 34 | Verb actions | Tempered (λ=0.25) | 0.3999 ± 0.0699 | 0.3806 ± 0.2514 | -0.3265 ± 0.4126 | -0.2167 ± 0.1017 | 0.0593 ± 0.2089 |
| 20 | Verb actions | Last only | 0.3519 ± 0.1401 | 0.2794 ± 0.0652 | 0.0043 ± 0.0005 | 0.3791 ± 0.0468 | 0.2537 ± 0.0631 |
| 20 | Verb final | Last only | 0.3519 ± 0.1401 | 0.2794 ± 0.0652 | 0.0043 ± 0.0005 | 0.3791 ± 0.0468 | 0.2537 ± 0.0631 |
| 31 | Verb final | final | -0.0907 ± 0.0000 | -0.0517 ± 0.0000 | 0.8264 ± 0.0000 | -0.2053 ± 0.0000 | 0.1197 ± 0.0000 |
| 14 | Perplexity | last | 0.5651 ± 0.0000 | -0.0582 ± 0.0000 | 0.5878 ± 0.0000 | 0.1082 ± 0.0000 | 0.3007 ± 0.0000 |
| 33 | Logistic regression | pinned | 0.1812 ± 0.1727 | -0.0829 ± 0.0588 | 0.0554 ± 0.0000 | 0.1822 ± 0.3031 | 0.0840 ± 0.1337 |

### Directions 5–6 — Same agent, different model

| Rank | UQ | Mode | ReAct · gpt-oss → ReAct · Qwen | ReAct · Qwen → ReAct · gpt-oss | smol · gpt-oss → smol · Qwen | smol · Qwen → smol · gpt-oss | Avg |
|---:|---|---|---:|---:|---:|---:|---:|
| 40 | Logprob | Quantile | -0.1562 ± 0.3030 | -0.3869 ± 0.1678 | 0.1807 ± 0.0866 | 0.0445 ± 0.7357 | -0.0795 ± 0.3233 |
| 44 | Logprob | SEP | -0.1929 ± 0.3086 | -0.6016 ± 0.0272 | 0.2056 ± 0.1798 | -0.5532 ± 0.1530 | -0.2855 ± 0.1672 |
| 39 | Logprob | LR+ | 0.5150 ± 0.1041 | -0.5770 ± 0.0853 | 0.2771 ± 0.0247 | -0.4083 ± 0.3242 | -0.0483 ± 0.1346 |
| 38 | Logprob | LR− | 0.3978 ± 0.0699 | -0.5195 ± 0.0529 | 0.1663 ± 0.1003 | -0.0487 ± 0.4751 | -0.0010 ± 0.1745 |
| 42 | Logprob | Double | 0.3539 ± 0.0420 | -0.5413 ± 0.1460 | 0.1064 ± 0.0842 | -0.3769 ± 0.3597 | -0.1145 ± 0.1580 |
| 43 | Logprob | Continuous (λ=1) | -0.2474 ± 0.0118 | -0.5209 ± 0.0651 | 0.0592 ± 0.0000 | -0.0693 ± 0.5439 | -0.1946 ± 0.1552 |
| 41 | Logprob | Tempered (λ=0.25) | -0.1740 ± 0.1284 | -0.2070 ± 0.2487 | 0.0605 ± 0.0031 | -0.0349 ± 0.8441 | -0.0888 ± 0.3061 |
| 23 | Logprob | Last only | 0.2398 ± 0.0558 | 0.2417 ± 0.3110 | 0.0478 ± 0.0000 | 0.8681 ± 0.0592 | 0.3493 ± 0.1065 |
| 24 | Perplexity | Quantile | 0.2783 ± 0.2076 | 0.1264 ± 0.1750 | 0.4420 ± 0.1098 | 0.5488 ± 0.3053 | 0.3489 ± 0.1994 |
| 34 | Perplexity | SEP | 0.3668 ± 0.2378 | -0.0349 ± 0.3211 | 0.3424 ± 0.2058 | 0.1247 ± 0.1680 | 0.1998 ± 0.2332 |
| 17 | Perplexity | LR+ | 0.3246 ± 0.3990 | 0.3981 ± 0.0610 | 0.3632 ± 0.0486 | 0.5482 ± 0.1318 | 0.4085 ± 0.1601 |
| 21 | Perplexity | LR− | 0.1478 ± 0.2895 | 0.2971 ± 0.0446 | 0.4739 ± 0.0808 | 0.5418 ± 0.2683 | 0.3651 ± 0.1708 |
| 26 | Perplexity | Double | 0.0842 ± 0.3437 | 0.3176 ± 0.0494 | 0.4146 ± 0.0793 | 0.5112 ± 0.1160 | 0.3319 ± 0.1471 |
| 32 | Perplexity | Continuous (λ=1) | 0.2546 ± 0.2463 | -0.1715 ± 0.1301 | 0.3800 ± 0.0599 | 0.4147 ± 0.1893 | 0.2195 ± 0.1564 |
| 12 | Perplexity | Tempered (λ=0.25) | 0.4613 ± 0.1315 | -0.0317 ± 0.0960 | 0.5511 ± 0.0147 | 0.7299 ± 0.0798 | 0.4276 ± 0.0805 |
| 11 | Perplexity | Last only | 0.3700 ± 0.1534 | 0.2141 ± 0.1235 | 0.5575 ± 0.0073 | 0.6074 ± 0.4111 | 0.4373 ± 0.1738 |
| 14 | MTE | Quantile | 0.4018 ± 0.1995 | 0.2578 ± 0.1718 | 0.4600 ± 0.0233 | 0.5520 ± 0.3332 | 0.4179 ± 0.1819 |
| 27 | MTE | SEP | 0.3567 ± 0.1278 | 0.0857 ± 0.0897 | 0.5332 ± 0.0449 | 0.3003 ± 0.1699 | 0.3190 ± 0.1081 |
| 10 | MTE | LR+ | 0.4786 ± 0.2409 | 0.4450 ± 0.0821 | 0.4285 ± 0.0450 | 0.6375 ± 0.0350 | 0.4974 ± 0.1007 |
| 22 | MTE | LR− | 0.2936 ± 0.1579 | 0.1526 ± 0.1533 | 0.4453 ± 0.0383 | 0.5126 ± 0.3373 | 0.3510 ± 0.1717 |
| 15 | MTE | Double | 0.3026 ± 0.2147 | 0.2729 ± 0.0804 | 0.4484 ± 0.0491 | 0.6355 ± 0.0988 | 0.4149 ± 0.1107 |
| 18 | MTE | Continuous (λ=1) | 0.3201 ± 0.2254 | 0.2546 ± 0.1247 | 0.4006 ± 0.0619 | 0.5957 ± 0.1215 | 0.3928 ± 0.1334 |
| 9 | MTE | Tempered (λ=0.25) | 0.5259 ± 0.1627 | 0.3325 ± 0.0696 | 0.5569 ± 0.0260 | 0.8106 ± 0.0680 | 0.5565 ± 0.0816 |
| 16 | MTE | Last only | 0.2054 ± 0.1931 | 0.0624 ± 0.1623 | 0.5673 ± 0.0237 | 0.8059 ± 0.2650 | 0.4102 ± 0.1610 |
| 1 | Self-certainty | Quantile | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | **0.5665 ± 0.0947** |
| 1 | Self-certainty | SEP | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 1 | Self-certainty | LR+ | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 1 | Self-certainty | LR− | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 1 | Self-certainty | Double | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 1 | Self-certainty | Continuous (λ=1) | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 1 | Self-certainty | Tempered (λ=0.25) | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 1 | Self-certainty | Last only | 0.5793 ± 0.0756 | 0.3331 ± 0.2910 | 0.4496 ± 0.0028 | 0.9041 ± 0.0092 | 0.5665 ± 0.0947 |
| 19 | Verb actions | Quantile | -0.0085 ± 0.2506 | 0.5365 ± 0.0474 | 0.2869 ± 0.1475 | 0.7392 ± 0.1034 | 0.3885 ± 0.1372 |
| 37 | Verb actions | SEP | -0.0446 ± 0.3838 | -0.4187 ± 0.1561 | 0.1723 ± 0.0249 | 0.3980 ± 0.6090 | 0.0267 ± 0.2935 |
| 20 | Verb actions | LR+ | -0.1069 ± 0.1447 | 0.5365 ± 0.0474 | 0.1813 ± 0.0287 | 0.8504 ± 0.0507 | 0.3653 ± 0.0679 |
| 13 | Verb actions | LR− | 0.0510 ± 0.3358 | 0.5365 ± 0.0474 | 0.3064 ± 0.0732 | 0.7929 ± 0.0167 | 0.4217 ± 0.1183 |
| 25 | Verb actions | Double | -0.1672 ± 0.0331 | 0.5100 ± 0.0583 | 0.2386 ± 0.0458 | 0.7821 ± 0.0303 | 0.3409 ± 0.0419 |
| 33 | Verb actions | Continuous (λ=1) | 0.1444 ± 0.4151 | -0.2540 ± 0.1614 | 0.3351 ± 0.0141 | 0.5964 ± 0.3301 | 0.2055 ± 0.2302 |
| 31 | Verb actions | Tempered (λ=0.25) | 0.2512 ± 0.3439 | -0.3228 ± 0.0189 | 0.3788 ± 0.0167 | 0.7762 ± 0.1277 | 0.2708 ± 0.1268 |
| 28 | Verb actions | Last only | 0.6712 ± 0.0659 | -0.0977 ± 0.0000 | 0.0756 ± 0.0403 | 0.5666 ± 0.4700 | 0.3039 ± 0.1440 |
| 28 | Verb final | Last only | 0.6712 ± 0.0659 | -0.0977 ± 0.0000 | 0.0756 ± 0.0403 | 0.5666 ± 0.4700 | 0.3039 ± 0.1440 |
| 36 | Verb final | final | -0.2053 ± 0.0000 | -0.0517 ± 0.0000 | -0.0907 ± 0.0000 | 0.8264 ± 0.0000 | 0.1197 ± 0.0000 |
| 30 | Perplexity | last | 0.1082 ± 0.0000 | -0.0582 ± 0.0000 | 0.5651 ± 0.0000 | 0.5878 ± 0.0000 | 0.3007 ± 0.0000 |
| 35 | Logistic regression | pinned | 0.1301 ± 0.3971 | -0.0977 ± 0.0000 | 0.3432 ± 0.2307 | 0.3340 ± 0.3158 | 0.1774 ± 0.2359 |

### Overall summary — all methods

Overall Avg equally weights all direction means; SD is the mean of the per-cell SDs.

| Rank | Method | Aggregation | Avg 1–2 | Avg 3–4 | Avg 5–6 | Overall Avg |
|---:|---|---|---:|---:|---:|---:|
| 1 | Self-certainty — Bayes UQ + tools | Quantile | 0.5458 | 0.4547 | 0.5665 | **0.5223 ± 0.1224** |
| 2 | Self-certainty — Bayes UQ + tools | SEP | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 3 | Self-certainty — Bayes UQ + tools | LR+ | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 4 | Self-certainty — Bayes UQ + tools | LR− | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 5 | Self-certainty — Bayes UQ + tools | Double | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 6 | Self-certainty — Bayes UQ + tools | Continuous (λ=1) | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 7 | Self-certainty — Bayes UQ + tools | Tempered (λ=0.25) | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 8 | Self-certainty — Bayes UQ + tools | Last only | 0.5458 | 0.4547 | 0.5665 | 0.5223 ± 0.1224 |
| 9 | Verb actions — Bayes UQ + tools | Quantile | 0.4675 | 0.4023 | 0.3885 | 0.4194 ± 0.1355 |
| 10 | MTE — Bayes UQ + tools | Quantile | 0.4719 | 0.3655 | 0.4179 | 0.4184 ± 0.1711 |
| 11 | Verb actions — Bayes UQ + tools | LR− | 0.4184 | 0.3680 | 0.4217 | 0.4027 ± 0.1352 |
| 12 | Perplexity — Bayes UQ + tools | Quantile | 0.3714 | 0.3335 | 0.3489 | 0.3512 ± 0.2231 |
| 13 | Verb actions — Bayes UQ + tools | Last only | 0.4841 | 0.2537 | 0.3039 | 0.3472 ± 0.1029 |
| 14 | Verb final — Bayes UQ + tools | Last only | 0.4841 | 0.2537 | 0.3039 | 0.3472 ± 0.1029 |
| 15 | MTE — Bayes UQ + tools | LR+ | 0.2458 | 0.2893 | 0.4974 | 0.3442 ± 0.1513 |
| 16 | Perplexity — Bayes UQ + tools | Last only | 0.3725 | 0.1821 | 0.4373 | 0.3306 ± 0.1943 |
| 17 | Verb actions — Bayes UQ + tools | LR+ | 0.3482 | 0.2521 | 0.3653 | 0.3219 ± 0.1273 |
| 18 | MTE — Bayes UQ + tools | SEP | 0.3852 | 0.2250 | 0.3190 | 0.3097 ± 0.1794 |
| 19 | Perplexity — Bayes UQ + tools | LR+ | 0.2352 | 0.2848 | 0.4085 | 0.3095 ± 0.1599 |
| 20 | Verb actions — Bayes UQ + tools | Double | 0.3047 | 0.2788 | 0.3409 | 0.3081 ± 0.0966 |
| 21 | MTE — Bayes UQ + tools | Double | 0.2342 | 0.2575 | 0.4149 | 0.3022 ± 0.1823 |
| 22 | Perplexity | last | 0.3007 | 0.3007 | 0.3007 | 0.3007 ± 0.0000 |
| 23 | MTE — Bayes UQ + tools | LR− | 0.3403 | 0.2085 | 0.3510 | 0.2999 ± 0.1983 |
| 24 | Logprob — Bayes UQ + tools | Last only | 0.2939 | 0.2328 | 0.3493 | 0.2920 ± 0.1358 |
| 25 | MTE — Bayes UQ + tools | Last only | 0.2397 | 0.1939 | 0.4102 | 0.2813 ± 0.1862 |
| 26 | Perplexity — Bayes UQ + tools | SEP | 0.3370 | 0.3049 | 0.1998 | 0.2806 ± 0.2503 |
| 27 | Perplexity — Bayes UQ + tools | LR− | 0.2366 | 0.2224 | 0.3651 | 0.2747 ± 0.1905 |
| 28 | MTE — Bayes UQ + tools | Tempered (λ=0.25) | 0.2049 | 0.0404 | 0.5565 | 0.2673 ± 0.1294 |
| 29 | Perplexity — Bayes UQ + tools | Double | 0.1729 | 0.2451 | 0.3319 | 0.2500 ± 0.1375 |
| 30 | MTE — Bayes UQ + tools | Continuous (λ=1) | 0.0996 | 0.2572 | 0.3928 | 0.2498 ± 0.1559 |
| 31 | Perplexity — Bayes UQ + tools | Tempered (λ=0.25) | 0.3312 | -0.0158 | 0.4276 | 0.2477 ± 0.1339 |
| 32 | Verb actions — Bayes UQ + tools | Tempered (λ=0.25) | 0.4093 | 0.0593 | 0.2708 | 0.2465 ± 0.1627 |
| 33 | Perplexity — Bayes UQ + tools | Continuous (λ=1) | 0.2193 | 0.2245 | 0.2195 | 0.2211 ± 0.1720 |
| 34 | Verb actions — Bayes UQ + tools | Continuous (λ=1) | 0.3639 | -0.0314 | 0.2055 | 0.1793 ± 0.1870 |
| 35 | Logistic regression | pinned | 0.2531 | 0.0840 | 0.1774 | 0.1715 ± 0.2101 |
| 36 | Verb final | final | 0.1197 | 0.1197 | 0.1197 | 0.1197 ± 0.0000 |
| 37 | Verb actions — Bayes UQ + tools | SEP | 0.3393 | -0.0210 | 0.0267 | 0.1150 ± 0.2298 |
| 38 | Logprob — Bayes UQ + tools | Quantile | 0.4096 | -0.0556 | -0.0795 | 0.0915 ± 0.2888 |
| 39 | Logprob — Bayes UQ + tools | LR− | 0.2905 | -0.0374 | -0.0010 | 0.0840 ± 0.1770 |
| 40 | Logprob — Bayes UQ + tools | LR+ | 0.1333 | 0.0974 | -0.0483 | 0.0608 ± 0.1707 |
| 41 | Logprob — Bayes UQ + tools | Tempered (λ=0.25) | 0.3291 | -0.1065 | -0.0888 | 0.0446 ± 0.2583 |
| 42 | Logprob — Bayes UQ + tools | Double | 0.1840 | 0.0107 | -0.1145 | 0.0267 ± 0.1660 |
| 43 | Logprob — Bayes UQ + tools | Continuous (λ=1) | 0.2287 | -0.1763 | -0.1946 | -0.0474 ± 0.1559 |
| 44 | Logprob — Bayes UQ + tools | SEP | 0.2021 | -0.2458 | -0.2855 | -0.1097 ± 0.2283 |
