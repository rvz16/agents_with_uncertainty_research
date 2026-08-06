# ALFWorld trajectory UQ report

- Episodes: 100 (50 calibration, 50 test)
- Successes: 37/100
- Steps with token logprobs: 2306/2365
- Stop reasons: {'max_steps': 63, 'success': 37}
- Fallbacks: {'inadmissible_action': 223, 'none': 2075, 'repeated_action': 67}
- Providers: {'Novita': 2365}
- Total tokens: 3352522
- Median episode time: 105.7s
- P95 episode time: 324.5s
- Available target/method pairs: 14
- Metric rows: 387

## Critic Bayes state

Episode state applies each summarized critic once; stepwise_uq_exps applies all critics after every generation.

| State model | Critic | P(pass | success) | P(pass | failure) | Informativeness |
|---|---|---:|---:|---:|
| episode | all_formats_valid | 0.7368 | 0.6000 | +0.1368 |
| episode | all_actions_valid | 0.5263 | 0.2571 | +0.2692 |
| episode | no_repeated_fallback | 0.9474 | 0.6286 | +0.3188 |
| stepwise_uq_exps | format_valid | 0.9429 | 0.9708 | -0.0279 |
| stepwise_uq_exps | action_valid | 0.8762 | 0.9153 | -0.0391 |
| stepwise_uq_exps | no_repeated_fallback | 0.9952 | 0.9587 | +0.0366 |

## Combined perplexity

| Model | AUROC | AUPRC | PRR@0.5 | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| bayes_state | 0.8225 | 0.9012 | 0.6744 | 0.1632 | 0.4968 | 0.0880 |
| bayes_state_plus_binary | 0.8350 | 0.7547 | 0.7207 | 0.1651 | 0.4977 | 0.1123 |
| bayes_state_plus_continuous | 0.7783 | 0.6802 | 0.5938 | 0.1887 | 0.5548 | 0.1729 |
| bayes_state_plus_double | 0.6900 | 0.5373 | 0.4801 | 0.2462 | 0.7216 | 0.1989 |
| bayes_state_plus_lr_neg | 0.7467 | 0.6045 | 0.6639 | 0.1981 | 0.5735 | 0.1232 |
| bayes_state_plus_lr_pos | 0.7817 | 0.6644 | 0.5781 | 0.1897 | 0.5646 | 0.1329 |
| bayes_state_plus_sep | 0.7650 | 0.6576 | 0.5293 | 0.1951 | 0.5774 | 0.1692 |
| bayes_state_plus_tempered | 0.8083 | 0.7071 | 0.6644 | 0.1673 | 0.5055 | 0.1283 |
| binary_bayes | 0.4800 | 0.3769 | 0.1818 | 0.2467 | 0.6874 | 0.0987 |
| binary_bayes_double | 0.4817 | 0.3669 | 0.3435 | 0.3517 | 0.9809 | 0.3635 |
| binary_bayes_lr_neg | 0.4867 | 0.3714 | 0.2503 | 0.2922 | 0.7788 | 0.3322 |
| binary_bayes_lr_pos | 0.5333 | 0.3935 | 0.3737 | 0.2901 | 0.7879 | 0.2920 |
| binary_bayes_sep | 0.5067 | 0.3820 | 0.2748 | 0.3050 | 0.8215 | 0.3120 |
| continuous_bayes | 0.4150 | 0.3421 | 0.1096 | 0.2782 | 0.7563 | 0.2852 |
| feature_last | 0.5183 | 0.4173 | 0.1339 | 0.2441 | 0.6819 | 0.0639 |
| feature_max | 0.6867 | 0.7108 | 0.2100 | 0.2289 | 0.6645 | 0.1436 |
| feature_mean | 0.4383 | 0.3858 | -0.1471 | 0.2449 | 0.6837 | 0.0607 |
| stepwise_bayes_state | 0.5833 | 0.4234 | 0.5691 | 0.3418 | 1.1142 | 0.3296 |
| stepwise_bayes_state_plus_binary | 0.5817 | 0.4258 | 0.4821 | 0.3439 | 1.1148 | 0.3534 |
| stepwise_bayes_state_plus_continuous | 0.5433 | 0.4038 | 0.4124 | 0.3609 | 1.1569 | 0.3973 |
| stepwise_bayes_state_plus_double | 0.5267 | 0.3888 | 0.4685 | 0.3874 | 1.2586 | 0.4022 |
| stepwise_bayes_state_plus_lr_neg | 0.5167 | 0.3884 | 0.4421 | 0.3752 | 1.1860 | 0.4059 |
| stepwise_bayes_state_plus_lr_pos | 0.5600 | 0.4117 | 0.4663 | 0.3500 | 1.1323 | 0.3633 |
| stepwise_bayes_state_plus_sep | 0.5567 | 0.4100 | 0.4645 | 0.3586 | 1.1727 | 0.3446 |
| stepwise_bayes_state_plus_tempered | 0.5933 | 0.4315 | 0.5639 | 0.3458 | 1.1206 | 0.3671 |
| tempered_continuous_bayes | 0.4150 | 0.3421 | 0.1096 | 0.2487 | 0.6919 | 0.1295 |

These pilot metrics use only 50 test episodes with 20 positive outcome(s). They validate the pipeline but are not stable performance estimates.

## Bayes state + thought sum_logprob

| Model | AUROC | Delta AUROC | PRR@0.5 | Delta PRR@0.5 | Brier | Delta Brier | NLL | Delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bayes_state | 0.8225 | +0.0000 | 0.6744 | +0.0000 | 0.1632 | +0.0000 | 0.4968 | +0.0000 |
| bayes_state_plus_binary | 0.9217 | +0.0992 | 0.8374 | +0.1631 | 0.1233 | -0.0400 | 0.4068 | -0.0900 |
| bayes_state_plus_continuous | 0.9067 | +0.0842 | 0.9331 | +0.2587 | 0.1721 | +0.0089 | 0.6468 | +0.1500 |
| bayes_state_plus_double | 0.7500 | -0.0725 | 0.5203 | -0.1541 | 0.1994 | +0.0362 | 0.5898 | +0.0929 |
| bayes_state_plus_lr_neg | 0.7700 | -0.0525 | 0.5564 | -0.1180 | 0.1801 | +0.0168 | 0.5400 | +0.0432 |
| bayes_state_plus_lr_pos | 0.7917 | -0.0308 | 0.6274 | -0.0470 | 0.1775 | +0.0143 | 0.5339 | +0.0370 |
| bayes_state_plus_sep | 0.9400 | +0.1175 | 0.9792 | +0.3048 | 0.1084 | -0.0549 | 0.3617 | -0.1351 |
| bayes_state_plus_tempered | 0.9567 | +0.1342 | 0.9693 | +0.2949 | 0.1135 | -0.0498 | 0.3701 | -0.1267 |

## Stepwise uq_exps-style state + thought sum_logprob

| Model | AUROC | Delta AUROC | PRR@0.5 | Delta PRR@0.5 | Brier | Delta Brier | NLL | Delta NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stepwise_bayes_state | 0.5833 | +0.0000 | 0.5691 | +0.0000 | 0.3418 | +0.0000 | 1.1142 | +0.0000 |
| stepwise_bayes_state_plus_binary | 0.6433 | +0.0600 | 0.6866 | +0.1175 | 0.3256 | -0.0161 | 1.0690 | -0.0451 |
| stepwise_bayes_state_plus_continuous | 0.8150 | +0.2317 | 0.8522 | +0.2831 | 0.2779 | -0.0638 | 1.1149 | +0.0007 |
| stepwise_bayes_state_plus_double | 0.5183 | -0.0650 | 0.4065 | -0.1625 | 0.3493 | +0.0076 | 1.1341 | +0.0199 |
| stepwise_bayes_state_plus_lr_neg | 0.5717 | -0.0117 | 0.4440 | -0.1251 | 0.3466 | +0.0048 | 1.1296 | +0.0154 |
| stepwise_bayes_state_plus_lr_pos | 0.5450 | -0.0383 | 0.4310 | -0.1380 | 0.3431 | +0.0013 | 1.1127 | -0.0015 |
| stepwise_bayes_state_plus_sep | 0.6400 | +0.0567 | 0.7243 | +0.1552 | 0.3021 | -0.0397 | 0.9816 | -0.1326 |
| stepwise_bayes_state_plus_tempered | 0.6800 | +0.0967 | 0.7530 | +0.1839 | 0.3119 | -0.0299 | 0.9806 | -0.1336 |

The stepwise variant is a mechanics-matching stress test: proxy critics are correlated and failed episodes contribute more observations because they are usually longer.

## Best observed features

- Best AUROC: `combined/sum_logprob/bayes_state_plus_sep` = 0.9633 (Brier 0.1054).
- Best Brier: `combined/sum_logprob/bayes_state_plus_sep` = 0.1054 (AUROC 0.9633).
- Best PRR@0.5: `combined/sum_logprob/bayes_state_plus_sep` = 1.0000.
- Sum log-probability is length-sensitive; compare it against normalized mean token log-probability/perplexity before treating it as epistemic UQ.

## Success by task type

| Task type | Success | Total | Rate |
|---|---:|---:|---:|
| look_at_obj_in_light | 9 | 10 | 90.0% |
| pick_and_place_simple | 15 | 27 | 55.6% |
| pick_clean_then_place_in_recep | 4 | 21 | 19.0% |
| pick_cool_then_place_in_recep | 2 | 16 | 12.5% |
| pick_heat_then_place_in_recep | 3 | 10 | 30.0% |
| pick_two_obj_and_place | 4 | 16 | 25.0% |

## Example episodes

- `look_at_obj_in_light-06ee93788541`: success=True, steps=23, stop=success
- `look_at_obj_in_light-086a37787df8`: success=True, steps=6, stop=success
- `look_at_obj_in_light-d6f809b07d30`: success=False, steps=30, stop=max_steps
- `pick_and_place_simple-22b554a2eda4`: success=False, steps=30, stop=max_steps
