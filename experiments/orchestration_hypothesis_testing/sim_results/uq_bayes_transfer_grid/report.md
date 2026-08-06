# UQ Bayes fusion transfer grid

All fused predictions are out-of-fold. Delta is PRR@0.5 minus the existing `bayes_state` on the same instances.

## Method averages

| Method | Mean PRR@0.5 | Mean delta PRR@0.5 | Mean delta Brier | Mean delta NLL | Significant wins | Significant losses |
|---|---:|---:|---:|---:|---:|---:|
| sep | 0.966 | -0.034 | -0.005 | -0.013 | 0 | 0 |
| lr_pos | 0.987 | -0.013 | -0.000 | +0.001 | 0 | 0 |
| lr_neg | 0.988 | -0.012 | +0.002 | +0.005 | 0 | 0 |
| double | 0.976 | -0.024 | +0.003 | +0.009 | 0 | 0 |
| continuous | 0.852 | -0.148 | +0.039 | +0.375 | 0 | 0 |
| tempered | 0.892 | -0.108 | +0.017 | +0.240 | 0 | 0 |

## Best method per dataset and aggregation

| Run | Dataset | Generator | Aggregation | n | Baseline PRR@0.5 | Best method | Best PRR@0.5 | Delta PRR@0.5 | 95% CI |
|---|---|---|---|---:|---:|---|---:|---:|---|
| sage_uq_longchain_med | lcb_medium | qwen3_coder | final | 75 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.006] |
| sage_uq_longchain_med | lcb_medium | qwen3_coder | max | 75 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.006] |
| sage_uq_longchain_med | lcb_medium | qwen3_coder | mean | 75 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_longchain_med | lcb_medium | qwen3_coder | min | 75 | 1.000 | sep | 1.000 | +0.000 | [-0.036, +0.000] |
| sage_uq_qwen3_coder | lcb_easy | qwen3_coder | final | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_easy | qwen3_coder | max | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_easy | qwen3_coder | mean | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_easy | qwen3_coder | min | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_medium | qwen3_coder | final | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_medium | qwen3_coder | max | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_medium | qwen3_coder | mean | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | lcb_medium | qwen3_coder | min | 30 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder | mbpp | qwen3_coder | final | 30 | 1.000 | lr_neg | 0.964 | -0.036 | [-0.153, +0.000] |
| sage_uq_qwen3_coder | mbpp | qwen3_coder | max | 30 | 1.000 | lr_pos | 0.945 | -0.055 | [-0.202, +0.000] |
| sage_uq_qwen3_coder | mbpp | qwen3_coder | mean | 30 | 1.000 | tempered | 0.945 | -0.055 | [-0.202, +0.000] |
| sage_uq_qwen3_coder | mbpp | qwen3_coder | min | 30 | 1.000 | tempered | 0.945 | -0.055 | [-0.202, +0.000] |
| sage_uq_qwen3_coder_full | lcb_easy | qwen3_coder | final | 101 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_easy | qwen3_coder | max | 101 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_easy | qwen3_coder | mean | 101 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_easy | qwen3_coder | min | 101 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_medium | qwen3_coder | final | 155 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_medium | qwen3_coder | max | 155 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_medium | qwen3_coder | mean | 155 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
| sage_uq_qwen3_coder_full | lcb_medium | qwen3_coder | min | 155 | 1.000 | sep | 1.000 | +0.000 | [+0.000, +0.000] |
