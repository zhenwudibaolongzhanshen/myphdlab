# RC-49

This folder is a standalone RC-49 selective-regression benchmark.

Flow:

1. `extract_cnn_features.py` trains a CNN on RC-49 with CUDA and exports locked train/validation/test CNN features.
2. `run_comparison.py` trains seven downstream methods on the frozen features.
3. `run_all_gpu.ps1` runs both steps.

The SOGN implementation uses the Bike Sharing style:

- ordinal pretraining
- joint training with HC + LS gate
- reg_net-only plain-MSE fine-tuning

By default the RC-49 report ranks SOGN predictions by the learned HC+LS gate.
Use `--sogn-score-mode bike` to rank with the exact Bike Sharing score
`max_prob + local_support`, or `--sogn-score-mode combined` to add both signals.
