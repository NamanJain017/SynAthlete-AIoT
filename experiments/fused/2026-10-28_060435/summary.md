# Piece-7 fused model, run 2026-10-28_060435

Shared days; mean ± SD over the 5 development rounds; seeds 0-2 averaged.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| fused | 0.022 ± 0.003 | 0.609 ± 0.024 |
| sequence arm (chosen) | 0.021 ± 0.003 | 0.604 ± 0.026 |
| tabular arm (chosen) | 0.019 ± 0.001 | 0.612 ± 0.025 |

Chosen arms per round: r0 tcn+catboost, r1 lstm+catboost, r2 lstm+catboost, r3 gru+catboost, r4 tcn+catboost
