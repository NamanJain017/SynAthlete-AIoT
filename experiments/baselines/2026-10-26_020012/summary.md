# Piece-3 baselines, run 2026-10-26_020012

Mean ± SD over the 5 development rounds (test groups); chance level of PR-AUC = the test injury rate. Written by src/models/run.py.

Leak canary: PASSED (|mean test ROC-AUC - 0.5| <= 0.05 for every file); day mean test ROC-AUC 0.498, week mean test ROC-AUC 0.513

| Model | File | PR-AUC | ROC-AUC | Test injury rate |
|---|---|---|---|---|
| constant_risk | day | 0.013 ± 0.000 | 0.500 ± 0.000 | 0.0134 |
| constant_risk | week | 0.013 ± 0.000 | 0.500 ± 0.000 | 0.0132 |
| acwr_style | week | 0.015 ± 0.001 | 0.540 ± 0.021 | 0.0132 |
| logistic_regression | day | 0.022 ± 0.004 | 0.626 ± 0.019 | 0.0134 |
| logistic_regression | week | 0.020 ± 0.004 | 0.621 ± 0.029 | 0.0132 |
