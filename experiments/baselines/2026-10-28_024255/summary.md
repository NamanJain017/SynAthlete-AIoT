# Piece-3 baselines, run 2026-10-28_024255

Mean ± SD over the 5 development rounds (test groups); chance level of PR-AUC = the test injury rate. Written by src/models/run.py.

| Model | File | PR-AUC | ROC-AUC | Test injury rate |
|---|---|---|---|---|
| random_forest | day | 0.019 ± 0.002 | 0.603 ± 0.024 | 0.0134 |
| random_forest | week | 0.022 ± 0.007 | 0.614 ± 0.042 | 0.0132 |
| catboost | day | 0.021 ± 0.004 | 0.598 ± 0.043 | 0.0134 |
| catboost | week | 0.019 ± 0.002 | 0.606 ± 0.028 | 0.0132 |
| lovdal_label_free | day | 0.017 ± 0.003 | 0.584 ± 0.044 | 0.0134 |
| lovdal_label_free | week | 0.021 ± 0.006 | 0.589 ± 0.066 | 0.0132 |

Reproduction check (exact copy, runners [0, 53] -> [54, 63], label-reading scaling, never used to choose; pass = 95% runner-bootstrap lower bound above 0.5):
- day: test ROC-AUC 0.617 (0.573 to 0.664), PASSED
- week: test ROC-AUC 0.613 (0.546 to 0.675), PASSED
