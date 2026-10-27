# Piece-3 baselines, run 2026-10-26_022902

Mean ± SD over the 5 development rounds (test groups); chance level of PR-AUC = the test injury rate. Written by src/models/run.py.

| Model | File | PR-AUC | ROC-AUC | Test injury rate |
|---|---|---|---|---|
| gru | day | 0.023 ± 0.006 | 0.631 ± 0.009 | 0.0134 |

| PAMAP2 model | Test macro-F1 (4 rounds) |
|---|---|
| pamap2_cnn1d | 0.632 ± 0.097 |
| pamap2_gru | 0.566 ± 0.054 |
