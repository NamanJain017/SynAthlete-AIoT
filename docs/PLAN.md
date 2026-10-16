# PLAN: piece by piece

Order is fixed; dates are flexible until the Review 2 date is confirmed. Tick a piece only with evidence (file path + numbers) in docs/STATE.md, and only after Naman says it is clear.

| # | Piece | What "done" looks like | Review 2 part |
|---|---|---|---|
| 0 | Setup | `python tasks.py check` passes on the laptop; GPU visible to PyTorch | - |
| 1 | Download + audit all 6 datasets | Audit table in docs/DATASETS.md filled from `src/data/audit.py`; a role decided for each dataset (DECISIONS entry) | Methodology |
| 2 | Loaders + person-level splits | Loaders for every dataset kept; splits saved once in data/splits; leakage tests pass; data cards written | Methodology, Implementation |
| 3 | Real-data baselines | `python tasks.py baselines` writes metrics.json + summary table (injury: majority, ACWR rule, LR, RF, CatBoost, GRU on the 5 Lövdal development groups; Lövdal paper method rebuilt, pipeline-checked on stand-in runners 54-63; locked runners untouched, D-017; activity: small CNN/GRU) | Implementation |
| 4 | Evaluation protocol | Blanks in docs/EVAL_PROTOCOL.md filled BEFORE any generator runs | Methodology |
| 5 | Generators | TimeGAN, WGAN-GP, TimeVAE, classic augmentation; fidelity, usefulness (train on synthetic, test on real) and privacy checks | Implementation |
| 6 | Edge-fog-cloud tiers | Reflex / supervisor / deep layers as separate processes; latency and bytes benchmark, cloud-only vs tiered | Implementation |
| 7 | Fused model | Sequence arm (LSTM/GRU/TCN) + tabular arm (RF/CatBoost), stacking, calibration, risk score 0-100, 3 seeds | Implementation |
| 8 | Explanations | TreeSHAP, integrated gradients, attention and 1D Grad-CAM views; deletion test; agreement with known risk factors; plain-language sentences | Implementation |
| 9 | Dashboard + demo | `python tasks.py demo` replays a held-out development runner (never runners 64-73 before the evaluation freeze) through the tiers into Streamlit | Implementation |
| 10 | Paper + Review 2 deck | IEEE conference draft with figures generated from metrics.json; deck with corrected gap sentence and dataset audit | Paper |

## If time runs short (cut in this order)
1. Extra seeds beyond 3, TCN, WGAN-GP tuning, Grad-CAM.
2. Datasets whose audit shows no usable role.
3. Never cut: person-level splits, training-fold-only fitting, real-label evaluation, honest reporting.
