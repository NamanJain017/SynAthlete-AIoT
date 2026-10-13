# EVALUATION PROTOCOL (pre-register BEFORE final runs; deviations go to DECISIONS.md)

## Primary metrics
- Track A injury detection (grouped 5-fold by runner): PR-AUC, recall at fixed precision (set: ____), MCC, Brier score, expected calibration error. Mean and std over folds and seeds (3 seeds).
- Track B: macro-F1 for activity (and stress, if WESAD is included) classification on a fixed subject-grouped split (3 folds).
## Baselines to beat
majority class; ACWR rule; logistic regression; CatBoost; GRU; real-only vs real+augmented.
## Generator criteria (set now, before looking at results)
- Discriminative accuracy within ____ of 50%.
- TSTR macro-F1 within ____ of real-trained.
- Autocorrelation and cross-channel correlation error below ____.
- Nearest-real-record distance: no near-duplicates (threshold ____).
## Statistics
Paired comparison across folds (Wilcoxon or bootstrap CI); report effect sizes with intervals.
## Explanation criteria
- Deletion curve: removing top-k attributed inputs lowers predicted risk by at least ____ vs random removal.
- Model-randomisation check: attribution rank correlation drops below ____.
- Stability: rank correlation above ____ for near-identical inputs.
- Prior list (written BEFORE running): high load spikes, high monotony/strain, low recovery, recent injury history -> expected to raise risk.
## Evaluation freeze
Tag the repo `eval-freeze`; run the matrix once; log deviations.
