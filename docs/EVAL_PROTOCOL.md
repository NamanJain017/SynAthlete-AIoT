# EVALUATION PROTOCOL (pre-register BEFORE final runs; deviations go to DECISIONS.md)

## Splits and the main result (D-017, D-020; files in data/splits/, sealed by SHA256SUMS)
- Track A (Lövdal): 5 development test groups over runners 0-63, grouped by runner; the same runner groups for the day file, the week file and fusion. Each fold's validation runners come from its non-test runners.
- HEADLINE = mean ± spread over the 5 development groups (and seeds).
- FINAL CHECK = the locked runners 64-73 (the paper's test set), opened once after the `eval-freeze` tag through src/data/lockbox.py. Reported next to the headline, never instead of it. Call it "newer athletes", not a prospective or time-based test. Its injury rate is never used for calibration.
- Each arm is also scored on its own full-file lockbox (day file 3,044 windows, week file 3,015); fusion uses the shared days (D-014).
- Lövdal paper method: rebuilt in piece 3. Its pipeline check uses stand-in runners 54-63 (trained on 0-53). It is compared with our models on the same 5 development groups, and on the lockbox at the end (`paper_split`).
- The list of compared models is fixed in piece 4, before any comparison. Every model is reported, not only the best.
- Track B: PAMAP2 4 test groups (subject 109 always in training); WESAD 5 test groups. Validation rotates, so nobody validates twice (D-020).
- Track C (exploratory, D-018): SoccerMon and PMData leave-one-person-out, predictions pooled.
## Primary metrics
- Track A injury detection (5 development groups by runner, D-017): PR-AUC, recall at fixed precision (set: ____), MCC, Brier score, expected calibration error. Mean and std over the 5 groups and 3 seeds. ROC-AUC is also reported, for comparison with the Lövdal paper (proposed; confirm in piece 4).
- Track B: macro-F1 for activity (and stress, if WESAD is included) classification on the subject-grouped splits above (PAMAP2 4 folds, WESAD 5 folds).
- Track C (proposed; confirm in piece 4): pooled leave-one-person-out predictions, so threshold-free metrics only (PR-AUC, ROC-AUC).
  - SoccerMon: also scored within its main team (almost all reports come from one team), with a team-only baseline.
  - PMData: reported with and without p12 ("without" = p12 removed from training and scoring).
  - Statistics computed across people (for example scaling) use training people only.
## Fitting rules
- Scalers, imputers, generators and resamplers are fitted on each fold's training people only.
- Fusion stacker (proposed; confirm in piece 7): trained on inner out-of-fold predictions from the training runners. Validation runners are used only for calibration, threshold choice and early stopping.
- Open decision for piece 3: per-athlete normalisation (each runner compared with their own history) or one scaler for everyone. Ask Naman before the first baseline run.
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
