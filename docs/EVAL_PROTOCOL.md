# EVALUATION PROTOCOL (pre-register BEFORE final runs; deviations go to DECISIONS.md)

## Splits and the main result (D-017, D-020; files in data/splits/, sealed by SHA256SUMS)
- Track A (Lövdal): 5 development test groups over runners 0-63, grouped by runner; the same runner groups for the day file, the week file and fusion. Each fold's validation runners come from its non-test runners.
- HEADLINE = mean ± spread over the 5 development groups (and seeds).
- FINAL CHECK = the locked runners 64-73 (the paper's test set), opened once after the `eval-freeze` tag through src/data/lockbox.py. Reported next to the headline, never instead of it. Call it "newer athletes", not a prospective or time-based test. Its injury rate is never used for calibration.
- Each arm is also scored on its own full-file lockbox (day file 3,044 windows, week file 3,015); fusion uses the shared days (D-014).
- Lövdal paper method: rebuilt in piece 3. Its pipeline check uses stand-in runners 54-63 (trained on 0-53). It is compared with our models on the same 5 development groups, and on the lockbox at the end (`paper_split`).
- The list of compared models is fixed in piece 4, before any comparison. Every model is reported, not only the best.
- Track B: PAMAP2 4 test groups (subject 109 always in training); WESAD 5 test groups. Validation rotates, so nobody validates twice (D-020).
- Track C (exploratory, D-018): SoccerMon and PMData leave-one-person-out, predictions pooled, fixed default settings (no tuning: there is no validation person).
## Primary metrics
- Track A injury detection (5 development groups by runner, D-017): PR-AUC, recall at fixed precision (set: ____), MCC, Brier score, expected calibration error. Mean and std over the 5 groups and 3 seeds. ROC-AUC is also reported, for comparison with the Lövdal paper (proposed; confirm in piece 4).
- Track B: macro-F1 for activity (and stress, if WESAD is included) classification on the subject-grouped splits above (PAMAP2 4 folds, WESAD 5 folds).
- Track C (proposed; confirm in piece 4): pooled leave-one-person-out predictions, so threshold-free metrics only (PR-AUC, ROC-AUC).
  - SoccerMon: also scored within its main team (almost all reports come from one team), with a team-only baseline.
  - PMData: reported with and without p12 ("without" = p12 removed from training and scoring).
  - Statistics computed across people (for example scaling) use training people only.
  - Caveat: pooled AUC mixes ranking between people with ranking within a person. Most people have no episode (daily_meta.json: 14 of 50 SoccerMon players and 6 of 15 PMData people have one), so a model can score well just by telling people apart. The team view and a person-level baseline guard against that.
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
Paired comparison across folds (Wilcoxon or bootstrap CI); report effect sizes with intervals. With only 5 folds a two-sided Wilcoxon test can never go below p = 0.0625 (2/32), so the main tool is a bootstrap that resamples runners (proposed; confirm in piece 4).
## Open decisions before piece 3 (Naman decides; no piece-3 number before this)
1. Fill the Track A blanks (precision level for "recall at fixed precision", calibration bins, seeds) and fix the model list. The 5 development groups now carry the headline, so choosing these after seeing piece-3 scores would be choosing metrics after results (D-004).
2. Settings, thresholds and early stopping are chosen inside each fold, on that fold's validation people only. Never pick one global setting by averaging validation scores across folds: one fold's validation people are another fold's test people (Lövdal overlap: 46 validation places, 30 runners). The final model may use all development runners.
3. Lövdal paper method. Their test scaling (each test athlete z-scored on their own healthy events, main.ipynb cell 1 line 263) reads test labels. Proposal: use it as published only as a labelled reproduction check on stand-in runners 54-63, and use a label-free variant for every comparison.
4. Per-athlete normalisation for our models. Proposal: statistics from training runners, or from each runner's own past only, never from labels.
## Explanation criteria
- Deletion curve: removing top-k attributed inputs lowers predicted risk by at least ____ vs random removal.
- Model-randomisation check: attribution rank correlation drops below ____.
- Stability: rank correlation above ____ for near-identical inputs.
- Prior list (written BEFORE running): high load spikes, high monotony/strain, low recovery, recent injury history -> expected to raise risk.
## Evaluation freeze
Tag the repo with an annotated tag (`git tag -a eval-freeze`) and push it. Open the lockbox only through src/data/lockbox.py (it opens at one frozen commit only). Run the matrix once, commit experiments/lockbox_access.log, and log deviations.
