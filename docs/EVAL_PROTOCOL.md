# EVALUATION PROTOCOL (pre-register BEFORE final runs; deviations go to DECISIONS.md)

## Splits and the main result (D-017, D-020; files in data/splits/, sealed by SHA256SUMS)
- Track A (Lövdal): 5 development test groups over runners 0-63, grouped by runner; the same runner groups for the day file, the week file and fusion. Each fold's validation runners come from its non-test runners.
- HEADLINE = mean ± spread over the 5 development groups (and seeds).
- FINAL CHECK = the locked runners 64-73 (the paper's test set), opened once after the `eval-freeze` tag through src/data/lockbox.py. Reported next to the headline, never instead of it. Call it "newer athletes", not a prospective or time-based test. Its injury rate is never used for calibration.
- Each arm is also scored on its own full-file lockbox (day file 3,044 windows, week file 3,015); fusion uses the shared days (D-014).
- Lövdal paper method: rebuilt in piece 3. Its pipeline check uses stand-in runners 54-63 (trained on 0-53). It is compared with our models on the same 5 development groups, and on the lockbox at the end (`paper_split`).
- The list of compared models is fixed (D-022, section "Models" below). Every model is reported, not only the best. The model for the locked test is declared in advance: the fused model (piece 7, D-021).
- Track B: PAMAP2 4 test groups (subject 109 always in training); WESAD 5 test groups. Validation rotates, so nobody validates twice (D-020).
- Track C (exploratory, D-018): SoccerMon and PMData leave-one-person-out, predictions pooled, fixed default settings (no tuning: there is no validation person).
## Primary metrics
- Track A injury detection (5 development groups by runner, D-017; scoring fixed by D-022):
  - HEADLINE: PR-AUC computed as average precision, mean ± SD over the 5 groups (and seeds 0, 1, 2), each shown with that group's injury rate (the chance level). ROC-AUC alongside, for the paper comparison.
  - ALERT LEVEL: smallest |sensitivity - specificity| on the group's validation runners (the Lövdal rule), applied to that group's test runners. At that level: sensitivity, specificity, precision, false alarms per caught injury, alerts per 100 runner-days, MCC.
  - CALIBRATION: Brier score, ECE with 10 equal-count bins, calibration slope and intercept. Platt scaling fitted on validation runners if calibration is needed.
  - EXPLORATORY: decision curve (net benefit) on calibrated risks over 0.5-10 %.
  - Not used: a fixed-precision target (precision depends on the injury rate); accuracy.
  - Limitation stated with every result: healthy days near injuries were removed by the authors, so alert rates are not real daily rates.
- Track B: macro-F1 for activity (and stress, if WESAD is included) classification on the subject-grouped splits above (PAMAP2 4 folds, WESAD 5 folds).
- Track C (proposed; confirm in piece 4): pooled leave-one-person-out predictions, so threshold-free metrics only (PR-AUC, ROC-AUC).
  - SoccerMon: also scored within its main team (almost all reports come from one team), with a team-only baseline.
  - PMData: reported with and without p12 ("without" = p12 removed from training and scoring).
  - Statistics computed across people (for example scaling) use training people only.
  - Caveat: pooled AUC mixes ranking between people with ranking within a person. Most people have no episode (daily_meta.json: 14 of 50 SoccerMon players and 6 of 15 PMData people have one), so a model can score well just by telling people apart. The team view and a person-level baseline guard against that.
## Fitting rules
- Scalers, imputers, generators and resamplers are fitted on each fold's training people only.
- Fusion stacker (proposed; confirm in piece 7): trained on inner out-of-fold predictions from the training runners. Validation runners are used only for calibration, threshold choice and early stopping.
- Tuning rules (D-021):
  - Settings, alert level, calibration and early stopping are chosen inside each group on that group's validation runners only; never one setting by averaging validation scores across groups. The final model's settings are chosen on its own validation group (13 runners).
  - Every model gets the same pre-declared tuning budget per group.
  - Our models: one scaler fitted on the group's training runners; per-runner scaling on that runner's own earlier events only, as a tuned setting. Never labels, never later days.
  - Lövdal method: the exact copy (label-reading test scaling) only once, as a labelled reproduction check on runners 54-63 (trained on 0-53); never used to choose anything. Comparisons use the label-free variant.
  - Class balancing and resampling on training runners only.
## Models (D-022; all reported)
- Injury (Lövdal, 5 groups): majority class; ACWR-style 3-week rule (D-015); logistic regression, random forest and CatBoost on the day file (70 inputs) and the week file (69 inputs); GRU on the day file; Lövdal bagged XGBoost (label-free variant; exact copy only as the reproduction check).
- Activity (PAMAP2, 4 groups): small 1D-CNN and GRU, macro-F1.
- Later pieces: real-only vs real+augmented (piece 5); the fused model (piece 7) is the one model sent to the locked test.
## Generator criteria (set now, before looking at results)
- Discriminative accuracy within ____ of 50%.
- TSTR macro-F1 within ____ of real-trained.
- Autocorrelation and cross-channel correlation error below ____.
- Nearest-real-record distance: no near-duplicates (threshold ____).
## Statistics
Differences between models: paired bootstrap that resamples runners, reported as effect sizes with intervals (D-022). With only 5 groups a two-sided Wilcoxon test can never go below p = 0.0625 (2/32), so it is not the main test. No claim of "better" without an interval.
## Explanation criteria
- Deletion curve: removing top-k attributed inputs lowers predicted risk by at least ____ vs random removal.
- Model-randomisation check: attribution rank correlation drops below ____.
- Stability: rank correlation above ____ for near-identical inputs.
- Prior list (written BEFORE running): high load spikes, high monotony/strain, low recovery, recent injury history -> expected to raise risk.
## Evaluation freeze
Tag the repo with an annotated tag (`git tag -a eval-freeze`) and push it. Open the lockbox only through src/data/lockbox.py (it opens at one frozen commit only). Run the matrix once, commit experiments/lockbox_access.log, and log deviations.
## Sources for the scoring and tuning rules (DOIs verified 2026-10-22)
- Lövdal, den Hartigh & Azzopardi 2021, Int J Sports Physiol Perform, doi 10.1123/ijspp.2020-0518 (alert rule; main.ipynb cell 1 line 86).
- Saito & Rehmsmeier 2015, PLoS ONE, doi 10.1371/journal.pone.0118432 (PR plots for imbalanced data).
- Davis & Goadrich 2006, ICML, doi 10.1145/1143844.1143874 (no linear interpolation in PR space).
- Varma & Simon 2006, BMC Bioinformatics, doi 10.1186/1471-2105-7-91 (tune inside each round).
- Cawley & Talbot 2010, J Mach Learn Res 11:2079-2107 (selection bias).
- Kapoor & Narayanan 2023, Patterns, doi 10.1016/j.patter.2023.100804 (leakage types).
- Van Calster et al. 2019, BMC Medicine, doi 10.1186/s12916-019-1466-7 (calibration).
- Niculescu-Mizil & Caruana 2005, ICML, doi 10.1145/1102351.1102430 (Platt scaling and isotonic regression).
- Vickers & Elkin 2006, Med Decis Making, doi 10.1177/0272989X06295361; Vickers, Van Calster & Steyerberg 2016, BMJ, doi 10.1136/bmj.i6 (net benefit).
- Bahr 2016, Br J Sports Med, doi 10.1136/bjsports-2016-096256 (limits of individual injury prediction).
- Bullock et al. 2022, Sports Medicine, doi 10.1007/s40279-022-01698-9 (quality of sports-injury prediction models).
- Collins et al. 2024 (TRIPOD+AI), BMJ, doi 10.1136/bmj-2023-078378 (reporting checklist).
