# EVALUATION PROTOCOL (pre-register BEFORE final runs; deviations go to DECISIONS.md)

## Splits and the main result (D-017, D-020, D-023; files in data/splits/, sealed by SHA256SUMS)
- Track A (Lövdal): 5 development test groups over runners 0-63, grouped by runner; the same runner groups for the day file, the week file and fusion. Each group's validation runners come from its non-test runners.
- HEADLINE = mean ± SD over the 5 development groups; the 3 seeds are averaged within each group first (D-023).
- FINAL CHECK = the locked runners 64-73, opened once after the `eval-freeze` tag through src/data/lockbox.py. Reported next to the headline, never instead of it. Call it "newer athletes", not a prospective or time-based test. Its injury rate is never used for calibration.
  - Primary on the lockbox: the fused model, on the shared days (2,994 events, 48 injuries).
  - Secondary, scored once and never used to choose anything: the chosen arms on their own files (day 3,044 / 50; week 3,015 / 48) and the Lövdal label-free variant trained on runners 0-63 (`paper_split`). Only the day file matches the paper's test set exactly. A runner bootstrap over 10 runners understates uncertainty: disclosed.
- Lövdal paper method: the exact copy runs once, as a labelled reproduction check on stand-in runners 54-63 (trained on 0-53); the label-free variant is the one compared on the 5 groups (D-021, D-023).
- The model list is fixed (D-022, section "Models" below); every model is reported, not only the best. The model for the locked test is declared in advance: the fused model (D-021, D-023).
- Track B: PAMAP2 4 test groups (subject 109 always in training). WESAD 5 test groups exist but WESAD is not a piece-3 baseline (D-023). Validation rotates, so nobody validates twice (D-020).
- Track C (exploratory, D-018): SoccerMon and PMData leave-one-person-out, predictions pooled, fixed default settings (no tuning: there is no validation person).
## Primary metrics
- Track A injury detection (5 development groups by runner; scoring fixed by D-022 and D-023):
  - HEADLINE: PR-AUC computed as average precision, on raw scores, each group shown with its injury rate (the chance level). ROC-AUC alongside, for the paper comparison.
  - ALERT LEVEL: smallest |sensitivity - specificity| on the group's validation runners (the Lövdal rule), applied to that group's test runners. Candidates = the distinct validation scores plus +infinity ("alert nobody"); alert if score >= threshold; ties broken by the larger sensitivity + specificity, then the higher threshold. At that level: sensitivity, specificity, precision, false alarms per caught injury, alerts per 100 scored events, MCC. Undefined values (0/0) are reported as n/a, never 0.
  - CALIBRATION: Platt scaling always fitted on the group's validation runners, on logit(p) for probability outputs and on the raw score otherwise (D-025). Brier score, ECE with 10 equal-count bins (tied probabilities share a bin, D-025), calibration slope and intercept, reported raw and after Platt; raw calibration is n/a for outputs that are not probabilities; a Platt slope of 0 or below is flagged.
  - EXPLORATORY: decision curve (net benefit) on Platt-calibrated risks over 0.5-10 %. The event sample's injury rate is not real incidence, so this is not clinical utility.
  - Not used: a fixed-precision target (precision depends on the injury rate); accuracy.
  - Limitation stated with every result: healthy days near injuries were removed by the authors, so alert rates are not real daily rates.
- Track B: macro-F1 for PAMAP2 activity classification (4 subject groups).
- Track C (proposed; confirm before any Track C run): pooled leave-one-person-out predictions, so threshold-free metrics only (PR-AUC, ROC-AUC).
  - SoccerMon: also scored within its main team (almost all reports come from one team), with a team-only baseline.
  - PMData: reported with and without p12 ("without" = p12 removed from training and scoring).
  - Statistics computed across people (for example scaling) use training people only.
  - Caveat: pooled AUC mixes ranking between people with ranking within a person. Most people have no episode (daily_meta.json: 14 of 50 SoccerMon players and 6 of 15 PMData people have one), so a model can score well just by telling people apart. The team view and a person-level baseline guard against that.
## Fitting rules (D-021, D-023)
- Scalers, imputers, generators and resamplers are fitted on each group's training people only.
- Scaling: one z-score scaler fitted on the group's training-runner rows, for every model. No per-runner statistics anywhere: in this event design a runner's own event history reveals the label (data/processed/lovdal/event_design_check.json).
- Never model inputs: date or day index, athlete ID, event counts, gaps, history length. Piece 3 adds a leak-canary report and a test for this.
- Validation runners are used for settings, Platt calibration, the alert level and early stopping, inside their own group only; never one setting by averaging validation scores across groups. The final model's settings are chosen on its own validation group (13 runners).
- Tuning budget: 8 settings per tunable model per group, drawn by seeded random search from a grid written in configs/ before the first run. Untuned: majority class (constant training injury rate), the ACWR-style rule, both Lövdal copies.
- Class balancing and resampling on training runners only.
- Fused model: sequence arm from {GRU, LSTM, TCN} on the day file; tabular arm from {random forest, CatBoost} on the week file; combiner = logistic regression on inner out-of-fold predictions from the training runners, on the shared days. Arms and settings are chosen inside each group on its validation runners only.
## Models (D-022, D-023; all reported)
- Injury (Lövdal, 5 groups):
  - majority class;
  - ACWR-style rule: week file, most recent week's total km / max(mean total km of the two earlier weeks, 1 km), the uncoupled ratio; higher = riskier;
  - logistic regression, random forest and CatBoost on the day file (70 inputs) and the week file (69 inputs);
  - GRU on the day file;
  - Lövdal bagged XGBoost (notebook settings: 9 bags, up to 2,048 events per class drawn per athlete with replacement, learning rate 0.01, depth {2, 3}, trees {256, 512}, Platt): label-free variant on the 5 groups; the exact copy only as the reproduction check on runners 54-63, with its library deviations documented.
- Activity (PAMAP2, 4 groups): small 1D-CNN and GRU, macro-F1.
- Later pieces: real-only vs real+augmented (piece 5); the fused model (piece 7) is the primary model on the locked test.
## Piece-3 details (D-024; settings in configs/baselines/*.yaml and configs/fused_candidates.yaml)
- Selection score: validation average precision (ties: validation ROC-AUC); PAMAP2: validation macro-F1.
- Budget: 8 settings per tunable model per round, seeded random search from the config grids; fused model (piece 7): 44 tries per round, declared now.
- Constant-risk baseline: alert candidates include "alert nobody", so it raises no alerts (alert numbers n/a).
- Leak canary: declared-input test + shuffled-label logistic regression; fails if the 5-round mean test ROC-AUC is more than 0.05 from 0.5.
- Reproduction check passes if the runner-bootstrap (2,000) lower bound of its test ROC-AUC is above 0.5.
- Seeds: settings tried with seed 0; the chosen one refit with seeds 0-2 on training runners only.
Before the evaluation freeze: which runners give validation (Platt, alert level) for the Lövdal label-free variant trained on `paper_split`, which has no validation group.
Track C (proposed; confirm before any Track C run): the same history-signal ban; features come from the full daily tables, never from the selected event rows.
## Generator criteria (set now, before looking at results)
- Discriminative accuracy within 0.15 of 50% (post-hoc GRU classifier, real test-people windows vs synthetic, held-out 30 %) (D-027).
- TSTR macro-F1 within 0.15 of real-trained (piece-3 CNN, the round's chosen setting; trained on synthetic only, tested on the round's real test people).
- Autocorrelation (lags 1-10) and cross-channel correlation error below 0.10 (mean absolute difference, synthetic vs real test-people windows).
- Nearest-real-record distance: at most 5 % of synthetic windows closer to a real training window than the 5th percentile of real test-to-training nearest distances.
## Statistics
Differences between models: paired bootstrap that resamples runners within each test group and recomputes the 5-group mean difference (2,000 resamples, percentile 95 % interval), on the same rows (shared days across files; D-023). Primary comparisons, fixed now: each model vs the Lövdal label-free variant on the same file; in piece 7, the fused model vs each arm. With only 5 groups a two-sided Wilcoxon test can never go below p = 0.0625 (2/32), so it is not the main test. No claim of "better" without an interval.
## Explanation criteria
- Deletion curve: removing the top-10 attributed inputs lowers predicted risk at least 1.5 times as much as removing 10 random inputs (D-027).
- Model-randomisation check: attribution rank correlation (Spearman) between the trained and a random-weight model below 0.5.
- Stability: rank correlation above 0.8 for near-identical inputs (Gaussian noise of 1 % of each input's SD).
- Prior list (written BEFORE running): high load spikes, high monotony/strain, low recovery -> expected to raise risk. ("Recent injury history" is not testable: injury history is not an input under D-023, and the event design keeps injuries at least 22 days from other events.)
## Evaluation freeze
Tag the repo with an annotated tag (`git tag -a eval-freeze`) and push it. Open the lockbox only through src/data/lockbox.py (it opens at one frozen commit only). Run the matrix once, commit experiments/lockbox_access.log, and log deviations.
## Sources for the scoring and tuning rules (checked 2026-10-22 on the page named)
- Lövdal, den Hartigh & Azzopardi 2021, Int J Sports Physiol Perform, doi 10.1123/ijspp.2020-0518 (alert rule; main.ipynb cell index 1 line 86). Full text: https://pure.rug.nl/ws/portalfiles/portal/183763727/
- Saito & Rehmsmeier 2015, PLoS ONE, doi 10.1371/journal.pone.0118432. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
- Davis & Goadrich 2006, ICML, doi 10.1145/1143844.1143874. https://mlanthology.org/icml/2006/davis2006icml-relationship
- Varma & Simon 2006, BMC Bioinformatics, doi 10.1186/1471-2105-7-91. https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873
- Cawley & Talbot 2010, J Mach Learn Res 11:2079-2107 (no DOI). https://www.jmlr.org/papers/v11/cawley10a.html
- Kapoor & Narayanan 2023, Patterns, doi 10.1016/j.patter.2023.100804. https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856
- Van Calster et al. 2019, BMC Medicine, doi 10.1186/s12916-019-1466-7. https://pubmed.ncbi.nlm.nih.gov/31842878/
- Niculescu-Mizil & Caruana 2005, ICML, doi 10.1145/1102351.1102430. https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting
- Vickers & Elkin 2006, Med Decis Making, doi 10.1177/0272989X06295361. https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036
- Vickers, Van Calster & Steyerberg 2016, BMJ, doi 10.1136/bmj.i6. https://pubmed.ncbi.nlm.nih.gov/26810254/
- Bahr 2016, Br J Sports Med, doi 10.1136/bjsports-2016-096256. https://bjsm.bmj.com/content/early/2016/04/25/bjsports-2016-096256.info
- Bullock et al. 2022, Sports Medicine, doi 10.1007/s40279-022-01698-9. Keele repository record, output 423719
- Collins et al. 2024 (TRIPOD+AI), BMJ, doi 10.1136/bmj-2023-078378. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/
- Lolli et al. 2019, Br J Sports Med 53:921-922, doi 10.1136/bjsports-2017-098110. https://bjsm.bmj.com/content/early/2017/11/03/bjsports-2017-098110.info
- Notebook references: "cell 1" means notebook cell index 1 (the second cell) of data/raw/lovdal/main.ipynb.
