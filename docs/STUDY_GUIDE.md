# SynAthlete-AIoT: the complete study guide (explained like you're in first year)

Last updated 2026-10-15. Read it once from top to bottom, then use it as a reference before every review and the viva.
Short per-piece Q&A stays in `docs/VIVA_NOTES.md`; this file explains everything from zero.

**Status labels used everywhere**
- ✅ **Done**: built, run on real data, numbers come from files written by code.
- 🔜 **Planned**: decided, not built yet. No results exist yet; never quote numbers for these.
- 📝 **Proposed**: suggested, waiting for Naman's approval.

**Golden rule for the viva:** every number you say must come from a file (an `audit.json`, a `metrics.json`, or the Lövdal paper). If something isn't built yet, say "that part is planned; we have no result yet". That honesty is a strength.

**How this guide is organised**
- **Part 1 (sections 1-19): our project**, from the problem to the demo.
- **Part 2 (sections 20-32): general AI / ML / deep learning / IoT foundations.** A teacher can ask about any of these, even if our project doesn't use them directly.
- **Part 3 (sections 33-34): question bank and numbers cheat sheet.**

---

# PART 1: OUR PROJECT

## 1. The project in plain words

**Problem.** Athletes get injured when training load builds up over days and weeks. Coaches can't easily see this risk building. A 2026 AIoT framework paper (Applied Sciences 16(9):4542) designed a wearable + edge/fog/cloud system for athletes but left the "AI part" (injury-risk prediction and explanations) unimplemented.

**Our idea: build that missing AI part as working software.**
1. Wearable-style sensor data flows through three computer "tiers" (edge = watch, fog = phone, cloud = server).
2. A model trained on **real runners' daily training logs with real injuries** predicts injury risk (0 to 100).
3. Generators (GANs) create extra realistic data to help training where real data is scarce.
4. Every prediction comes with a reason a coach can read (explainability).
5. A coach dashboard shows athletes, risk scores, reasons, and the system.

**What is real and what is not (very important).**
- Injury results come **only** from real injury labels (the Lövdal runner logs). Rule 7 in `AGENTS.md`.
- Sensor datasets (PAMAP2, WESAD) are **different people** with no injury records. They show the IoT pipeline and train generators; they never prove anything about injuries.
- Synthetic (generated) data is only ever used for **training**, never for testing (rule 3).

**Review 2 rubric (20 marks):** Methodology 5 (pipeline, dataset audit, data roles, evaluation rules), Implementation 10 (working code with real results), Paper draft 5 (IEEE-style conference paper).

**What kind of ML problem is this?** *Supervised binary classification on time-series data*: supervised because we have answers (injury yes/no); binary because there are two classes; time series because each example is a sequence of days.

---

## 2. Words you must know (mini-dictionary)

| Word | Simple meaning | Example in our project |
|---|---|---|
| Dataset | A collection of data in files | Lövdal runner logs |
| Row / sample / example | One item the model looks at | One runner's 7 days before a day |
| Feature | One input number describing a sample | "total km on the day before" |
| Label / target | The answer the model must predict | injury = 1 or 0 |
| Window | A fixed slice of a time series | 7 days (day file) |
| Model | A formula/program mapping features to a prediction | Random Forest, GRU |
| Training | The model learns from examples with known answers | Fitting on training runners |
| Inference / prediction | Using a trained model on new data | Scoring a new week |
| Parameter (weight) | A number *inside* the model, learned in training | GRU weights |
| Hyperparameter | A setting *you* choose before training | learning rate, number of trees |
| Overfitting | Memorising training data instead of learning patterns | Great on training runners, bad on new ones |
| Underfitting | Model too simple to learn the pattern | Always says "no injury" |
| Validation set | "Mock test": used to choose settings and thresholds | Held-out runners |
| Test set | "Final exam": used for the honest score | Runners never seen |
| Leakage | Test information sneaking into training | Same runner in train and test |
| Class imbalance | One label far rarer than the other | 1.36 % injuries |
| Synthetic data | Data made by a computer program | GAN output |
| Seed | A number that fixes randomness so results repeat | seed 0 |
| Epoch / batch | One full pass over the training data / a small group processed together | 30 epochs, batch of 128 |
| Loss | A number measuring how wrong the model is | binary cross-entropy |
| Gradient | Direction to change each weight to reduce the loss | from backpropagation |
| Threshold | Cut-off turning a probability into yes/no | alert if risk ≥ 0.45 |
| Calibration | Making "70 %" really mean about 70 % | Platt scaling |
| AUC | How well the model ranks risky above safe | 0.5 random, 1.0 perfect |
| sha256 | A unique fingerprint of a file's bytes | `data/MANIFEST.csv` |
| AIoT | AI + Internet of Things | watch → phone → server |
| Edge / fog / cloud | Device / nearby gateway / remote server | watch / phone / server |
| Latency | Delay between event and response | ms from reading to alert |
| SHAP | Splits a prediction into each feature's contribution | "risk up because km jumped" |

---

## 3. The datasets: what each one is, whether it's real, what it's used for

Numbers come from files written by code: the audit (latest run `experiments/audit/2026-10-15_0819/`), the loader metadata in `data/processed/` (also shown in `docs/data_cards/`) and the split files in `data/splits/`, unless marked "paper".

### #1 Lövdal runner logs ✅ (our main evidence)
- **What:** daily training logs of a Dutch high-level running team, 2012-2019, 800 m to marathon (paper). 74 runners (27 women, 47 men) after the authors' cleaning; 77 before (paper).
- **Two files:**
  - *Day file:* each row = one runner, one "event day", described by the 7 days before it × 10 features = 70 numbers. 42,766 rows; 583 injury rows (1.36 %).
  - *Week file:* each row = the 3 weeks before the event day, each week summarised by 22 features, + 3 "mileage change" features = 69 numbers. 42,798 rows; 575 injury rows (1.34 %).
- **The 10 day features:** number of sessions, total km, km in zones 3-4, km in zone 5 / track intervals (T1, T2), km sprinting, strength training, hours of cross-training, perceived exertion, perceived training success, perceived recovery. The three "perceived" ones use a 0-1 scale; **-0.01 means rest day** (authors' README).
- **What counts as an injury (paper):** a "flag" in the runner's own training log: "unable to complete the scheduled session due to injury". Injuries less than 3 weeks apart count as one, so all 583 are at least 22 days apart (`data/processed/lovdal/facts.json`).
- **Real?** Yes. Peer-reviewed paper (Int J Sports Physiol Perform 2021), university repository with DOI, ethics approval, CC0 licence. Our audit counts match the paper exactly (583 + 42,183 day; 575 + 42,223 week).
- **Trap we found:** the two files count days in **opposite directions**. Day file: suffix `.6` = the day before the event, no suffix = 7 days before. Week file: no suffix = the week before. The loader must put both in oldest-to-newest order.
- **Athlete IDs:** 0-73. The authors' notebook takes the last 10 IDs (64-73) as "the last athletes that joined the club": the paper's test runners. IDs 0-63 are only roughly in joining order (runners joined in groups), so never read an ID as a date.
- **Role (D-012):** the only source of injury evidence (sequence model, snapshot model, explanations, dashboard replay).

### #2 Kaggle "Injury Prediction for Competitive Runners" ✅
Byte-for-byte identical to Lövdal (same sha256 fingerprints). It's the dataset shown in Review 1. **Role:** none separate; we cite the original.

### #3 PAMAP2 ✅
- 9 people (readme: 1 female, 8 male, age 27.22 ± 3.31) wearing 3 motion sensors (wrist, chest, ankle; 100 readings per second) + a heart-rate strap (about 9 per second; we measured a median of 9.13).
- 10.7 hours (Protocol 7.98 h + Optional 2.72 h), 7.57 h labelled with one of 18 activities (lying, walking, running, cycling…).
- Subject 109 has only ~85 seconds of protocol data (1 activity); subject 103 did 8 of the 12 protocol activities.
- **Real?** Yes: published benchmark; our check shows heart rate 156.6 bpm running vs 75.5 lying.
- **Not athletes** (mostly employees and students): say "wearable sensor data".
- **Role:** live sensor stream for the edge/fog tiers; generator training.

### #4 WESAD ✅
- 15 people (S1 and S12 removed by the authors), 24.13 h, chest device (700 readings/s, 8 channels) + wrist device (4-64 readings/s, 6 channels). Baseline 293.5 min, stress 166.1 min, amusement 92.9 min, meditation 196.8 min. 12 men, 3 women, ages 24-35.
- **Real?** Yes: published paper (ACM ICMI 2018). Licence: academic, non-commercial.
- **Role:** second sensor dataset, to test whether a generator also works on different data.

### #5 Sport DB 2.0 ✅
- 168 heart recordings, 11 sports; 129 subject folders + 4 skating recordings without one. 55 are continuous (~127.6 h of heart-rate data); 113 are short ECG clips at rest or after exercise.
- Sampling rates aren't stored in the files; our code infers heart rate ≈ 1 sample/second and ECG = 250 samples per heart-rate sample. Marked [VERIFY].
- **Real?** Probably: university deposit with an ethics document; height and weight correlate (r = 0.686) as in real people.
- **Role:** realism check, i.e. does synthetic heart rate look like real athletes' heart rate?

### #6 SIRP-600 ✅ (demo only)
- 600 rows, 15 features, label "Injury_Risk" (411 zeros, 189 ones); no athlete ID, no time.
- **Warning signs:**
  - A 2-question decision tree predicts the label 89.7 % of the time, against 68.5 % for always guessing the common answer. It uses only Injury_History and Training_Intensity.
  - The file was created and last saved 4 seconds apart.
  - It's named "High_Accuracy".
- **Role (D-013):** demo / sanity check only. It was proposed for the snapshot model in Review 1; that role moved to Lövdal's week file.

### #7 ziya07 wearable ✅ (demo only)
- 500 rows, 10 athlete IDs. **Fails a real-body check:** heart rate "Resting" 120.6 vs "Running" 118.0 (p = 0.569); real PAMAP2 shows +81 bpm.
- **Role:** dashboard demo only.

### #8 SoccerMon (subjective part, unzipped) 🔜
- Two elite women's football teams in Norway, 2020-2021 (paper). `subjective.zip` (705.8 kB; checksum matches Zenodo): daily training load (incl. ACWR), wellness (fatigue, mood, sleep, stress, soreness, readiness), injuries, illness. 731 days × 50 players (Team A 27, Team B 23).
- **Injuries:** 162 reports from 15 players (11 in Team A, 4 in Team B). An "injury" is a tap on a body map, e.g. `{"right_foot":"minor"}`: self-reported pain, not a diagnosis. The same pain is often reported on consecutive days, and a few players filed most reports.
- GPS / heart-rate files (99.1 GB) not downloaded. Licence CC BY 4.0.
- **Role:** 📝 proposed: secondary, exploratory check.

### #9 PMData (unzipped) 🔜
- 16 participants aged 23-60 (13 men, 3 women), 5 months; Fitbit Versa 2 (heart rate, steps, sleep…), daily self-reports (PMSys app), meals/weight (Google Forms). 1.42 GB zip. Food photos were not extracted (not needed).
- **Injuries:** 225 reports, but only **77 name an injury**; the rest are empty "no injury" check-ins. 9 people reported at least one. **One person (p12) filed 48 of the 77.** p08 has no injury file at all.
- **Cautions:** self-reported pain, only 16 people, not athletes.
- **Role:** 📝 proposed: secondary, exploratory check.

---

## 4. What has been done so far (with real numbers)

### Piece 0: setup ✅
- Python 3.12.2 in a private environment (`.venv`).
- PyTorch 2.14.1 with CUDA on an NVIDIA RTX 4050 Laptop GPU (~6 GB).
- Exact package versions recorded in `requirements.lock.txt`.
- `python tasks.py check` runs ruff (code checker) + pytest (automatic tests). 23 tests passed at the end of piece 1; the current count is in `docs/STATE.md`.

### Piece 1: dataset audit ✅
- `python tasks.py audit` opens every dataset and writes facts: people, hours, signals, labels, injuries, missing values, duplicates, plausibility checks.
- **Outputs:**
  - `experiments/audit/<run>/`: `audit.json`, `audit_table.md`, `config.yaml`, `run_info.json` (code version, seed, data fingerprints);
  - `data/MANIFEST.csv`: sha256 of every raw file read (729 after the 2026-10-15 run, which added SoccerMon and PMData);
  - the table in `docs/DATASETS.md`.
- **Speed:** files are processed in parallel on all CPU cores, limited by free memory. A full run takes about 1.5-2 minutes with enough free RAM; it once took ~18 minutes with only 2.6 GB free.
- **Safety:** WESAD pickle files are opened with a "numpy-only" loader that refuses hidden code. A test proves it refuses a pickle that tries to run a command.
- **Findings:**
  - Kaggle = Lövdal;
  - SIRP-600 and ziya07 fail plausibility checks;
  - Sport DB hours were first overcounted, then fixed;
  - the day/week order trap.
- **Decisions D-012 to D-015 (approved by Naman):** dataset roles; the snapshot model uses Lövdal week features; fusion uses the 42,680 days present in both Lövdal files; the ACWR baseline is an adapted 3-week version.

### After piece 1 ✅
- Lövdal paper read in full (section 15).
- SoccerMon and PMData found, downloaded, checked and unzipped.
- D-016: dashboard Training-lab rules.
- Splits re-verified (section 5).

### Piece 2: loaders + splits ✅ (built 2026-10-15; waiting for Naman to say it is clear)
- **Easiest explanation of all 9 parts of piece 2, with everyday examples: `docs/PIECE2_EXPLAINED.md`.**
- Loaders, splits, safeguards and the one-time re-make: section 5 below and `docs/VIVA_NOTES.md` (piece 2).

---

## 5. Honest testing: train / validation / test, splits, leakage

### The exam analogy
- **Training runners = the textbook.** The model learns from them.
- **Validation runners = the mock test.** Used to choose settings, the alert threshold and calibration, and when to stop training.
- **Test runners = the final exam.** Used for the honest score. Peeking at the exam while studying makes the marks meaningless.

### Why split by person, not by row
Days from the same runner look alike. If runner #12's Monday is in training and Tuesday in the test, the model "recognises" runner #12 and the score is inflated. So each runner is **entirely** in training, validation, or test (D-002, rule 2).

### Lövdal split design (saved 2026-10-15 in data/splits/lovdal_folds.json; decision D-017)
```
74 runners
├── 10 newest runners (IDs 64-73)  →  LOCKED FINAL TEST, opened ONCE, after the evaluation freeze
└── 64 older runners (IDs 0-63)    →  DEVELOPMENT, 5 balanced groups
       each round: 1 group = test (12-14 runners), 9-10 runners = validation, the rest = training (41-43)
       → each of the 64 is tested exactly once
```
- **Why lock the 10 newest?**
  1. Nobody can say we tuned on the test.
  2. They are exactly the paper's test runners. We checked: in the day file, IDs 64-73 contain 50 injuries and 2,994 healthy events, the same as the paper. So the day-file comparison is like-for-like. Our combined model is scored on the days present in both files (2,994 events, 48 injuries), and it trains on 51 runners while the paper trained on 64: we say so. Also, the paper's 0.724 was measured with its label-reading scaling, so even on the same runners the methods differ.
  3. They joined last, so the test also asks "does it work on newer athletes?". It is NOT a "future" (prospective) test: it is a test on 10 different runners who joined later. Say "newer athletes", never "future data".
- **Main result vs final check (D-017, Naman chose option A).**
  - **Main result** = the average ± spread over the 5 development test groups: 64 runners, 533 injuries. More injuries make the number more precise.
  - **Final check** = the 10 locked runners (50 injuries), opened once at the very end. Their score is noisier, but nobody could have tuned on it.
  - Both are reported. If they disagree a lot, we say so.
  - **Not before the end, even for Review 2.** Review 2 shows development results and says the locked test is sealed.
  - **Pipeline check in piece 3** (rebuilding the paper's method): uses **stand-in runners 54-63** (trained on 0-53), never the locked runners.
- **Balance (re-verified):**
  - Injuries per test group: day 105 / 108 / 108 / 105 / 107; week 103 / 107 / 107 / 105 / 105.
  - Validation groups: 79-85 injuries each.
  - Every development runner is tested exactly once, and the locked runners never appear in development.
- **One grouping for both files.** Re-verification found that grouping the day file and the week file *separately* gives slightly different runner groups. Fusion needs the same runners in both arms, so the groups are built **once** (from the day file) and the same runner lists are used for the week file and for fusion.
- **Final model:** trained on 51 runners, validated on 13 (development group 0), tested once on the 10 locked runners.
- **Validation runners repeat across folds (we disclose it).** Each fold draws its validation group from its own non-test runners. Across the 5 folds, 46 validation places are filled by 30 runners; runners 20, 21, 33 and 37 validate three times. A test pins these numbers. Inside one fold, validation and test never overlap. But a runner can validate in one fold and be tested in another, so settings must be chosen inside each fold. One setting picked by averaging validation scores across folds would leak test runners (D-021).
- **Safety rules:**
  - seed 0 chosen in advance;
  - balance checked on label counts only (each group within ±20 % of the average). Honest note: we looked at the group counts before writing this rule, so it is a sanity check, not a pre-registered test;
  - the split script writes all files or none, and refuses to overwrite;
  - **sealed:** `data/splits/SHA256SUMS` stores a fingerprint of every split file; the tests fail if one byte changes, and the script refuses any further re-make;
  - **locked runners kept apart:** their arrays are in a separate folder and only `src/data/lockbox.py` opens them. It refuses unless the code is at the commit tagged `eval-freeze`, and it writes every opening to a log;
  - every loader checks that the raw files it reads match the fingerprints in `data/MANIFEST.csv`;
  - automatic tests: no runner in two roles, everyone tested once, locked runners never in development, files complete and matching the audited data;
  - an independent AI reviewer (eval-auditor) checks the splits.

### Other datasets (saved 2026-10-15; D-017, D-020)
| Dataset | Split | Why |
|---|---|---|
| PAMAP2 | 4 test groups of 2 subjects; validation = 1 person from the next group; subject 109 always in training | 109 has only ~85 s of protocol data |
| WESAD | 5 test groups of 3 subjects; validation = 2 people from the next group | balanced and simple |
| Sport DB 2.0 | 5 groups by subject; all 4 skating recordings kept together | their identity is unknown, so they might be one person |
| SoccerMon | leave-one-player-out, all predictions pooled into one score | pain reports come from few players (15 of 50; `daily_meta.json`), so fixed groups would get very different numbers of reports |
| PMData | leave-one-person-out, pooled; results reported with and without p12; p08 excluded from injury tests | p12 has 44 of the 72 pain-report days (`daily_meta.json`); p08's injury file is missing |
| SIRP-600 | 80/20 by rows, labelled "demo only" | no people to split by |

"Leave-one-person-out": train on everyone except one person, test on that person, repeat for every person, then score all predictions together. Best for very small groups of people.

**The one-time re-make (D-020).** The first PAMAP2 and WESAD files picked validation people badly: in WESAD, 10 validation slots were filled by only 5 people (S4 and S14 three times each). An independent review caught it. Before any model was trained, Naman approved re-making only those 2 files, once:
- the test groups stayed exactly the same (the code refuses otherwise);
- validation now rotates: fold k validates on people from test group k+1, so nobody validates twice (PAMAP2: 4 different people, WESAD: 10);
- the old files are kept in `data/splits/superseded/`, and the decision is logged;
- then the folder was sealed, so it can never happen again.

Why this is not cheating: the rule "never regenerate splits" stops people from reshuffling until the *results* look good. No results existed yet, and the test groups did not change.

### Other leakage rules
- Scaling is learned from training runners only (rule 3). The paper scaled test runners using their own non-injured days, which reads test labels. Our comparisons won't (D-021, D-023): their exact method is run only once, as a labelled check on stand-in runners 54-63, and every comparison uses a version with one scaler learned from training runners.
- **No per-runner scaling at all (D-023).** In this dataset, simply counting a runner's events in the previous 3 weeks finds injuries almost perfectly (ROC-AUC 0.9946, `data/processed/lovdal/event_design_check.json`). That's because the authors kept every injury at least 22 days away from other events, while healthy events follow each other day after day. It's an artefact of how the data was built, not knowledge about the body. So anything based on a runner's own history (scaling, counts, gaps, dates) is banned, and a test checks it.
- **Tuning happens inside each round (D-021).** Each round picks its settings, alert level and stopping point using only its own practice runners. One setting picked by averaging practice scores over all rounds would use exam runners, because runner 20 is practice in rounds 0, 2 and 4 and exam in round 1.
- **Every single model gets the same number of tuning tries (8 per round)**: each baseline and each candidate arm. The combined model also chooses among several arms, so its total budget is larger. That total is written down before piece 3 runs and reported next to its score.
- **The final model is named in advance:** the fused model (piece 7). We never send whichever model scored best on the rounds to the locked test, because picking the winner on test data inflates its score.
- The alert threshold is chosen on validation runners only (rule 5).
- Synthetic data never enters validation or test (rule 3).
- A result that looks too good is treated as a bug until proven otherwise.

---

## 6. How we score models (metrics)

### The confusion matrix
| | Really injured | Really healthy |
|---|---|---|
| **Model says "injury"** | True Positive (TP) | False Positive (FP): false alarm |
| **Model says "healthy"** | False Negative (FN): missed injury | True Negative (TN) |

### Why accuracy is useless here
Accuracy = (TP + TN) / all. Only 1.36 % of day windows are injuries, so a model that **always says "healthy"** gets 98.6 % accuracy while catching zero injuries.

### Scores that matter
- **Sensitivity (recall)** = TP / (TP + FN): share of real injuries caught.
- **Specificity** = TN / (TN + FP): share of healthy windows correctly left alone.
- **Precision** = TP / (TP + FP): share of alarms that were real.
- **F1** = 2 × precision × recall / (precision + recall): one number balancing both.
- **ROC curve:** sensitivity vs false-alarm rate as the threshold slides from 0 to 1.
- **ROC-AUC:** pick one injury window and one healthy window at random; AUC = the chance the injury window gets the higher risk. 0.5 = coin flip, 1.0 = perfect. The paper: 0.724 (day file).
- **PR-AUC:** area under the precision-recall curve; focuses on the rare injuries. Random ≈ the injury rate (0.0136), so it shows how far above chance we are.
- **Calibration:** among windows given "70 %", about 70 % should really be injuries. Checked with a reliability plot and the **Brier score** = average of (predicted probability − real label)²; lower is better.

**Practice example (made-up numbers).** 1,000 windows, 14 real injuries. The model raises 100 alarms, 7 of them real.
- Sensitivity = 7 / 14 = 50 %.
- Specificity = (986 − 93) / 986 = 90.6 %.
- Precision = 7 / 100 = 7 %. That's 5 times better than random alarms, which would be right 1.4 % of the time.

**Which metrics we report (fixed by D-022, before any model runs):**
- **Main score: PR-AUC**, computed as "average precision". It needs no alert level and is built for rare events. A model that guesses randomly gets about the injury rate (about 0.013), so we always show that next to it.
- **ROC-AUC** next to it, because the paper reports ROC-AUC (0.724).
- **At the alert level:** injuries caught, healthy days left alone, share of alerts that are real, **false alarms per caught injury** and **alerts per 100 scored events** (the rows are selected events, not every day), plus MCC.
- **Calibration:** Brier score, ECE with 10 equal-count bins, and calibration slope and intercept.
- **Extra (exploratory): a decision curve.** It shows whether using the model does more good than harm, over risk levels from 0.5 % to 10 %.
- **Not used:** accuracy, or a fixed-precision target. Precision depends on how common injuries are, so a target such as "5 % of alerts real" behaves differently in every group.

### Threshold
The model outputs a risk (e.g. 0.37); the dashboard needs "alert" or "no alert". A lower threshold catches more injuries but gives more false alarms. The paper chose the threshold where sensitivity ≈ specificity on validation data (their notebook, line 86). We use the same rule, but inside each round on that round's own practice runners (D-022). The paper drew its validation days from the same runners it trained on (notebook lines 308-311); ours are always different people.

### Reporting honestly
- The 3 seeds are averaged inside each group first; then report **mean ± spread** over the 5 groups (e.g. "0.70 ± 0.04": example only).
- The locked test has only 50 injuries, so it gets an uncertainty range. It is the final check; the main number is the 5-group development average (D-017).

---

## 7. How a model learns (the training loop)
1. Start with random weights.
2. **Forward pass:** feed a batch; get predictions.
3. **Loss:** compare with true labels (binary cross-entropy for yes/no).
4. **Backpropagation:** the chain rule computes each weight's share of the error (its gradient).
5. **Gradient descent:** nudge every weight against its gradient; the step size is the **learning rate**.
6. Repeat over all batches = one **epoch**; do many epochs.
7. **Early stopping:** stop when the validation loss stops improving (that's where overfitting starts).

**Learning rate intuition:** walking down a foggy hill. Steps too big and you jump over the valley; too small and it takes forever.

**Class imbalance during training (training data only):** class weights (missing an injury costs more), or balanced sampling (like the paper's 2,048 + 2,048 bags). Never rebalance the test set.

---

## 8. The models in our project, one by one

### Baselines 🔜 (piece 3): the yardsticks
| Model | How it works | Why include it |
|---|---|---|
| Majority class | Always "no injury" | The floor; shows accuracy is useless |
| ACWR-style rule (3-week) | Acute:chronic workload ratio = this week's load ÷ the usual load; alert on big jumps. Adapted to 3 weeks (D-015) | The rule coaches already use; ML must beat it |
| Logistic regression | Weighted sum of features → sigmoid → probability | Simplest model; tells us if complexity helps |
| Random Forest | Many decision trees on random samples of rows and features; they vote | Robust on small tables; TreeSHAP works on it |
| CatBoost | Gradient boosting: trees added one by one, each fixing the previous ones' mistakes | Usually best on small tables |
| GRU | Small recurrent network reading the 7-day sequence | Tests whether the timeline beats the snapshot |
| Paper's method, rebuilt | 9 XGBoost models on balanced samples, averaged, Platt-calibrated | Fair comparison with the published result |

**Decision tree example:** "Was total km last week > 80? → yes → ≥ 2 rest days? → no → high risk." Each question splits the data into "purer" groups (mostly injury or mostly healthy), measured with Gini impurity or entropy.

**Bagging vs boosting:**
- **Bagging** (Random Forest, the paper's 9 models): models trained in parallel on random samples, answers averaged. This reduces variance (wobbliness).
- **Boosting** (CatBoost, XGBoost): models trained one after another, each focusing on the previous ones' errors. This reduces bias (systematic error).

### Sequence arm 🔜 (Review 1 commitment): LSTM, GRU, TCN
- **RNN:** reads day 1 … day 7 one at a time, keeping a memory (hidden state). Plain RNNs forget early days during training (vanishing gradient).
- **LSTM:** adds a long-term memory lane (the cell state) and three gates (learned switches between 0 and 1):
  - forget gate: what to erase;
  - input gate: what to store;
  - output gate: what to show.
- **GRU:** a lighter LSTM with two gates (update, reset) and no separate cell state, so fewer parameters, which suits small data. Example: a PyTorch GRU with 10 inputs and 32 hidden units has 3 × 32 × (10 + 32) + 6 × 32 = **4,224 parameters** (VGG16 has ~138 million).
- **TCN:** convolutions over time instead of recurrence, built from:
  - **causal** convolutions (only past days);
  - **dilations** (skip steps to see further back);
  - **residual blocks** (the ResNet idea, section 9).

  It's stable and fast to train. It's first to be cut if time is short.
- **Why not BiLSTM?** A BiLSTM reads the sequence forwards and backwards. It wouldn't cheat here (the 7-day window is all in the past), but:
  - it doesn't fit live streaming (it must re-read the whole window every new day);
  - it doubles the parameters on small data;
  - it gains little on 7 steps;
  - its explanations are muddier;
  - it wasn't a Review 1 commitment.

  It could be added later as a one-row comparison.
- **Why not Transformers?** They need much more data; with 74 runners they would memorise.

### Snapshot arm 🔜: Random Forest and CatBoost
Trained on Lövdal's **week file**: one row of 3-week summary numbers per runner per event day. Same runners and days as the sequence arm, so the two can be fused (D-013).

---

## 9. CNNs from your teacher's slides, and how they connect to our project

The teacher showed VGG16, GoogLeNet/Inception v1 and a ResNet video ("Residual Networks (ResNet) Explained Intuitively", ByteQuest). These are image models; our data isn't images, but the same building blocks appear in our TCN and sensor models. **The teacher may ask about any deep-learning topic, so Part 2 also covers other architectures.**

### 9.1 Convolution layer
- An image is a grid of numbers: height × width × channels (3 = red, green, blue); 224 × 224 × 3 for VGG16.
- A **filter (kernel)** is a small grid of learned weights, e.g. 3 × 3 × 3. It **slides** over the image, multiplying overlapping numbers and adding them up at each position → one number per position. All positions together = a **feature map**.
- One filter detects one pattern (an edge, a corner). 64 filters → 64 feature maps (64 output channels).
- Early layers find simple things (edges); deeper layers combine them into complex things (eyes, wheels, faces), as in the "eyes, ears, wheels, faces, object parts" list on your teacher's slide.
- **Stride (S):** jump size. **Padding (P):** zeros added around the border.
- **Output size:** `out = (W − K + 2P) / S + 1`. VGG16 conv: (224 − 3 + 2)/1 + 1 = **224** (size kept).
- **Parameters:** `(K × K × C_in + 1) × C_out` (+1 = bias).
  - VGG16 first conv: (3 × 3 × 3 + 1) × 64 = **1,792**.
  - Second conv: (3 × 3 × 64 + 1) × 64 = **36,928**.
- **Receptive field:** how much of the original input one output number "sees". It grows layer by layer: two stacked 3 × 3 convs see 5 × 5, three see 7 × 7.

### 9.2 Activation (ReLU)
`ReLU(x) = max(0, x)`: negatives become 0. It adds non-linearity (without it, many layers collapse into one linear formula), is cheap, and helps against vanishing gradients.

### 9.3 Pooling
Shrinks feature maps: **max pooling** keeps the largest value in each window, **average pooling** the average. VGG16 uses 2 × 2 windows with stride 2 (original paper; some diagrams draw it differently): (224 − 2)/2 + 1 = **112**. Less computation, a little tolerance to small shifts, no learnable parameters.

### 9.4 Fully connected layers and softmax
- **Flatten** the last feature maps (7 × 7 × 512 = 25,088 numbers).
- **Fully connected (FC) layers** connect every input to every output.
- **Softmax** turns the final scores into probabilities that sum to 1 (1,000 ImageNet classes).

### 9.5 VGG16 (Simonyan & Zisserman, 2014)
- Only 3 × 3 convolutions (stride 1, padding 1), stacked deep, with 2 × 2 max-pooling between blocks.
- **16 learnable layers:** 13 convolution + 3 fully connected.

| Block | Conv layers | Filters | Size after the block's pool |
|---|---|---|---|
| 1 | 2 | 64 | 224 → 112 |
| 2 | 2 | 128 | 112 → 56 |
| 3 | 3 | 256 | 56 → 28 |
| 4 | 3 | 512 | 28 → 14 |
| 5 | 3 | 512 | 14 → 7 |

- **Image size decreases while channels increase:** the network trades "where" for "what".
- **End:** 7 × 7 × 512 → FC 4,096 → FC 4,096 → FC 1,000 → softmax. Trained on ImageNet (1,000 classes).
- **~138 million parameters**, mostly in the first FC layer: 25,088 × 4,096 + 4,096 = 102,764,544 (~74 %).
- **Why stack 3 × 3s?** Same receptive field as a bigger filter, fewer parameters, more ReLUs (more non-linearity).
- **Weakness:** huge, slow, memory-hungry.

### 9.6 GoogLeNet / Inception v1 (Szegedy et al., 2014)
- **Inception module:** parallel branches (1 × 1, 3 × 3 and 5 × 5 convolutions, plus a 3 × 3 max-pool), their outputs **concatenated** along channels. The network looks at several scales at once.
- **1 × 1 convolutions** cheaply reduce channels before the expensive 3 × 3 and 5 × 5 filters (a "bottleneck").
- **Global average pooling** at the end replaces most of the huge FC layers.
- **Auxiliary classifiers:** two small side outputs used only in training, giving extra gradient signal against vanishing gradients.
- **~22 learnable layers but only ~5-7 million parameters** (the slide says ~6.8 M), against VGG16's ~138 M.

| Feature | VGG16 | GoogLeNet / Inception v1 |
|---|---|---|
| Basic idea | Sequential convolutions | Parallel multi-scale branches |
| Main convolution | 3 × 3 | 1 × 1, 3 × 3, 5 × 5 |
| 1 × 1 convolution | Not central | Very important |
| Pooling | Separate layer between blocks | Also a branch inside Inception |
| FC layers | 3 large | Mostly replaced by GAP |
| Parameters | ~138 M | ~5-7 M |
| Depth | 16 learnable layers | ~22 learnable layers |

### 9.7 ResNet (He et al., 2015): the video
- **Problem:** beyond some depth, plain networks get *worse* training error ("degradation"); gradients fade or distort through many layers.
- **Fix: skip (residual) connections.** A block computes `output = F(x) + x`, so it only learns the *change* F(x). If a layer isn't useful it can learn F(x) ≈ 0 and just pass x through.
- **Why it works:** the "+ x" path is a gradient highway during backpropagation, so even 152 layers train well. When shapes differ, the shortcut uses a 1 × 1 convolution (projection).
- **Variants:**
  - ResNet-18/34: "basic" blocks (two 3 × 3 convs);
  - ResNet-50/101/152: "bottleneck" blocks (1 × 1 → 3 × 3 → 1 × 1).

  All use batch normalisation. ResNet-50 has ~25.6 million parameters; ResNet won ImageNet 2015.

### 9.8 How these ideas connect to OUR project
| Image idea | Where it appears for us |
|---|---|
| Convolution | **1D convolution** over time: the kernel slides along days (Lövdal) or sensor readings (PAMAP2/WESAD windows) |
| Kernel, stride, padding | Same meaning in 1D. Kernel 3 over 7 days with no padding → (7 − 3)/1 + 1 = 5 outputs; each filter over 10 features has 3 × 10 + 1 = 31 parameters |
| Dilation | TCN skips steps: kernel 3 with dilations 1, 2, 4 sees 1 + 2 × (1 + 2 + 4) = 15 steps, more than the 7-day window |
| Residual connection | **The TCN is built from residual blocks** (`F(x) + x`) |
| Global average pooling | Summarise a sequence over time before the output |
| Feature maps | 1D Grad-CAM (section 13) turns them into "which days mattered" |
| Softmax | We predict yes/no, so we use one **sigmoid** output |
| Small CNN | 🔜 Piece 3: a small 1D CNN classifying activities from PAMAP2 windows |

**Why not use VGG16 / ResNet-50 / Inception directly?** They're built for big images and millions of training examples; our data is small sequences (7 days × 10 numbers, 74 runners). A 138-million-parameter network would memorise our runners, and ImageNet pretraining (cats, cars) doesn't transfer to training logs. So we use their *ideas* (convolution, pooling, residual connections) in small 1D models sized for our data.

---

## 10. Fusion: combining two models into one risk score 🔜 (piece 7)
- **Two arms, same runner and same day:** the sequence arm reads the 7-day timeline (day file); the snapshot arm reads the 3-week summary (week file).
- **Stacking / blending:** a tiny combiner model (e.g. logistic regression) takes both arms' predictions and learns how much to trust each. It's trained on the **validation runners'** predictions, runners neither arm trained on.
- **Calibration:** Platt scaling (a fitted sigmoid) or isotonic regression on validation runners.
- **Risk score:** calibrated probability × 100.
- **Why late fusion?** Each arm stays separately testable and explainable, and it works with small data.
- Uses the 42,680 days present in both files (D-014).

---

## 11. Generators: realistic synthetic data 🔜 (piece 5)

**Why we need them:** injuries are rare (583 windows), and no public dataset links rich sensor streams to many real injuries. Synthetic data can give extra *training* examples.

**GAN:** two networks play a game.
- The **generator** turns random noise into fake data.
- The **discriminator** tries to tell real from fake.
- They improve together, like a forger and a detective, until the detective is right only about 50 % of the time.
- Known problems: unstable training, and **mode collapse** (only a few kinds of fakes).

| Generator | Key idea | Why it's here |
|---|---|---|
| TimeGAN | A GAN for sequences: also learns an embedding and a "next step" loss, so the *order over time* is realistic | Review 1 commitment |
| WGAN-GP | Uses the Wasserstein distance ("how much earth to move to turn fake into real") + a gradient penalty: much more stable | Review 1 commitment |
| TimeVAE | Variational autoencoder: an encoder compresses data to a small random code, a decoder rebuilds it; sample new codes to generate | A non-GAN comparison (D-003) |
| Classic augmentation | Jitter, scaling, time-warping | The cheap baseline to beat |

**How generators are judged:**
- **Fidelity:** can a classifier tell real from synthetic (≈ 50 % = realistic)?
- **Usefulness:** train on synthetic, test on real (TSTR); and does adding synthetic data to training improve real test scores?
- **Privacy:** no near-copies of real people.

**Strict rules:**
- Generators train on training runners only (one per training group).
- Synthetic data never goes into validation or test, and is labelled synthetic everywhere.
- A synthetic "athlete" that combines sensor data and injury data is a **demo only** (D-001), because those come from different people.
- If GANs don't help, we say so.

---

## 12. AIoT: edge, fog and cloud tiers 🔜 (piece 6)

| Tier | Device | Our layer | Example job |
|---|---|---|---|
| Edge | Watch / sensor | Reflex | Instant checks on raw readings (e.g. abnormally high heart rate), filtering, compressing |
| Fog | Phone / gateway | Supervisor | Session summaries, short-term trends |
| Cloud | Server | Deep | Daily risk model, explanations, dashboard, training |

- **Why tiers?** Sending every raw reading to the cloud costs bandwidth, battery and time. Simple work near the sensor gives faster alerts and sends far fewer bytes.
- **Measured:** latency (ms from reading to alert) and bytes sent, tiered vs cloud-only.
- **How:** three separate Python programs passing messages (D-008).
- **Data:** PAMAP2 (and Sport DB) replayed as if live. The injury risk comes from Lövdal logs, a different set of people, so the demo shows them as **two separate tracks**.

---

## 13. Explainability: why the model says "risky" 🔜 (piece 8)

| Method | Used for | Simple idea |
|---|---|---|
| SHAP / TreeSHAP | RF, CatBoost | Game theory: each feature gets its fair share of the prediction, like splitting a team prize by contribution. TreeSHAP is exact and fast for trees |
| Integrated gradients | GRU / LSTM / TCN | Morph a "blank" input into the real one step by step and add up how the prediction changes; the importances add up to the prediction |
| Attention weights | Sequence models | Which days the model looked at most. **Never shown alone** (rule 12) |
| 1D Grad-CAM | TCN | Gradients of the risk w.r.t. the last convolution's feature maps → important days |
| Deletion test | All | Remove the "most important" features: the prediction should change more than when removing random ones. Proves faithfulness |
| Known risk factors | All | Does the model flag what sports science knows matters, e.g. sudden jumps in weekly mileage? |

**For coaches:** "Risk 72/100, mainly because weekly distance jumped 40 % and recovery scores dropped" (example wording; real numbers come from the explanation method).

From the paper: individual features correlate only weakly with injury (r between −0.04 and 0.07). Injuries come from **combinations** of factors, which is why models and explanations are needed.

---

## 14. The coach dashboard and the Training lab

### Dashboard 🔜 (piece 9)
- **Streamlit pages:** athlete list with risk scores, athlete detail with plain-language reasons, system view (tiers, latency, bytes), alert log.
- `python tasks.py demo` replays a held-out athlete through the tiers into the dashboard.

### Training lab ✅ approved design (D-016), 🔜 to build in piece 9
- **For when the examiner says** "change a hyperparameter and retrain".
- **How it works:** pick a **fast** model (logistic regression, Random Forest, CatBoost, small GRU), move sliders inside **safe ranges**, press Train. It trains on one development group and shows the validation score, the ROC curve and the time taken. It uses the GPU if available, otherwise the CPU.
- **Rough times (to be measured):** seconds for LR and RF; tens of seconds for CatBoost; about 1-3 minutes for a small GRU.
- **Never touches the real model:**
  - each run is saved in its own folder, `experiments/training_lab/<date-time>/`, labelled "demo";
  - it never overwrites the official tested models or their results;
  - it never uses the locked final-test runners.
- **Safe by design:** inputs limited to sensible ranges, and errors caught and shown as messages instead of crashing.
- **GANs** take too long for live retraining, so the lab shows their saved results.

---

## 15. The Lövdal paper in 2 minutes (verified against the full text)
- **Citation:** Lövdal, Den Hartigh, Azzopardi. "Injury Prediction in Competitive Runners With Machine Learning." *Int J Sports Physiol Perform* 2021;16(10):1522-1531. doi 10.1123/ijspp.2020-0518.
- **Method:** bagged XGBoost: 9 models, each trained on 2,048 injury + 2,048 healthy samples drawn with replacement, averaged, Platt-calibrated. Learning rate 0.01, depth 2 or 3, 256 or 512 trees.
- **Test:** the 10 runners who joined most recently, trained on the first 64; 5 repeated experiments.
- **Results (Table 3):**
  - day: AUC 0.724, sensitivity 58.4 %, specificity 74.1 % (threshold 0.448);
  - week: AUC 0.678, sensitivity 50.4 %, specificity 74.6 % (threshold 0.476).

  About 30 % of injuries can be caught at a 10 % false-alarm rate (day).
- **Most important features:**
  - day: high-intensity km and perceived training success, recovery and exertion;
  - week: training success 2-3 weeks before, the longest single-day run, rest days, week-to-week mileage change.
- **Weak points (say them politely):**
  - test runners were scaled using their own non-injured days (that uses test labels);
  - validation data came from the training runners;
  - their Figure 4 reads the day order the way their later README says is reversed;
  - healthy days near injuries were removed, which makes the task easier than real life.

---

## 16. Limitations we state openly
1. Healthy days within 3 weeks of an injury aren't in the Lövdal files, so no model is tested on the tricky "almost injured" days. Real-life scores may be lower.
2. 74 runners from one team: results may not transfer to other teams or sports.
3. Injury = "couldn't complete a session" (self-logged), not a medical diagnosis.
4. The locked final test is small (10 runners, 50 injuries): we report an uncertainty range.
5. Sensor datasets are different people from the runners. SoccerMon and PMData give only small, self-reported, exploratory links.
6. Synthetic data can't prove anything about real bodies or injuries.
7. The tiers are simulated software processes, not real devices.
8. Gender balance: Lövdal 27 women / 47 men; WESAD 3 / 12; PMData 3 / 13. Results may be less reliable for women.

---

## 17. Tools, files and commands

| Tool | What it is | Why |
|---|---|---|
| Python 3.12 + `.venv` | Language + private package folder | Same versions every time |
| PyTorch + CUDA | Neural-network library + GPU | Train GRU/LSTM/TCN/GANs on the RTX 4050 |
| pandas, numpy, scipy | Tables, arrays, statistics | Data handling, audit checks |
| scikit-learn | Classic ML + splitting + metrics | Baselines, group splits |
| CatBoost, XGBoost 🔜 | Gradient boosting libraries | Snapshot arm; the paper's method |
| SHAP, Captum | Explanation libraries | TreeSHAP; integrated gradients |
| Streamlit | Python web apps | Dashboard |
| pytest, ruff | Tests; code checker | `python tasks.py check` |
| Git + GitHub | Version history + online backup | Every change saved with a message |
| VS Code | Code editor | Run and read everything |

| Command | What it does | Status |
|---|---|---|
| `python tasks.py check` | Code checks + all tests | ✅ |
| `python tasks.py audit` | Audit all datasets (~1.5-2 min) | ✅ |
| `python tasks.py audit --only kaggle_runners` | Audit one dataset (seconds) | ✅ |
| `python tasks.py snapshot` | Zip a backup of the project | ✅ |
| `python tasks.py data / baselines / gen / pipeline / train / explain / demo / paper-figures` | Later pieces | 🔜 (print "not implemented") |

**Folders:**
- `docs/`: explanations, decisions, this guide;
- `src/data/`: readers, audit, parallel;
- `configs/`: settings;
- `tests/`;
- `experiments/`: every run's results;
- `data/raw/`: datasets (laptop only, never on GitHub);
- `data/splits/` 🔜: saved splits.

---

## 18. What you can demonstrate today
1. `python tasks.py check` → "23 passed". Explain the test that proves the WESAD loader refuses a pickle trying to run a command.
2. `python tasks.py audit --only kaggle_runners` → open the newest `experiments/audit/` folder and `audit_table.md`. It also creates a run folder and updates `data/MANIFEST.csv`; discard those in VS Code Source Control if you were only demonstrating.
3. Kaggle = Lövdal: the same fingerprint (bf5e4401…) in `run_info.json`.
4. A "fake data" check: in `audit.json`, ziya07 resting 120.6 vs running 118.0, next to PAMAP2's 156.6 vs 75.5.
5. `docs/DECISIONS.md`: D-001 to D-016, with who decided.
6. The split design (section 5), and why the 10 newest runners are locked.

---

## 19. How to show the project is yours (and be honest about AI)
- **You made the decisions:** `docs/DECISIONS.md` lists who decided each one. Know *why* for each.
- **Run everything yourself** before the review (section 18).
- **Know your numbers** (section 34) and limitations (section 16).
- **Be honest about AI help:** the paper's AI-use statement comes from `docs/AI_LOG.md`. A good answer: *"Yes, I used an AI coding assistant, and it's declared in the paper. I made the decisions, checked every result against the data, and every number comes from code I can run right now."*
- **Practise explaining** each section to a friend. If you can teach it, you own it.
- **Exercises:**
  - compute VGG16's sizes yourself;
  - redo the confusion-matrix example with your own numbers;
  - explain the split diagram from memory.

---

# PART 2: GENERAL FOUNDATIONS (the teacher may ask anything)

## 20. AI, ML, DL and the types of learning
- **Artificial Intelligence (AI):** any technique that makes computers act "smart".
- **Machine Learning (ML):** AI that learns patterns from data instead of hand-written rules.
- **Deep Learning (DL):** ML with neural networks of many layers.
- **Supervised learning:** examples come with answers (labels). Our injury model.
  - *Classification:* predict a category (injury yes/no).
  - *Regression:* predict a number (e.g. tomorrow's heart rate).
- **Unsupervised learning:** no answers; find structure, e.g. clustering (k-means) or compression (PCA, autoencoders).
- **Self-supervised learning:** the data provides its own task (predict the next step, fill in a hidden part).
- **Reinforcement learning:** an agent learns by trial and error with rewards (games, robots). Not used in our project.
- **Generative models:** learn to create new data like the training data (GANs, VAEs, diffusion models).

## 21. Preparing data
- **Cleaning:** remove impossible values and duplicates, and fix types (e.g. dates stored as text).
- **Missing values:** drop rows, fill with the mean/median, or use a special code. Lövdal uses −0.01 for rest days, which is a code, not a measurement.
- **Outliers:** extreme values; check whether they're errors or real.
- **Scaling:**
  - *Normalisation (min-max):* x' = (x − min) / (max − min) → range 0-1.
  - *Standardisation (z-score):* z = (x − mean) / std → mean 0, std 1. Example: mean 10 km, std 4 km; a 18 km day → z = 2.
  - **Always fit scaling on training data only**, then apply it to validation and test.
- **Encoding categories:** one-hot encoding turns "walking/running/cycling" into separate 0/1 columns.
- **Feature engineering:** creating useful features from raw data (e.g. weekly load change).
- **Data splits:** train / validation / test (section 5).
- **Cross-validation:**
  - *k-fold:* split into k groups and test on each once;
  - *stratified:* keeps the label rate similar in each group;
  - *group k-fold:* keeps all data of one person together, which is what we use.
- **Data leakage:** any way test information reaches training (section 5).

## 22. Imbalanced data (our biggest data challenge)
| Method | Idea | Note |
|---|---|---|
| Class weights | Errors on the rare class cost more in the loss | Simple, no new data |
| Random oversampling | Repeat rare examples | Can overfit to the copies |
| SMOTE | Create new rare-class points between existing ones and their neighbours | For tabular data; never on test data |
| Undersampling | Drop some common-class examples | Loses information |
| Balanced bagging | Many models, each on a balanced sample (the paper's method) | Uses all data overall |
| GAN augmentation | Generate realistic rare examples | Our piece 5, training only |
| Threshold moving | Lower the alert threshold | Chosen on validation data |
| Right metrics | ROC-AUC, PR-AUC, sensitivity, not accuracy | Section 6 |

## 23. Neural networks from zero
- **Neuron:** `output = activation(w1·x1 + w2·x2 + … + b)`. Weights `w` and bias `b` are learned.
- **Layers:** input layer → hidden layers → output layer. "Deep" = many hidden layers.
- **MLP (multi-layer perceptron):** layers fully connected one after another.

**Activation functions:**

| Name | Formula / shape | Output range | Used for |
|---|---|---|---|
| Sigmoid | 1 / (1 + e^(−x)) | 0 to 1 | Yes/no output (our risk) |
| Tanh | (e^x − e^(−x)) / (e^x + e^(−x)) | −1 to 1 | Inside LSTM/GRU |
| ReLU | max(0, x) | 0 to ∞ | Hidden layers of CNNs/MLPs |
| Leaky ReLU | x if x > 0, else 0.01·x | −∞ to ∞ | Avoids "dead" ReLUs |
| Softmax | e^(xi) / Σ e^(xj) | 0 to 1, sums to 1 | Multi-class output |

**Loss functions:**
- MSE (mean squared error): average of (prediction − truth)², for regression.
- Binary cross-entropy: −[y·log(p) + (1 − y)·log(1 − p)], for yes/no. Big penalty for confident wrong answers.
- Categorical cross-entropy: for multi-class with softmax.

**Optimisers:**
- **Gradient descent:** update w ← w − learning rate × gradient.
- **SGD (stochastic gradient descent):** uses one example or one mini-batch at a time. Faster, a bit noisy.
- **Momentum:** keeps some of the previous step's direction (like a ball rolling downhill).
- **Adam:** adapts the step size for each weight automatically; the usual default.
- **Learning-rate schedule:** lower the learning rate as training goes on.

**Backpropagation:** the chain rule from calculus, applied backwards from the loss through every layer, to find each weight's gradient.

**Weight initialisation:** start weights at sensible random sizes. Xavier/Glorot suits sigmoid/tanh; He initialisation suits ReLU. Bad starts cause vanishing or exploding signals.

## 24. Problems in training, and their fixes
| Problem | What happens | Fixes |
|---|---|---|
| Overfitting | Great on training, bad on new data | More data, smaller model, dropout, L1/L2, early stopping, augmentation |
| Underfitting | Bad even on training | Bigger model, more features, train longer |
| Vanishing gradients | Early layers stop learning in deep/recurrent nets | ReLU, residual connections, LSTM/GRU gates, batch norm, good initialisation |
| Exploding gradients | Weights blow up | Gradient clipping, a lower learning rate, good initialisation |
| Slow/unstable training | Loss jumps around | Tune the learning rate, batch norm, Adam |

- **Bias-variance trade-off:** simple models have high bias (underfit); complex models have high variance (overfit). We look for the balance.
- **Dropout:** randomly turn off some neurons during training so the network can't depend on any single one.
- **L1 / L2 regularisation:** add a penalty for big weights to the loss. L1 can push weights to exactly zero; L2 ("weight decay") keeps them small.
- **Batch normalisation:** rescale each layer's outputs within a batch, giving faster, more stable training.
- **Hyperparameter tuning:**
  - grid search (try every combination);
  - random search (try random combinations; often better for the same time);
  - Bayesian optimisation (e.g. Optuna, which chooses the next try from past results).

  Always judged on **validation**, never test.
- **Transfer learning:** take a model pretrained on big data (e.g. ResNet on ImageNet), keep its early layers ("freeze"), and retrain the last layers on your data. Great for images, but there's no suitable pretrained model for running logs, so we don't use it.
- **Data augmentation:** images get flips, crops, rotations; time series get jitter, scaling, time-warping.

## 25. Famous architectures (a timeline you can quote)
| Year | Model | Key idea |
|---|---|---|
| 1998 | LeNet-5 | Early CNN for handwritten digits |
| 2012 | AlexNet | Deep CNN on GPUs; ReLU and dropout; won ImageNet 2012 |
| 2014 | VGG16/19 | Only 3 × 3 convs, very deep, ~138 M parameters |
| 2014 | GoogLeNet / Inception | Parallel multi-scale branches, 1 × 1 convs, global average pooling |
| 2015 | ResNet | Residual / skip connections, 152 layers |
| 2017 | DenseNet | Each layer receives the feature maps of all earlier layers |
| 2017 | MobileNet | Depthwise separable convolutions: small and fast, made for phones and **edge devices** |
| 2019 | EfficientNet | Scales depth, width and image size together |
| 2017 | Transformer | Self-attention instead of recurrence |

- **Transformer / attention, simply:** each element asks "which other elements matter to me?", using queries, keys and values. Attention weights mix information from everywhere at once. Powerful, but data-hungry (that's why not for us).
- **Autoencoder:** an encoder compresses input to a small code and a decoder rebuilds it; used for compression and anomaly detection.
- **VAE:** an autoencoder whose code is random, so new samples can be generated (TimeVAE is one).
- **Diffusion models:** learn to remove noise step by step; they generate very realistic images, but they're slow and heavy (future work for us).
- **Depthwise separable convolution (MobileNet):** split a normal convolution into a per-channel filter + a 1 × 1 mix. Many times fewer computations, which matters on watches and phones (the edge tier).

## 26. Classic ML models (beyond the ones we use)
| Model | Idea in one line |
|---|---|
| Linear regression | Best straight-line fit for predicting a number |
| Logistic regression | Linear score → sigmoid → probability |
| k-Nearest Neighbours | Predict like the k most similar training examples |
| Naive Bayes | Probability rules, assuming features are independent |
| SVM | Find the widest-margin boundary between classes; kernels allow curved boundaries |
| Decision tree | A flowchart of yes/no questions (Gini / entropy to choose questions) |
| Random Forest | Many random trees voting (bagging) |
| Gradient boosting (XGBoost, LightGBM, CatBoost) | Trees added one by one to fix errors |
| k-means (unsupervised) | Group points into k clusters around centres |
| PCA (unsupervised) | Compress many features into a few directions that keep most of the variation |

## 27. Evaluation and statistics basics
- **Formulas:** accuracy, precision, recall, specificity, F1 (section 6).
- **Mean / median / standard deviation:** centre and spread. The median resists outliers.
- **Pearson correlation r:** −1 to +1, the strength of a straight-line relationship. Example: Sport DB height-weight r = 0.686.
- **p-value:** how surprising the data would be if "no effect" were true; small (e.g. < 0.05) means an effect is likely. We used a t-test for the heart-rate checks.
- **Kolmogorov-Smirnov (KS) test:** checks whether values follow a given distribution. We used it to check whether SIRP-600 columns look "evenly spread" (a sign of generated data).
- **Confidence interval:** a range likely to contain the true value.
- **Bootstrap:** resample the test results many times to estimate the uncertainty. 🔜 We'll use it for the locked test.
- **Statistical vs practical significance:** with huge data, tiny differences become "significant" but may not matter.

## 28. Time-series and signal basics
- **Sampling rate (Hz):** readings per second (PAMAP2 IMU 100 Hz; WESAD chest 700 Hz).
- **Nyquist rule:** to capture a signal you must sample at least twice its fastest frequency. Human movement is mostly slow, so the data contract's 25 Hz is enough for our windows.
- **Resampling / interpolation:** change the rate, filling in-between values. Heart rate at ~9 Hz upsampled to 25 Hz is interpolated, not measured; say so.
- **Windowing:** cut a long signal into fixed pieces. Data contract: 128 steps (~5.1 s at 25 Hz), stride 64 (50 % overlap).
- **Filtering:** remove noise (low-pass) or drift (high-pass).
- **Sensors:**
  - accelerometer (movement);
  - gyroscope (rotation);
  - magnetometer (direction);
  - an IMU combines the three;
  - ECG (electrical heartbeat, chest);
  - PPG / BVP (optical blood flow, wrist; used by Fitbit and Empatica);
  - EDA (skin sweat, linked to stress).

## 29. IoT and edge-AI basics
- **IoT:** everyday devices with sensors that send data over a network.
- **Communication:**
  - Bluetooth Low Energy (watch ↔ phone, low power);
  - Wi-Fi / 4G (phone ↔ cloud);
  - **MQTT** (a light publish/subscribe protocol through a broker, popular in IoT);
  - HTTP/REST (request/response web APIs).

  We simulate the links with Python processes (D-008).
- **Why edge computing:** low latency (instant alerts), less bandwidth, works offline, more privacy (raw data stays local), saves battery on radio use.
- **Edge-AI model compression:**
  - quantisation (e.g. 32-bit → 8-bit numbers);
  - pruning (remove unimportant weights);
  - knowledge distillation (a small "student" model learns from a big "teacher");
  - efficient architectures (MobileNet).
- **Trade-offs:** device power and memory vs model size; latency vs accuracy.

## 30. Explainability and ethics
- **Global vs local explanations:** global = what matters overall; local = why *this* athlete got *this* score.
- **Feature importance vs SHAP:** tree "gain" importance says how often/strongly a feature is used; SHAP gives a consistent, signed contribution for each prediction.
- **LIME:** explains one prediction by fitting a simple model on randomly perturbed inputs. It's quick but unstable (different runs, different answers), which is why we prefer SHAP.
- **Why explainability matters:** coach trust, catching wrong reasoning, ethics, and medical caution.
- **Privacy and consent:** datasets have ethics approval and masked IDs; we never try to identify people; raw data never goes on GitHub.
- **Fairness:** few women in some datasets (section 16).
- **Synthetic data ethics:** always labelled; never presented as real people; checked for near-copies.
- **AI-use declaration:** we log AI assistance in `docs/AI_LOG.md` and declare it in the paper.

## 31. Software engineering and reproducibility
- **Reproducible =** someone else gets the same numbers. We use:
  - fixed seeds;
  - a lock file of package versions;
  - configs in YAML;
  - run folders with `run_info.json` (code commit, data fingerprint);
  - sha256 fingerprints of every data file.
- **Git:** `commit` = a saved snapshot with a message; `push` = upload to GitHub; history lets us see and undo changes.
- **Tests:** small programs that check the code still works (`pytest`); the guard hook stops the AI editing raw data, splits and results.
- **Virtual environment:** private packages per project, so versions don't clash.

## 32. Math refresher
- **Dot product:** [1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32. A neuron is a dot product + bias.
- **Matrix shapes:** (batch × features) × (features × neurons) = (batch × neurons). Example: a (128 × 70) input times a (70 × 32) weight matrix gives (128 × 32).
- **Sigmoid:** σ(0) = 0.5; σ(large) → 1; σ(very negative) → 0.
- **Softmax example:** scores [2, 1, 0] → e^x ≈ [7.39, 2.72, 1] → probabilities ≈ [0.67, 0.24, 0.09].
- **Chain rule:** if y = f(g(x)), then dy/dx = f′(g(x)) × g′(x). Backpropagation applies this layer by layer.
- **Logarithm in the loss:** log(1) = 0 (perfect confidence when right); log(0.01) ≈ −4.6 (a huge penalty when confidently wrong).

---

# PART 3: QUESTIONS AND NUMBERS

## 33. Big question bank

### A. Project and motivation
**Q: What problem are you solving?** Coaches can't see injury risk building. We predict it from daily training data, explain it, and show it on a dashboard inside an edge-fog-cloud design.

**Q: What's new compared to the 2026 AIoT framework?** It left the AI layer unimplemented. We implement it and test it on real injuries, with explanations checked, not assumed.

**Q: Why "AIoT" with no hardware?** The tiers run as separate programs passing messages, and we measure latency and bytes. The framework paper was also software-only.

**Q: What changed since Review 1?** We found and verified a real injury dataset (Lövdal). GANs became a training aid, not the only injury data. SIRP-600 moved to demo-only after the audit.

**Q: What type of ML problem is it?** Supervised binary classification on time-series data.

### B. Data
**Q: How do you know your data is real?** For Lövdal: a peer-reviewed paper, a DOI repository, and our audit counts match the paper exactly. For the others: published papers and body-logic checks with positive controls.

**Q: Why not use SIRP-600?** No athlete IDs, an undocumented origin, a 2-question rule predicts its label 89.7 % of the time, and its rows are different people from the runners, so fusion is impossible.

**Q: Is the Kaggle dataset fake?** No: it's byte-identical to the original Lövdal data. We cite the original.

**Q: Can your sensor data predict injuries?** No, and we don't claim it. The sensor people aren't the runners.

**Q: What does −0.01 mean?** A rest day in the perceived-effort columns.

**Q: Which column is the most recent day?** Day file: `.6` is the day before the event. Week file: no suffix is the week before. The loader orders both oldest to newest.

**Q: Why are SoccerMon and PMData only "exploratory"?** Self-reported pain, few injured people (15 and 9), very uneven reports (p12 alone has 48 of PMData's 77).

**Q: What is a data fingerprint?** A sha256 code computed from a file's bytes; one changed byte gives a different code. It proves exactly which data was used.

### C. Splits and honest testing
**Q: Why split by athlete?** A runner's days look alike; a random split lets the model recognise people.

**Q: Why lock the 10 newest runners?** An untouched final test, the paper's exact test runners, and a check on newer athletes.

**Q: How is the threshold chosen?** On validation runners only.

**Q: What is leakage? Examples?** Test information reaching training: the same person in train and test, scaling with test data, synthetic data in the test, a threshold chosen on test data.

**Q: Why did you build the Lövdal groups only once?** Grouping the two files separately gave slightly different groups, and fusion needs the same runners in both arms.

**Q: Why leave-one-person-out for PMData?** Only 16 people with very uneven injuries; fixed groups would be unbalanced. Training on everyone else and testing on each person in turn uses the data best.

### D. Metrics
**Q: Why not accuracy?** Always saying "no injury" gives 98.6 %.

**Q: What does AUC 0.72 mean?** A random injury window is ranked above a random healthy window 72 % of the time.

**Q: Why PR-AUC?** It focuses on the rare injuries; random scores ≈ 0.0136.

**Q: Precision vs recall?** Precision: how many alarms were real. Recall: how many injuries we caught.

**Q: What is calibration?** "70 %" should really mean about 70 %; checked with a reliability plot and the Brier score.

### E. Models
**Q: Why GRU and not a Transformer?** Small data and short sequences.

**Q: Why not BiLSTM?** It doesn't fit live streaming, doubles the parameters, and gains little on 7 steps.

**Q: LSTM vs GRU?** LSTM: 3 gates + a cell state. GRU: 2 gates, fewer parameters.

**Q: Bagging vs boosting?** Parallel and averaged (less variance) vs sequential, each fixing errors (less bias).

**Q: What is a hyperparameter?** A setting chosen before training, e.g. the learning rate.

**Q: What if the examiner wants a hyperparameter changed live?** The Training lab retrains a fast model on one development group in seconds to minutes. It's saved separately and never touches the real model or the locked test.

**Q: Why rebuild the paper's method?** To check that our data loading reproduces ~0.72, and to compare under identical rules.

### F. Neural-network basics (general)
**Q: What is an activation function, and why is it needed?** It adds non-linearity; without it, stacked layers act like one linear layer.

**Q: Sigmoid vs softmax?** Sigmoid: one probability (yes/no). Softmax: probabilities over many classes summing to 1.

**Q: What does backpropagation do?** It computes each weight's gradient using the chain rule, backwards from the loss.

**Q: What is the learning rate?** The step size of the weight updates.

**Q: Adam vs SGD?** Adam adapts each weight's step size automatically; SGD uses one fixed step (often with momentum).

**Q: How do you stop overfitting?** Dropout, L1/L2, early stopping, augmentation, smaller models, more data.

**Q: What is batch normalisation?** Rescaling each layer's outputs within a batch, for faster, stable training.

**Q: Vanishing gradient?** Gradients shrink through many layers, so early layers stop learning. Fixed by ReLU, residual connections, gates, batch norm.

**Q: What is an epoch? A batch?** One pass over all training data; a small group processed together.

**Q: Transfer learning?** Reusing a pretrained model's early layers. Not used here: no pretrained model exists for running logs.

### G. CNNs (teacher's topics)
**Q: What does a convolution layer do?** Slides learned filters over the input; each filter makes a feature map of one pattern.

**Q: What is pooling, and why?** Shrinks feature maps (max/average): less computation, some shift tolerance, no parameters.

**Q: 3 × 3 conv on 224 × 224, padding 1, stride 1: output size?** (224 − 3 + 2)/1 + 1 = 224.

**Q: 2 × 2 max-pool, stride 2, on 224?** 112.

**Q: Parameters in VGG16's first conv?** (3 × 3 × 3 + 1) × 64 = 1,792.

**Q: Why so many parameters in VGG16?** Mostly the first FC layer: 25,088 × 4,096 ≈ 103 million of ~138 million.

**Q: Why use 3 × 3 filters?** Stacked, they cover the same area as big filters with fewer parameters and more non-linearity.

**Q: What does a 1 × 1 convolution do?** Mixes and reduces channels cheaply; key in Inception and ResNet bottlenecks.

**Q: What are auxiliary classifiers?** Extra side outputs in GoogLeNet training to help gradients reach early layers; removed at test time.

**Q: What does global average pooling replace?** The big fully connected layers, which saves parameters.

**Q: What problem does ResNet solve?** Degradation and vanishing gradients in very deep nets; skip connections `F(x) + x` act as a gradient highway.

**Q: What if F(x) learns nothing useful?** It learns ≈ 0, so the block passes x through unchanged. That's why adding layers doesn't hurt.

**Q: Image size vs channels through VGG?** Size goes down (224 → 7) while channels go up (64 → 512).

**Q: Do you use CNNs?** Not image CNNs. Our TCN uses 1D convolutions over days in residual blocks (the ResNet idea), and a small 1D CNN is planned for PAMAP2 activities.

**Q: Why not ResNet-50 on your data?** 25.6 M parameters vs 74 runners means memorisation; ImageNet features don't transfer to training logs.

**Q: What is Grad-CAM?** A heat map from the gradients of the output w.r.t. the last conv layer's feature maps, showing which input parts (pixels, or days for us) drove the prediction.

**Q: What makes MobileNet suitable for edge devices?** Depthwise separable convolutions: far fewer computations.

### H. Generators
**Q: How does a GAN work?** Generator vs discriminator, improving together until fakes look real.

**Q: What is mode collapse?** The generator makes only a few kinds of examples.

**Q: Why WGAN-GP?** The Wasserstein distance plus a gradient penalty make training much more stable.

**Q: Can synthetic data be used for testing?** Never. The test must be real people.

**Q: How do you know synthetic data is good?** Fidelity (real-vs-fake classifier), usefulness (TSTR), privacy (no near-copies).

**Q: Why compare with TimeVAE and simple augmentation?** A GAN-only comparison is weak evidence; GANs must beat simple methods to be worth it.

### I. IoT and explanations
**Q: Why edge/fog/cloud?** Faster alerts and less data sent; we measure latency and bytes, tiered vs cloud-only.

**Q: What protocols would a real system use?** Bluetooth LE (watch-phone), Wi-Fi/4G (phone-cloud), MQTT or HTTP. We simulate them with processes.

**Q: How would you fit a model on a watch?** Quantisation, pruning, distillation, or small architectures (MobileNet-style).

**Q: Why isn't attention enough as an explanation?** It can disagree with what really drives the prediction, so we show it only next to SHAP or integrated gradients plus a deletion test.

**Q: SHAP vs LIME?** SHAP has fairness guarantees and is stable; LIME uses random perturbations and can change between runs.

### J. Honesty, ethics and limitations
**Q: What if your score is lower than the paper's?** Possible and acceptable: our testing is stricter (no label-based scaling, validation on separate people). We compare under the same rules.

**Q: What if your model scores 0.95?** We treat it as a bug (likely leakage) and investigate before believing it.

**Q: Did you use AI?** Yes, declared in the paper from `docs/AI_LOG.md`. The decisions, checks and understanding are ours, and every number comes from code we can run.

**Q: Privacy?** Masked IDs and ethics-approved datasets; raw data stays local; synthetic data is checked for near-copies.

**Q: Biggest limitation?** No "almost injured" days in the Lövdal data, a single team, and a small final test.

---

## 34. Numbers cheat sheet (all verified)

| Fact | Number | Source |
|---|---|---|
| Lövdal runners | 74 (27 women, 47 men); 77 before cleaning | paper |
| Day windows / injuries | 42,766 / 583 (1.36 %) | audit |
| Week windows / injuries | 42,798 / 575 (1.34 %) | audit |
| Injured runners (day file) | 63 of 74; 11 never injured | audit; data/processed/lovdal/facts.json |
| Shared day+week runner-days | 42,680 (575 injuries) | data/processed/lovdal/facts.json |
| Locked test (IDs 64-73) | day 50 injuries / 3,044 windows (= paper: 50 + 2,994); week 48 / 3,015 | data/splits/lovdal_folds.json, data/processed/lovdal_lockbox/*_meta.json; paper |
| Development test groups | 105-108 injuries each (day) | data/splits/lovdal_folds.json (balance) |
| Lövdal validation overlap | 46 places filled by 30 runners | data/splits/lovdal_folds.json (pinned by a test) |
| Development runners 0-63 | day 39,722 windows / 533 injuries; week 39,783 / 527 | data/processed/lovdal/*_meta.json |
| Validation people (D-020) | PAMAP2 4 different people in 4 folds; WESAD 10 in 5 | data/splits |
| Paper AUC day / week | 0.724 / 0.678 | paper Table 3 |
| Paper sensitivity / specificity (day) | 58.4 % / 74.1 % | paper Table 3 |
| Kaggle = Lövdal | identical sha256 (fingerprint bf5e4401…) | audit |
| PAMAP2 | 9 people, 10.7 h, 18 activities, HR median 9.13 values/s | audit |
| PAMAP2 check | HR running 156.6 vs lying 75.5 bpm | audit |
| WESAD | 15 people, 24.13 h; stress 166.1 min | audit |
| Sport DB 2.0 | 168 recordings, 55 continuous (~127.6 h); height-weight r = 0.686 | audit |
| SIRP-600 warning | 2-question tree 89.7 % vs 68.5 % majority | audit |
| ziya07 warning | HR resting 120.6 vs running 118.0 | audit |
| SoccerMon pain reports | 162 reports on 156 days from 15 of 50 players -> 27 episodes in 14 players | audit; data/processed/soccermon/daily_meta.json |
| PMData pain reports | 77 reports on 72 days from 9 of 15 labelled people (p12: 44 days) -> 8 episodes in 6 people | audit; data/processed/pmdata/daily_meta.json |
| Files fingerprinted | 729 in `data/MANIFEST.csv` | audit run 2026-10-15_0819 |
| Tests passing | see docs/STATE.md (grows with every piece) | `python tasks.py check` |
| VGG16 parameters | ~138 million (first FC layer: 102,764,544) | standard architecture |
| GoogLeNet parameters | ~5-7 million (slide: ~6.8 M) | standard architecture |
| ResNet-50 parameters | ~25.6 million | standard architecture |
| Our example GRU (10 in, 32 hidden) | 4,224 parameters | formula |
