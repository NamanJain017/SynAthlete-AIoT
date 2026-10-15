# SynAthlete-AIoT: the complete study guide (explained like you're in first year)

Written 2026-10-15. Read it top to bottom once, then use it as a reference before reviews and the viva.
Short Q&A per piece stays in `docs/VIVA_NOTES.md`; this file explains everything from zero.

**Status labels used everywhere below**
- ✅ **Done**: built, run on real data, numbers come from files written by code.
- 🔜 **Planned**: decided, not built yet. No results exist yet: never quote numbers for these.
- 📝 **Proposed**: suggested, waiting for Naman's approval.

**Golden rule for the viva:** every number you say must come from a file (an `audit.json`, a `metrics.json`, or the Lövdal paper). If you don't have a number yet, say "that part is planned; we have no result yet". That honesty is a strength, not a weakness.

---

## Contents
1. The project in plain words
2. Words you must know (mini-dictionary)
3. The datasets: what each one is, is it real, what it is used for
4. What has been done so far (with real numbers)
5. Honest testing: train / validation / test, splits, leakage
6. How we score models (metrics), with tiny worked examples
7. Machine learning basics: how a model learns
8. The models in our project, one by one
9. CNNs from your teacher's slides: convolution, pooling, VGG16, GoogLeNet, ResNet, and how they connect to our project
10. Fusion: combining two models into one risk score
11. Generators (GANs and friends): making realistic synthetic data
12. AIoT: edge, fog and cloud tiers
13. Explainability: showing why the model says "risky"
14. The coach dashboard and live retraining
15. The Lövdal paper in 2 minutes
16. Limitations we state openly
17. Tools, files and commands
18. What you can demonstrate today
19. How to show the project is yours (and honest about AI use)
20. Big question bank with answers
21. Numbers cheat sheet

---

## 1. The project in plain words

**Problem.** Athletes get injured when training load builds up over days and weeks. Coaches can't easily see this risk building. A 2026 AIoT framework paper (Applied Sciences 16(9):4542) designed a wearable + edge/fog/cloud system for athletes but did not implement the "AI part" (injury-risk prediction and explanations).

**Our idea.** Build that missing AI part as working software:
1. Wearable-style sensor data flows through three computer "tiers" (edge = watch, fog = phone, cloud = server).
2. A model trained on **real runners' daily training logs with real injuries** predicts injury risk (0 to 100).
3. Generators (GANs) create extra realistic data to help training where real data is scarce.
4. Every prediction comes with a reason a coach can read (explainability).
5. A coach dashboard shows athletes, risk scores, reasons, and the system.

**What is real and what is not (very important).**
- Injury results come **only** from real injury labels (the Lövdal runner logs). Rule 7 in `AGENTS.md`.
- Sensor datasets (PAMAP2, WESAD) are **different people** with no injury records. They show the IoT pipeline and train generators. They never prove anything about injuries.
- Synthetic (generated) data is only ever used for **training**, never for testing. Rule 3.

**Review 2 rubric (20 marks):** Methodology 5 (pipeline, dataset audit, data roles, evaluation rules), Implementation 10 (working code with real results), Paper draft 5 (IEEE-style conference paper).

---

## 2. Words you must know (mini-dictionary)

| Word | Simple meaning | Example in our project |
|---|---|---|
| Dataset | A collection of data in files | Lövdal runner logs |
| Row / sample / example | One item the model looks at | One runner's last 7 days before a day |
| Feature | One input number describing a sample | "total km on the day before" |
| Label / target | The answer the model must predict | injury = 1 (yes) or 0 (no) |
| Window | A fixed slice of a time series | 7 days of training (day file) |
| Model | A formula/program that maps features to a prediction | Random Forest, GRU |
| Training | Letting the model learn from examples with known answers | Fitting on training runners |
| Inference / prediction | Using a trained model on new data | Scoring a new runner's week |
| Parameter (weight) | A number *inside* the model, learned during training | GRU weights |
| Hyperparameter | A setting *you* choose before training | learning rate, number of trees |
| Overfitting | Memorising training data instead of learning general patterns | Great on training runners, bad on new ones |
| Underfitting | Model too simple to learn the pattern | Always predicts "no injury" |
| Validation set ("practice test") | Data used to choose settings, thresholds | Runners held out from training |
| Test set ("final exam") | Data used once for the honest final score | Runners the model never saw |
| Leakage | Information from the test sneaking into training | Same runner in train and test |
| Class imbalance | One label far rarer than the other | Only 1.36 % of windows are injuries |
| Synthetic data | Data made by a computer program | GAN-generated sequences |
| Seed | A number that fixes "randomness" so results repeat | seed 0 |
| Epoch | One full pass of training over all training data | GRU trained for 30 epochs |
| Batch | A small group of samples processed together | 128 windows at a time |
| Loss | A number measuring how wrong the model is | binary cross-entropy |
| Gradient | Which direction to change each weight to reduce the loss | from backpropagation |
| Probability | A number 0 to 1 meaning "how likely" | 0.70 = 70 % chance |
| Threshold | Cut-off that turns a probability into yes/no | alert if risk ≥ 0.45 |
| Calibration | Making "70 %" really mean about 70 % | Platt scaling |
| AUC | A score: how well the model ranks risky above safe | 0.5 random, 1.0 perfect |
| sha256 | A unique fingerprint of a file's bytes | in `data/MANIFEST.csv` |
| AIoT | AI + Internet of Things (devices that sense and send data) | watch → phone → server |
| Edge / fog / cloud | Device / nearby gateway / remote server | watch / phone / server |
| Latency | Delay between event and response | ms from sensor to alert |
| SHAP | Method that splits a prediction into each feature's contribution | "risk up because km jumped" |

---

## 3. The datasets: what each one is, is it real, what it is used for

All numbers below are from our audit (run `experiments/audit/2026-10-14_2346/`) unless marked "paper" or "page".

### #1 Lövdal runner logs ✅ (our main evidence)
- **What:** daily training logs of a Dutch high-level running team, 2012-2019, 800 m to marathon (paper). 74 runners (27 women, 47 men) after the authors' cleaning; 77 before (paper).
- **Two files:**
  - *Day file:* each row = one runner, one "event day", described by the 7 days before it × 10 features = 70 numbers. 42,766 rows, 583 injury rows (1.36 %).
  - *Week file:* each row = 3 weeks before the event day, each week summarised by 22 features, + 3 "mileage change" features = 69 numbers. 42,798 rows, 575 injury rows (1.34 %).
- **The 10 day features:** number of sessions, total km, km in zones 3-4, km in zone 5/track intervals (T1, T2), km sprinting, strength training, hours of cross-training, perceived exertion, perceived training success, perceived recovery. The three "perceived" ones are on a 0-1 scale; **-0.01 means rest day** (authors' README).
- **What counts as an injury (paper):** a "flag" in the runner's own training log: "unable to complete the scheduled session due to injury". Injuries less than 3 weeks apart count as one, so all 583 injuries are at least 22 days apart (we checked).
- **Real?** Yes: peer-reviewed paper (Int J Sports Physiol Perform 2021), university repository with DOI, ethics approval, CC0 licence. Our audit counts match the paper exactly (583 + 42,183 day; 575 + 42,223 week).
- **Trap we found:** the two files count days in **opposite directions**. Day file: suffix `.6` = the day before the event; no suffix = 7 days before. Week file: no suffix = the week before. Our loader must put both in oldest-to-newest order.
- **Role (D-012):** the only source of injury evidence: sequence model, snapshot model, explanations, dashboard replay.

### #2 Kaggle "Injury Prediction for Competitive Runners" ✅
- Byte-for-byte identical to Lövdal (same sha256 fingerprint for both files). It is the dataset shown in Review 1. **Role:** none separate; we cite the original.

### #3 PAMAP2 ✅
- 9 people (readme: 1 female, 8 male, age 27.22 ± 3.31) wearing 3 motion sensors (wrist, chest, ankle; 100 readings per second) + a heart-rate strap (about 9 per second: we measured a median of 9.13 values per second).
- 10.7 hours (Protocol 7.98 h + Optional 2.72 h), 7.57 h labelled with one of 18 activities (lying, walking, running, cycling…).
- **Real?** Yes: published benchmark; our check shows heart rate 156.6 bpm while running vs 75.5 lying, as real bodies behave.
- **Not athletes** (mostly employees/students). Say "wearable sensor data", not "athlete data".
- **Role:** live sensor stream for the edge/fog tiers; training data for generators.

### #4 WESAD ✅
- 15 people (S1 and S12 removed by the authors), 24.13 hours, chest device (700 readings/s, 8 channels) + wrist device (4-64 readings/s, 6 channels). Conditions: baseline 293.5 min, stress 166.1 min, amusement 92.9 min, meditation 196.8 min. 12 men, 3 women, ages 24-35.
- **Real?** Yes: published paper (ACM ICMI 2018). Licence: academic, non-commercial.
- **Role:** second sensor dataset: tests whether a generator also works on different data.

### #5 Sport DB 2.0 ✅
- 168 heart recordings from athletes in 11 sports; 129 subject folders + 4 skating recordings without a subject folder. Only 55 are continuous (ECG, heart rate, breathing; about 127.6 h of heart-rate data); 113 are short ECG clips at rest or after exercise.
- Sampling rates are not stored in the files: our code infers heart rate ≈ 1 sample/second and ECG = 250 samples per heart-rate sample. Marked [VERIFY].
- **Real?** Probably yes: university data deposit with an ethics document; height and weight correlate strongly (r = 0.686), as in real people.
- **Role:** realism check: does synthetic heart rate look like real athletes' heart rate?

### #6 SIRP-600 ✅ (demo only)
- 600 rows, 15 features, label "Injury_Risk" (411 zeros, 189 ones). No athlete ID, no time.
- **Warning signs:** a tiny decision tree with 2 questions predicts the label 89.7 % of the time (always guessing the common answer gives 68.5 %), using only Injury_History and Training_Intensity; the file was created and last saved 4 seconds apart; it is named "High_Accuracy". Real injuries are never that predictable.
- **Role (D-013):** demo/sanity check only. It was proposed for the "snapshot model" in Review 1; that role moved to Lövdal's week file.

### #7 ziya07 wearable ✅ (demo only)
- 500 rows, 10 athlete IDs, heart rate, temperature, blood pressure, activity…
- **Fails a real-body check:** heart rate "Resting" 120.6 vs "Running" 118.0 (no real difference, p = 0.569), while real PAMAP2 data shows +81 bpm.
- **Role:** dashboard demo only.

### #8 SoccerMon (subjective part downloaded) 🔜
- Two elite women's football teams in Norway, 2020-2021 (paper). We downloaded `subjective.zip` (705.8 kB; checksum matches Zenodo): daily training load (including ACWR), wellness (fatigue, mood, sleep, stress, soreness, readiness), injuries, illness. 731 days × 50 players.
- **Injuries:** 162 reports from 15 players (Team A 151, Team B 11). An "injury" is a tap on a body map, e.g. `{"right_foot":"minor"}`: self-reported pain, not a diagnosis. The same pain is often reported on consecutive days.
- GPS/heart-rate files (99.1 GB) not downloaded. Licence CC BY 4.0.
- **Role:** 📝 proposed (D-016): secondary, exploratory check.

### #9 PMData (downloaded) 🔜
- 16 participants, ages 23-60 (13 men, 3 women), 5 months, Fitbit Versa 2 (heart rate, steps, sleep…), daily self-reports (PMSys app), meals/weight (Google Forms). 1.42 GB zip, 912 files.
- **Injuries:** 225 reports, but only **77 actually name an injury**; the rest are empty "no injury" check-ins. 9 people reported at least one injury. Participant p08 has no injury file.
- **Why interesting:** the same people wear a sensor *and* report injuries.
- **Cautions:** self-reported pain, only 16 people, not athletes.
- **Role:** 📝 proposed (D-016): secondary, exploratory check.

---

## 4. What has been done so far (with real numbers)

### Piece 0: setup ✅
- Python 3.12.2 in a private environment (`.venv`), PyTorch 2.14.1 with CUDA (GPU: NVIDIA RTX 4050 Laptop, about 6 GB), exact package versions in `requirements.lock.txt` (70 packages at the time).
- `python tasks.py check` runs ruff (code checker) and pytest (automatic tests). Now: 23 tests pass.

### Piece 1: dataset audit ✅
- `python tasks.py audit` opens every dataset and writes facts: people, hours, signals, labels, injuries, missing values, duplicates, plausibility checks.
- Outputs: `experiments/audit/<run>/audit.json` (all facts), `audit_table.md`, `config.yaml`, `run_info.json` (code version, seed, data fingerprints), `data/MANIFEST.csv` (sha256 of 554 files, including the 2 Lövdal files), the table in `docs/DATASETS.md`.
- Speed: files processed in parallel on all CPU cores, limited by free memory; a full run takes about 1.5-2 minutes when enough RAM is free (it took about 18 minutes once when only 2.6 GB RAM was free).
- Safety: WESAD pickle files are opened with a "numpy-only" loader that refuses any hidden code (a test proves it refuses a pickle that tries to run a command).
- Findings: Kaggle = Lövdal (identical); SIRP-600 and ziya07 fail plausibility checks; Sport DB hours were first overcounted and fixed; the day/week order trap.
- Decisions D-012 to D-015 approved by Naman: dataset roles; snapshot model on Lövdal week features; fusion uses the 42,680 days present in both Lövdal files; the ACWR baseline is an adapted 3-week version.

### After piece 1 ✅
- Lövdal paper read in full (Section 15). SoccerMon and PMData found, downloaded and checked (Section 3).

### Piece 2: loaders + splits 🔜 (plan being written)
- Proposed split design in Section 5 (📝 awaiting approval).

---

## 5. Honest testing: train / validation / test, splits, leakage

### The exam analogy
- **Training runners = the textbook.** The model learns from them.
- **Validation runners = the mock test.** We use them to choose settings, the alert threshold, calibration, and when to stop training.
- **Test runners = the final exam.** Used to measure the honest score. Peeking at the exam while studying makes the marks meaningless.

### Why split by person, not by row
Days from the same runner look alike. If runner #12's Monday is in training and their Tuesday is in the test, the model "recognises" runner #12 and the score is inflated. So a runner is **entirely** in training, or validation, or test. Decision D-002, rule 2.

### 📝 Proposed split design for Lövdal (awaiting approval)
```
74 runners
├── 10 newest runners (IDs 64-73)  →  LOCKED FINAL TEST, used ONCE at the very end
└── 64 older runners (IDs 0-63)    →  DEVELOPMENT, 5 balanced groups
       each round: 1 group = test (~13 runners), ~10 runners = validation, rest = training (~41)
       → each of the 64 is tested exactly once
```
- **Why lock the 10 newest?** (1) Nobody can say we tuned on the test. (2) They are exactly the paper's test runners (we checked: IDs 64-73 contain 50 injuries and 2,994 healthy events, the same numbers the paper reports), so the final comparison is like-for-like. (3) They joined last, so it also tests "does it still work on newer athletes?".
- **Why 5 groups for the 64?** Balanced: our preview gives 105, 108, 108, 105, 107 injuries per group (1.32-1.36 %). Enough injuries per group for a stable score; not too many GAN trainings.
- **Safety rules:** seed 0 chosen in advance (no reshuffling until results look good); balance checked on label counts only; the split script refuses to overwrite saved splits; automatic tests (no runner in two roles, every development runner tested once, locked runners never in development); an independent AI reviewer (eval-auditor) checks the splits.
- **Other datasets (proposed):** PAMAP2 4 groups (subject 109 always in training: only ~85 seconds of protocol data); WESAD 5 groups of 3 people; SoccerMon 5 groups of 10 players with injured players spread evenly; PMData 4 groups of 4 people (p08 left out of injury experiments); Sport DB 5 groups by subject folder; SIRP-600 a row-level 80/20 "demo only" split.

### Other leakage rules
- Scaling (adjusting numbers to a common range) is learned from training runners only (rule 3). The paper scaled test runners using their own non-injured days, which uses test labels: we will not do that.
- The alert threshold is chosen on validation runners only (rule 5).
- Synthetic data never enters validation or test (rule 3).
- A result that looks too good is treated as a bug until proven otherwise.

---

## 6. How we score models (metrics), with tiny worked examples

### The confusion matrix
After choosing a threshold, every prediction is one of four kinds:

| | Really injured | Really healthy |
|---|---|---|
| **Model says "injury"** | True Positive (TP) | False Positive (FP): false alarm |
| **Model says "healthy"** | False Negative (FN): missed injury | True Negative (TN) |

### Why accuracy is useless here
Accuracy = (TP + TN) / all. Only 1.36 % of day windows are injuries, so a model that **always says "healthy"** gets 98.6 % accuracy and catches zero injuries.

### The scores that matter
- **Sensitivity (recall)** = TP / (TP + FN): of all real injuries, how many did we catch?
- **Specificity** = TN / (TN + FP): of all healthy windows, how many did we correctly leave alone?
- **Precision** = TP / (TP + FP): of all alarms, how many were real?
- **ROC curve:** sensitivity vs false-alarm rate as the threshold slides from 0 to 1.
- **ROC-AUC:** pick one injury window and one healthy window at random; AUC is the chance the model gives the injury window the higher risk. 0.5 = coin flip, 1.0 = perfect. The paper got 0.724 (day file).
- **PR-AUC:** area under the precision-recall curve; focuses on the rare injuries. A random model scores about the injury rate (0.0136 for the day file), so it shows clearly how far above chance we are.
- **Calibration:** if the model says 70 % for many windows, about 70 % of them should really be injuries. Checked with a reliability plot and the **Brier score** = average of (predicted probability − real label)². Lower is better.

**Tiny example (made-up numbers, just to practise):** 1,000 windows, 14 real injuries. A model raises 100 alarms; 7 of them are real injuries.
- Sensitivity = 7 / 14 = 50 %. Specificity = (986 − 93) / 986 = 90.6 %. Precision = 7 / 100 = 7 %.
- Precision looks low, but random alarms would be right only 1.4 % of the time, so 7 % is 5 times better than chance.

**Which metrics will we report?** 🔜 Decided in piece 4, **before** any generator runs (decision D-004). Proposal: ROC-AUC and PR-AUC as main scores, sensitivity/specificity at the chosen threshold, and calibration.

### The threshold
The model outputs a risk (e.g. 0.37). The dashboard needs "alert" or "no alert". The threshold is the cut-off. Lower threshold → more injuries caught but more false alarms. The paper picked the threshold where sensitivity ≈ specificity on validation data. We choose it on validation runners only.

### Reporting with honesty
- Report results as **mean ± spread** across groups and seeds (e.g. "0.70 ± 0.04": example only), not one lucky number.
- The locked final test has only 50 injuries, so its score gets an uncertainty range.

---

## 7. Machine learning basics: how a model learns

1. **Start:** the model's weights are random numbers.
2. **Forward pass:** feed a batch of training examples; the model outputs predictions.
3. **Loss:** compare predictions with true labels. For yes/no problems we use **binary cross-entropy**: it punishes confident wrong answers heavily.
4. **Backpropagation:** calculus (the chain rule) computes how much each weight contributed to the error: the **gradient**.
5. **Gradient descent:** change each weight a little in the direction that reduces the loss. The step size is the **learning rate**.
6. **Repeat** for many batches; one pass over all training data is one **epoch**.
7. **Early stopping:** watch the loss on validation runners; stop when it stops improving (that's when overfitting begins).

**Learning-rate intuition:** walking down a foggy hill. Too big steps → you jump over the valley and bounce around. Too small → it takes forever.

**Overfitting vs underfitting:** a student who memorises past papers (overfits) fails new questions; one who studied nothing (underfits) fails everything. We want the student who understood the pattern.

**Tools against overfitting:** small models, dropout (randomly switching off neurons during training so the network can't rely on any one), weight decay (penalising big weights), early stopping, more data (where GANs may help), and honest validation.

**Class imbalance tricks (training data only):** class weights (an injury mistake costs more), or balanced sampling (like the paper's 2,048 injury + 2,048 healthy bags). Never rebalance the test set.

---

## 8. The models in our project, one by one

### Baselines 🔜 (piece 3): the yardsticks
| Model | How it works | Why include it |
|---|---|---|
| Majority class | Always says "no injury" | Floor; proves accuracy is useless (98.6 %) |
| ACWR-style rule (3-week) | Acute:chronic workload ratio: this week's load ÷ usual load; alert if the ratio jumps. Adapted to 3 weeks because the files hold only 3 (D-015) | The rule coaches already use; ML must beat it to be worth it |
| Logistic regression | Weighted sum of features → squeezed into 0-1 by the sigmoid function | Simplest model; if fancy models barely beat it, that is a finding |
| Random Forest | Many decision trees, each trained on a random sample of rows and features; they vote | Robust on small tabular data; explainable with TreeSHAP |
| CatBoost | Gradient boosting: trees built one after another, each fixing the previous trees' mistakes; strong defaults | Usually the best on small tables |
| GRU | Small recurrent neural network reading the 7-day sequence | First test of "timeline beats snapshot?" |
| Paper's method rebuilt | 9 XGBoost models on balanced samples, averaged, Platt-calibrated | Fair comparison with the published result |

**Decision tree in one picture:** "Was total km last week > 80? → yes → Were there ≥ 2 rest days? → no → high risk." Each question splits the data so the groups become "purer" (mostly injury or mostly healthy).

**Bagging vs boosting:**
- *Bagging* (Random Forest, the paper's 9 models): many models trained **in parallel** on random samples; average their answers. Reduces wobbliness (variance).
- *Boosting* (CatBoost, XGBoost): models trained **one after another**; each new tree focuses on the examples the previous ones got wrong. Reduces systematic error (bias).

### Sequence arm 🔜 (R1): LSTM, GRU, TCN
**Recurrent neural network (RNN):** reads a sequence one step at a time (day 1, day 2, … day 7), keeping a "memory" (hidden state) that is updated each day. Problem: in plain RNNs, the signal from early days fades away during training (**vanishing gradient**).

**LSTM (Long Short-Term Memory):** an RNN with a separate long-term memory lane (the **cell state**) and three gates (small learned switches between 0 and 1):
- *forget gate:* what to erase from memory ("that easy week doesn't matter any more"),
- *input gate:* what new information to store ("today was a hard interval session"),
- *output gate:* what to reveal as the current output.

**GRU (Gated Recurrent Unit):** a lighter LSTM with two gates (*update* and *reset*) and no separate cell state. Fewer parameters, often equally good: good for small data. Example size: a PyTorch GRU with 10 input features and 32 hidden units has 3 × 32 × (10 + 32) + 6 × 32 = **4,224 parameters** (compare VGG16's ~138 million in Section 9).

**TCN (Temporal Convolutional Network):** convolutions over time instead of recurrence. Uses **causal** convolutions (only looks at the past) and **dilations** (skipping steps to see further back), stacked in **residual blocks** (the ResNet idea, Section 9). Stable to train, fast, good for short fixed windows. First to be cut if time is short (PLAN).

**Why not BiLSTM?** A BiLSTM reads the sequence forwards *and* backwards. It wouldn't be cheating here (the 7-day window is all past), but: it doesn't fit live streaming (must re-read the whole window every new day), doubles the parameters on very small data, gains little on only 7 steps, makes explanations muddier, and was not a Review 1 commitment. It can be added later as a one-row comparison if time allows.

**Why not Transformers?** They need lots of data; we have 74 runners and 7-day windows, so they would mostly memorise.

### Snapshot arm 🔜 (R1): Random Forest and CatBoost
Trained on Lövdal's **week file** (one row of 3-week summary numbers per runner per event day). Decision D-013: the same runners and days as the sequence arm, so the two can be fused.

---

## 9. CNNs from your teacher's slides, and how they connect to our project

Your teacher showed VGG16, GoogLeNet/Inception v1, and a ResNet video ("Residual Networks (ResNet) Explained Intuitively | Why Deep Networks Fail & How ResNet Fixes It", ByteQuest). These are **image** models. Our data are not images, but the same building blocks appear in our TCN and our sensor models. Know both: how they work, and how they relate to us.

### 9.1 The convolution layer
- An image is a grid of numbers: height × width × channels (3 channels = red, green, blue). 224 × 224 × 3 for VGG16.
- A **filter (kernel)** is a small grid of learned weights, e.g. 3 × 3 × 3. It **slides** over the image; at each position it multiplies overlapping numbers and adds them up → one output number. All positions together form a **feature map**.
- One filter detects one pattern (an edge, a corner). A layer with 64 filters produces 64 feature maps = 64 output channels.
- Early layers detect simple things (edges); deeper layers combine them into complex things (eyes, wheels, faces): exactly the "Eyes, Ears, Wheels, Faces, Object parts" list on your teacher's slide.
- **Stride (S):** how many pixels the filter jumps each step. **Padding (P):** zeros added around the border so the size doesn't shrink.
- **Output size formula:** `out = (W − K + 2P) / S + 1` (W = input size, K = kernel size).
  - VGG16 conv: (224 − 3 + 2×1)/1 + 1 = **224** (size kept).
- **Parameter formula:** `(K × K × C_in + 1) × C_out` (+1 is the bias).
  - VGG16's first conv: (3 × 3 × 3 + 1) × 64 = **1,792** parameters.
  - Its second conv: (3 × 3 × 64 + 1) × 64 = **36,928**.

### 9.2 ReLU (activation)
`ReLU(x) = max(0, x)`: negative numbers become 0, positive stay. It adds non-linearity (without it, many layers would collapse into one big linear formula) and is cheap.

### 9.3 Pooling
Shrinks feature maps while keeping the important information.
- **Max pooling:** take the largest number in each window. **Average pooling:** take the average.
- VGG16 uses 2 × 2 windows with stride 2 (as in the original paper; some diagrams draw it differently): (224 − 2)/2 + 1 = **112**, so each pool halves height and width.
- Why: fewer numbers → less computation, and a bit of tolerance to small shifts in position.

### 9.4 Fully connected (FC) layers and softmax
- **Flatten:** turn the last feature maps (7 × 7 × 512 in VGG16) into one long list (25,088 numbers).
- **FC layer:** every input connects to every output (like a classic neural network).
- **Softmax:** turns the final scores into probabilities that add up to 1 (1,000 classes for ImageNet).

### 9.5 VGG16 (Simonyan & Zisserman, 2014)
- **Idea:** only small 3 × 3 convolutions (stride 1, padding 1), stacked deep, with max-pooling between blocks.
- **16 learnable layers:** 13 convolution + 3 fully connected.
- **Blocks (your teacher's table):**

| Block | Conv layers | Filters (channels) | Size after the block's pool |
|---|---|---|---|
| 1 | 2 | 64 | 224 → 112 |
| 2 | 2 | 128 | 112 → 56 |
| 3 | 3 | 256 | 56 → 28 |
| 4 | 3 | 512 | 28 → 14 |
| 5 | 3 | 512 | 14 → 7 |

- **Image size decreases while channels increase:** the network trades "where" for "what".
- **End:** 7 × 7 × 512 → FC 4,096 → FC 4,096 → FC 1,000 → softmax. Trained on ImageNet (1,000 classes).
- **About 138 million parameters**, mostly in the first FC layer: 25,088 × 4,096 + 4,096 = 102,764,544 (about 74 %).
- **Why stack 3 × 3s?** Two 3 × 3 convs see a 5 × 5 area, three see 7 × 7, with fewer parameters and more ReLUs than one big filter.
- **Weakness:** huge and slow; expensive to run.

### 9.6 GoogLeNet / Inception v1 (Szegedy et al., 2014)
- **Idea:** instead of choosing one filter size, use **several in parallel** and let the network decide: the **Inception module** has branches with 1 × 1, 3 × 3 and 5 × 5 convolutions and a 3 × 3 max-pool, whose outputs are **concatenated** (stacked along channels).
- **1 × 1 convolutions** are cheap "channel reducers" placed before the expensive 3 × 3 and 5 × 5 filters (a bottleneck), which keeps it efficient.
- **Global average pooling (GAP)** at the end replaces most of the huge FC layers (average each feature map into one number).
- **Auxiliary classifiers:** two small extra output branches in the middle, used only during training, giving extra gradient signal to help with vanishing gradients.
- **About 22 learnable layers but only around 5-7 million parameters** (your teacher's slide says ~6.8 M), versus VGG16's ~138 M: much more parameter-efficient.

| Feature | VGG16 | GoogLeNet / Inception v1 |
|---|---|---|
| Basic idea | Sequential convolutions | Parallel multi-scale branches |
| Main convolution | 3 × 3 | 1 × 1, 3 × 3, 5 × 5 |
| 1 × 1 convolution | Not central | Very important |
| Pooling | Separate layer between blocks | Also a branch inside Inception |
| FC layers | 3 large FC layers | Mostly replaced by GAP |
| Parameters | ~138 million | ~5-7 million |
| Depth | 16 learnable layers | ~22 learnable layers |

### 9.7 ResNet (He et al., 2015): the video
- **The problem:** making plain networks deeper should help, but beyond a point **training error got worse**, not better (the "degradation" problem). Signals and gradients fade or get distorted passing through many layers (vanishing gradients).
- **The fix: skip (residual) connections.** A block computes `output = F(x) + x`: the input `x` is added back to the block's result. The block only has to learn the *change* (the residual) `F(x)`. If a layer isn't useful, it can learn F(x) ≈ 0 and simply pass `x` through.
- **Why it works:** the "+ x" path is a highway for the gradient during backpropagation, so even very deep networks (152 layers) train well. When shapes differ, the shortcut uses a 1 × 1 convolution to match them.
- **Variants:** ResNet-18/34 use "basic" blocks (two 3 × 3 convs); ResNet-50/101/152 use "bottleneck" blocks (1 × 1 → 3 × 3 → 1 × 1). Batch normalization after convolutions. ResNet-50 has about 25.6 million parameters. ResNet won the ImageNet 2015 challenge.
- **Batch normalization:** rescales each layer's outputs within a batch to a stable range: faster, more stable training.

### 9.8 How these ideas connect to OUR project
| Image idea | Where it appears in our project |
|---|---|
| Convolution | **1D convolution** over time: a kernel slides along days (Lövdal) or along sensor readings (PAMAP2/WESAD windows) instead of across pixels |
| Stride, padding, kernel size | Same meaning in 1D. Example: kernel 3 over 7 days without padding → (7 − 3)/1 + 1 = 5 outputs. Each filter with 10 features has 3 × 10 + 1 = 31 parameters |
| Dilation (not on the slides) | TCN skips steps to see further back. Kernel 3 with dilations 1, 2, 4 sees 1 + 2 × (1 + 2 + 4) = 15 steps, enough for a 7-day window |
| Residual connection (ResNet) | **TCN is built from residual blocks**: exactly `output = F(x) + x` |
| Pooling / GAP | Global average pooling over time to summarise a sequence before the output |
| Feature maps | Grad-CAM (Section 13) turns the last convolution's feature maps into "which days mattered" |
| Softmax | We predict yes/no, so we use a single **sigmoid** output (probability of injury) |
| Planned activity model | "Small CNN/GRU" on PAMAP2 sensor windows (piece 3): a 1D CNN classifying activities |

**Why we don't use VGG16 / ResNet-50 / Inception directly:** they are built for big images and millions of training examples. Our data is small sequences (7 days × 10 numbers; 74 runners). A 138-million-parameter network would memorise our runners instead of learning. ImageNet pretraining (cats, cars) does not transfer to training logs. So we use the *ideas* (convolution, pooling, residual connections) in small 1D models sized for our data. Some researchers turn time series into images to use 2D CNNs; we don't, because it adds complexity without a clear benefit on data this small.

---

## 10. Fusion: combining two models into one risk score 🔜 (piece 7)
- **Two arms:** the sequence arm reads the 7-day timeline (day file); the snapshot arm reads the 3-week summary (week file), for the **same runner on the same day**.
- **Stacking / blending:** a tiny combiner model (e.g. logistic regression) takes the two arm predictions and learns how much to trust each. It is trained on the **validation runners'** predictions: runners neither arm trained on, so it doesn't learn from over-confident answers.
- **Calibration:** Platt scaling (a sigmoid fit) or isotonic regression on validation runners, so "70" means about a 70 % chance.
- **Risk score 0-100:** calibrated probability × 100.
- **Why late fusion (not gluing all features into one model):** each arm stays separately testable and explainable; works with small data.
- **Only days in both files:** 42,680 shared runner-days (D-014).

---

## 11. Generators: making realistic synthetic data 🔜 (piece 5)

**Why:** injuries are rare (583 windows), and no public dataset links rich sensor streams to injuries. Synthetic data can give a model extra *training* examples.

**GAN (Generative Adversarial Network):** two networks play a game.
- **Generator:** starts from random noise, tries to create fake data that looks real.
- **Discriminator:** sees real and fake examples, tries to tell them apart.
- They improve together, like a forger and a detective. Training ends (ideally) when the detective can't tell (≈ 50 % right).
- Known problems: unstable training; **mode collapse** (the generator makes only a few kinds of examples).

| Generator | Key idea | Why it's in the project |
|---|---|---|
| TimeGAN | A GAN for time series: also learns an embedding and a supervised "next step" loss, so the *order over time* is realistic | Review 1 commitment |
| WGAN-GP | Uses the Wasserstein distance (how much "earth" must be moved to turn the fake distribution into the real one) plus a gradient penalty: much more stable training | Review 1 commitment |
| TimeVAE | Not a GAN: a variational autoencoder (encoder compresses to a small random code, decoder rebuilds) | A GAN-only comparison is weak evidence (D-003) |
| Classic augmentation | Jitter (small noise), scaling, time-warping | The cheap baseline: GANs must beat it |

**How generators are judged:**
- *Fidelity:* can a classifier tell real from synthetic? (near 50 % = realistic)
- *Usefulness:* **train on synthetic, test on real** (TSTR), and: does adding synthetic data to training improve the score on real test runners?
- *Privacy:* no synthetic example is a near-copy of a real person.

**Strict rules:** generators are trained only on training runners (one generator per training group); synthetic data is never used for validation or testing; synthetic data is labelled synthetic everywhere. Sensor data and injury data come from different people, so a synthetic "combined athlete" is a **demo**, never evidence (D-001). If GANs don't help, we report that honestly.

---

## 12. AIoT: edge, fog and cloud tiers 🔜 (piece 6)

| Tier | Real-world device | Our "reflex / supervisor / deep" layer | Example job |
|---|---|---|---|
| Edge | Watch / sensor | Reflex | Quick checks on raw readings (e.g. heart rate abnormally high), filter noise, compress |
| Fog | Phone / gateway | Supervisor | Session summaries (load of today's session), short-term trends |
| Cloud | Server | Deep | Daily injury-risk model, explanations, dashboard, (re)training |

- **Why tiers?** Sending every raw reading to the cloud costs bandwidth, battery and time. Doing simple work near the sensor gives faster alerts and sends far fewer bytes.
- **What we'll measure:** latency (milliseconds from reading to alert) and bytes sent, **tiered vs cloud-only**.
- **How:** three separate Python programs passing messages (D-008: not Docker/MQTT, same measurements, far less setup).
- **Data:** PAMAP2 (and Sport DB) replayed as if from a live device. The injury risk comes from Lövdal logs: different people, shown as **two separate tracks** in the demo.

---

## 13. Explainability: showing why the model says "risky" 🔜 (piece 8)

| Method | Used for | Simple idea |
|---|---|---|
| SHAP / TreeSHAP | Random Forest, CatBoost | From game theory: each feature gets its fair share of the prediction, like splitting a team prize by each player's contribution. TreeSHAP computes it exactly and fast for trees |
| Integrated gradients | GRU / LSTM / TCN | Slowly morph a "blank" input into the real input and add up how much the prediction changes along the way; the importances add up to the prediction |
| Attention weights | Sequence models | Which days the model "looked at" most. **Never shown alone**: research shows attention is not a reliable explanation (rule 12) |
| 1D Grad-CAM | TCN | Uses the gradient of the risk with respect to the last convolution's feature maps to highlight important days (from the image-CNN world) |
| Deletion test | All | Remove the features an explanation calls most important: the prediction should change more than when removing random features. Proves the explanation is faithful |
| Known risk factors | All | Does the model flag things sports science knows matter, e.g. sudden jumps in weekly mileage (the ACWR idea)? |

**Plain-language output for coaches:** "Risk 72/100, mainly because weekly distance jumped 40 % and recovery scores dropped": the numbers come from the explanation method.

**Note from the paper:** individual features correlate only weakly with injury (r between −0.04 and 0.07); injuries come from combinations of factors, which is why models + explanations are needed.

---

## 14. The coach dashboard and live retraining 🔜 (piece 9)
- **Planned pages (Streamlit):** athlete list with risk scores, athlete detail with plain-language reasons, system view (tiers, latency, bytes), alert log.
- **`python tasks.py demo`** will replay a held-out athlete through the tiers into the dashboard.

### 📝 Proposed: a "Training lab" page (for when the teacher says "change a hyperparameter and retrain")
- Pick a **fast** model (logistic regression, Random Forest, CatBoost, small GRU) and move sliders inside **safe ranges** (e.g. number of trees, tree depth, learning rate, GRU size, epochs).
- Press "Train": it trains on **one development group** and shows the validation score, the ROC curve and the time taken. It runs on the GPU if available, otherwise the CPU.
- **Rough time estimates (to be measured, not promises):** logistic regression and Random Forest: seconds; CatBoost: tens of seconds; small GRU: about 1-3 minutes.
- **Safe by design:**
  - inputs limited to sensible ranges, so no crashes from silly values;
  - errors are caught and shown as a message;
  - it never touches the locked final test;
  - demo runs are labelled "demo" and never overwrite official results.
- **GANs are too slow for live retraining** (minutes to hours): the dashboard shows their saved results instead.
- **If the teacher asks without this page:** you can edit a value in the model's `config.yaml` and run the training command, which works too, but the page is safer and faster to show.

---

## 15. The Lövdal paper in 2 minutes (verified against the full text)
- **Citation:** Lövdal, Den Hartigh, Azzopardi. "Injury Prediction in Competitive Runners With Machine Learning." *Int J Sports Physiol Perform* 2021;16(10):1522-1531. doi 10.1123/ijspp.2020-0518.
- **Method:** bagged XGBoost: 9 models, each trained on 2,048 injury + 2,048 healthy samples drawn with replacement, averaged, Platt-calibrated; learning rate 0.01, tree depth 2 or 3, 256 or 512 trees.
- **Test:** the 10 runners who joined most recently; trained on the first 64; 5 repeated experiments.
- **Results (Table 3):** day: AUC 0.724, sensitivity 58.4 %, specificity 74.1 % (threshold 0.448); week: AUC 0.678, sensitivity 50.4 %, specificity 74.6 % (threshold 0.476). Around 30 % of injuries can be caught at a 10 % false-alarm rate (day approach).
- **Most important features (their Figure 4):** day: high-intensity km and perceived training success / recovery / exertion; week: training success 2-3 weeks before, the longest single-day run, rest days, week-to-week mileage change.
- **Weak points we noticed (say them politely):**
  - test runners were scaled using their own non-injured days (uses test labels);
  - their validation data came from the training runners, not new people;
  - their Figure 4 reads the day order the way their later README says is reversed;
  - healthy days near injuries were removed from the data, which makes the task easier than real life.

---

## 16. Limitations we state openly
1. Healthy days within 3 weeks of an injury are not in the Lövdal files, so no model is tested on the tricky "almost injured" days. Real-life scores may be lower.
2. 74 runners from one team: results may not transfer to other teams or sports.
3. Injury = "couldn't complete a session" (self-logged), not a medical diagnosis.
4. The final locked test has 10 runners and 50 injuries: small, so we report an uncertainty range.
5. Sensor datasets are different people from the runners: no real sensor-to-injury link (SoccerMon/PMData give only small, self-reported, exploratory links).
6. Synthetic data cannot prove anything about real bodies or injuries.
7. The tiers are simulated as software processes, not real devices.

---

## 17. Tools, files and commands

| Tool | What it is | Why we use it |
|---|---|---|
| Python 3.12 + `.venv` | Programming language + private package folder | Same versions every time |
| PyTorch + CUDA | Neural-network library + GPU support | Train GRU/LSTM/TCN/GANs on the RTX 4050 |
| pandas, numpy, scipy | Tables, arrays, statistics | Data handling, audit checks |
| scikit-learn | Classic ML (LR, RF, splits, metrics) | Baselines, StratifiedGroupKFold splits |
| CatBoost, (XGBoost 🔜) | Gradient boosting libraries | Snapshot arm; rebuilding the paper's method |
| SHAP, Captum | Explanation libraries | TreeSHAP; integrated gradients |
| Streamlit | Python web-app library | Coach dashboard |
| pytest, ruff | Automatic tests; code checker | `python tasks.py check` |
| Git + GitHub | Version history + online backup | Every change saved with a message |

**Commands:**

| Command | What it does | Status |
|---|---|---|
| `python tasks.py check` | Run code checks + all tests | ✅ |
| `python tasks.py audit` | Audit all datasets (about 1.5-2 min) | ✅ |
| `python tasks.py audit --only kaggle_runners` | Audit one dataset (seconds) | ✅ |
| `python tasks.py snapshot` | Zip a backup of the project | ✅ |
| `python tasks.py data / baselines / gen / pipeline / train / explain / demo / paper-figures` | Later pieces | 🔜 (print "not implemented") |

**Key folders:** `docs/` (all explanations and decisions), `src/data/` (readers, audit, parallel), `configs/` (settings), `tests/`, `experiments/` (results of every run), `data/raw/` (datasets, laptop only, never on GitHub), `data/splits/` (the saved splits, 🔜).

---

## 18. What you can demonstrate today
1. **Run the tests:** `python tasks.py check` → "23 passed". Explain one test, e.g. the one that proves the WESAD loader refuses a pickle that tries to run a command.
2. **Run a fast audit:** `python tasks.py audit --only kaggle_runners` → open the new folder in `experiments/audit/` and `audit_table.md`. (It also adds a run folder and updates `data/MANIFEST.csv`; discard those changes in VS Code if you were only demonstrating.)
3. **Show the mirror check:** Kaggle = Lövdal, the same sha256 fingerprint in `run_info.json` (bf5e4401…).
4. **Show a "fake data" check:** in `audit.json`, ziya07's heart rate resting 120.6 vs running 118.0, next to PAMAP2's 156.6 vs 75.5.
5. **Show the decisions log:** `docs/DECISIONS.md` (D-001 to D-015, with who decided).
6. **Explain the split design** (Section 5) and why the 10 newest runners are locked.

---

## 19. How to show the project is yours (and honest about AI use)
- **You made the decisions:** every decision in `docs/DECISIONS.md` lists who decided (roles, splits, SIRP-600 change). Know *why* each one was made.
- **Run everything yourself** before the review: the commands in Section 18. Being able to run and explain beats memorising.
- **Know your numbers** (Section 21) and the limitations (Section 16).
- **Be honest about AI help:** the paper's AI-use statement is built from `docs/AI_LOG.md`. A good answer: *"Yes, I used an AI coding assistant, and it's declared in the paper. I made the decisions, checked every result against the data, and every number comes from code I can run right now."*
- **Practise explaining** each section to a friend in your own words. If you can teach it, you own it.
- **Small exercises:**
  - change `random_state` in a copy of the split preview and see how group balance changes;
  - compute VGG16's conv output sizes yourself;
  - explain the confusion-matrix example with your own numbers.

---

## 20. Big question bank with answers

### Project and motivation
**Q: What problem are you solving?** Coaches can't see injury risk building over days. We predict risk from daily training data, explain it, and show it on a dashboard, inside an edge-fog-cloud IoT design.

**Q: What is new compared to the 2026 AIoT framework paper?** That paper designed the system but left the AI layer (prediction + explanations) unimplemented. We implement it and test it on real injuries, with explanations checked rather than assumed.

**Q: Why is it "AIoT" without hardware?** The tiers run as separate programs passing messages, and we measure latency and bytes per tier. The framework paper we build on was also software-only.

**Q: What changed since Review 1?** We found and verified a real dataset with real injuries (Lövdal). Injury results now come from real injuries; GANs are a training aid, not the only source of injury data. SIRP-600 moved to demo-only after the audit.

### Data
**Q: How do you know your datasets are real?** For Lövdal: a peer-reviewed paper, a university repository with DOI, and our audit counts match the paper exactly. For the rest: published papers plus body-logic checks with positive controls (e.g. heart rate higher when running in PAMAP2).

**Q: Why not use SIRP-600?** No athlete IDs (can't test on unseen people), undocumented origin, a 2-question rule predicts its label 89.7 % of the time, and its rows are different people, so it can't be fused with the runner data.

**Q: Your Kaggle dataset: is it fake?** No: it is byte-for-byte identical to the original Lövdal data (same sha256). We cite the original.

**Q: Can your sensor data predict injuries?** No, and we don't claim it. Sensor people are not the runners and have no injury records.

**Q: What does -0.01 mean in the runner data?** A rest day in the perceived-effort columns (authors' README).

**Q: Which column is the most recent day?** Day file: suffix `.6` is the day before the event (README v2 correction). Week file: no suffix is the week before. Our loader orders both oldest to newest.

### Splits and honest testing
**Q: Why split by athlete?** A runner's days look alike; a random split lets the model recognise people and inflates the score.

**Q: Why lock the 10 newest runners?** A truly untouched final test, the same runners as the paper (fair comparison), and a check on newer athletes.

**Q: How do you avoid tuning on the test?** Settings, threshold, calibration and early stopping use validation runners only; the locked test is used once.

**Q: What is leakage?** Information from the test reaching training. Examples we prevent: the same person in train and test, scaling using test data, synthetic data in the test, choosing the threshold on test data.

### Metrics
**Q: Why not accuracy?** Always saying "no injury" gives 98.6 %. Useless.

**Q: What does AUC 0.72 mean?** Picking one injury window and one healthy window at random, the model ranks the injury window higher 72 % of the time.

**Q: Why PR-AUC?** It focuses on the rare injuries; random guessing scores only about 0.0136.

**Q: How do you choose the alert threshold?** On validation runners, e.g. where sensitivity ≈ specificity, or to keep false alarms below a coach-chosen budget. Fixed in piece 4.

### Models
**Q: Why GRU and not a Transformer?** Small data and short sequences: Transformers need far more data.

**Q: Why not BiLSTM?** It doesn't fit live streaming, it doubles the parameters on small data, and it gains little on 7 steps.

**Q: Difference between LSTM and GRU?** LSTM has a separate cell state and 3 gates; GRU has 2 gates, no cell state, fewer parameters.

**Q: Bagging vs boosting?** Bagging trains models in parallel on random samples and averages (less variance); boosting trains them one after another, each fixing the last one's errors (less bias).

**Q: What is a hyperparameter?** A setting chosen before training (learning rate, tree depth), unlike weights, which are learned.

**Q: What if the teacher asks to change hyperparameters and retrain?** With the proposed Training lab page: move a slider, retrain a fast model on one development group, see the new validation score in seconds to minutes. GANs show saved results because they take too long.

### CNNs (teacher's topics)
**Q: What does a convolution layer do?** Slides small learned filters over the input; each filter detects a pattern and produces a feature map.

**Q: What is pooling?** Shrinking feature maps (max or average) to reduce computation and add tolerance to small shifts. VGG16 uses 2 × 2 max-pooling with stride 2, halving the size.

**Q: Output size of a 3 × 3 conv on 224 × 224 with padding 1, stride 1?** (224 − 3 + 2)/1 + 1 = 224.

**Q: Why does VGG16 have so many parameters?** Mostly the first FC layer: 25,088 × 4,096 ≈ 103 million of the ~138 million.

**Q: What does a 1 × 1 convolution do?** Mixes and reduces channels cheaply, without looking at neighbours. A key part of Inception.

**Q: What are auxiliary classifiers?** Extra side outputs during GoogLeNet training that give extra gradient signal against vanishing gradients; removed at test time.

**Q: What problem does ResNet solve?** Very deep plain networks train *worse* (degradation, vanishing gradients). Skip connections (`F(x) + x`) give gradients a highway and let layers learn small corrections.

**Q: Do you use CNNs or ResNet?** Not image CNNs. Our TCN uses 1D convolutions over time in **residual blocks** (the ResNet idea), and the planned activity model on PAMAP2 windows is a small 1D CNN. Big image models would memorise our small data.

**Q: What is Grad-CAM?** A heat map of which input regions drove a CNN's prediction, from the gradients of the output with respect to the last convolution's feature maps. In 1D it shows which days or time steps mattered.

### Generators
**Q: How does a GAN work?** A generator makes fakes, a discriminator spots fakes, and both improve until the fakes look real.

**Q: Can synthetic data be used for testing?** Never. The test must be real people; otherwise we would be grading the model on the generator's habits.

**Q: How do you know synthetic data is good?** Fidelity (a real-vs-fake classifier), usefulness (train on synthetic, test on real), and privacy (no near-copies).

### AIoT and explanations
**Q: Why edge/fog/cloud?** Faster alerts and less data sent; we measure latency and bytes, tiered vs cloud-only.

**Q: Why is attention not enough as an explanation?** Research shows attention weights can disagree with what actually drives predictions; we show them only next to tested methods (SHAP, integrated gradients) and a deletion test.

### Honesty and limitations
**Q: What if your model scores lower than the paper?** Possible and acceptable: we test more strictly (no label-based scaling of test runners, validation on separate people). We compare under the same rules and say which is which.

**Q: What if your model scores 0.95?** We would treat it as a bug (likely leakage) and investigate before believing it.

**Q: Did you use AI?** Yes, declared in the paper from `docs/AI_LOG.md`; decisions, checks and understanding are ours, and every number comes from code we can run.

---

## 21. Numbers cheat sheet (all verified)

| Fact | Number | Source |
|---|---|---|
| Lövdal runners | 74 (27 women, 47 men); 77 before cleaning | paper |
| Day windows / injuries | 42,766 / 583 (1.36 %) | audit |
| Week windows / injuries | 42,798 / 575 (1.34 %) | audit |
| Injured runners (day file) | 63 of 74; 11 never injured | audit |
| Shared day+week runner-days | 42,680 | data check |
| Paper test runners | IDs 64-73: 50 injuries, 2,994 healthy (day) | paper + our check |
| Paper AUC day / week | 0.724 / 0.678 | paper Table 3 |
| Paper sensitivity / specificity (day) | 58.4 % / 74.1 % | paper Table 3 |
| Kaggle = Lövdal | identical sha256 (fingerprint bf5e4401…) | audit |
| PAMAP2 | 9 people, 10.7 h, 18 activities, HR median 9.13 values/s | audit |
| PAMAP2 check | HR running 156.6 vs lying 75.5 bpm | audit |
| WESAD | 15 people, 24.13 h; stress 166.1 min | audit |
| Sport DB 2.0 | 168 recordings, 55 continuous (~127.6 h) | audit |
| SIRP-600 warning | 2-question tree 89.7 % vs 68.5 % majority | audit |
| ziya07 warning | HR resting 120.6 vs running 118.0 | audit |
| SoccerMon injuries | 162 reports from 15 of 50 players | file check |
| PMData injuries | 77 real injury reports of 225, 9 of 16 people | file check |
| Tests passing | 23 | `python tasks.py check` |
| VGG16 parameters | ~138 million (FC6 alone 102,764,544) | standard architecture |
| GoogLeNet parameters | ~5-7 million (slide: ~6.8 M) | standard architecture |
| ResNet-50 parameters | ~25.6 million | standard architecture |
| Our example GRU (10 in, 32 hidden) | 4,224 parameters | formula |
