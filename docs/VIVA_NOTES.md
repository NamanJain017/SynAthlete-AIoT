# VIVA NOTES

For piece 2 in the simplest words (9 topics, everyday examples, project progress), read `docs/PIECE2_EXPLAINED.md`.

For the full beginner-level explanation of everything (datasets, splits, metrics, every model, CNN/VGG16/GoogLeNet/ResNet, GANs, tiers, explanations, a big question bank and a numbers cheat sheet), read `docs/STUDY_GUIDE.md`.

Plain-language explanation of every piece, added as each piece is built. Each section: what it does, why it is needed, what the numbers mean, likely questions with answers.

## The project in 30 seconds
Coaches can't see injury risk building up over days. We built a software-only AIoT system: wearable-style sensor data flows through three simulated tiers (watch, phone, server); a model trained on real runners' daily training logs with real injuries predicts risk; generative models (GANs) create extra realistic data for the rare injury cases; every prediction comes with a reason a coach can read.

## Likely questions (general)
**Q: Why is it "AIoT" if there is no hardware?**
A: The tiers (edge, fog, cloud) run as separate programs that pass messages, and we measure latency and data sent per tier. The May 2026 Applied Sciences AIoT framework we build on was also software-only.

**Q: Why not just split the data randomly?**
A: Windows from the same person look alike. A random split puts the same person in training and test, so the model "recognises" them and the score is inflated. We split by person, so the test is always on unseen athletes.

**Q: What changed since Review 1?**
A: We found a real dataset (Lövdal et al., 74 runners, 7 years, real injuries). So injury results are now measured on real injuries, and the GAN creates extra rare-injury examples instead of being the only source of injury data.

<!-- Add one section per piece below. -->

## Piece 0: setup
**What it does.** Prepares the laptop so every later step runs the same way every time.
- A **virtual environment** (`.venv`) is a private folder of Python packages just for this project, so other projects on the laptop cannot change our versions.
- **PyTorch with CUDA** lets the RTX 4050 GPU train the neural networks (generators, sequence models) much faster than the CPU.
- **`requirements.lock.txt`** lists the exact version of every installed package (70 of them), written by `pip freeze`. Anyone can rebuild the same environment later, so results can be reproduced.
- **`python tasks.py check`** runs ruff (finds code mistakes and style problems) and pytest (automatic tests). Green means the code base is in a known-good state.

**What the numbers mean.** `Python 3.12.2`; `2.14.1+cu132 True NVIDIA GeForce RTX 4050 Laptop GPU` means PyTorch version 2.14.1 built for CUDA 13.2, and `True` means it can see the GPU; `3 passed` means all three tooling tests pass (path guard and snapshot filter).

**Q: Why a lock file if you already have requirements.txt?**
A: requirements.txt says *which* packages we need; the lock file says *which exact versions* we had when results were produced. Libraries change behaviour between versions, so the lock file is what makes a result reproducible months later.

**Q: Your teammates have no GPU. Can they still run the project?**
A: Yes. They install the CPU build of PyTorch and requirements.txt. Only generator and model training need the GPU; data audit, splits, baselines, explanations and the dashboard run on CPU.

**Q: Why check `torch.cuda.is_available()`?**
A: On Windows a plain `pip install torch` silently installs the CPU-only build. Training would still run, just many times slower, so we check explicitly that PyTorch sees the GPU.

## Piece 1: dataset audit (done 2026-10-14)
**What it does.** `python tasks.py audit` opens every downloaded dataset and writes down facts about it: how many people, how much data, which signals, which labels, whether there are injury labels, missing values, duplicates. It also runs "is this real?" checks, records a fingerprint (sha256) of every file in `data/MANIFEST.csv`, and fills the table in docs/DATASETS.md. Run used here: `experiments/audit/2026-10-14_0433/` (code commit da3e0de, 121 s).

**Why it is needed.** Every number in the paper must come from code, not from a web page or memory. The audit also tells us which dataset can play which role, and which ones must never be used as evidence.

**What the numbers mean (from audit.json).**
- Kaggle runners (#2): 74 athletes. Day file: 42,766 rows, each = one athlete's last 7 days; only 583 rows (1.36 %) are followed by an injury, so the data is very imbalanced (a model that always says "no injury" is already 98.6 % "accurate", which is why accuracy is a bad metric here). `-0.01` appears in the three "perceived" columns on days without a value: a code for "no answer", not a real number.
- PAMAP2 (#3): 9 people, 10.7 h, 18 activities. The file has 100 rows per second, but heart rate has a value in only about 9 of them (median 9.13 values per second), which matches the ~9 Hz monitor in the readme. Check that works: heart rate while running is 156.6 bpm vs 75.5 lying.
- WESAD (#4): 15 people (S1 and S12 were removed by the authors), 24.13 h; stress 166 min vs baseline 294 min.
- Sport DB 2.0 (#5): 168 recordings, but only 55 are continuous (~127.6 h of heart-rate data); 113 are short ECG clips taken at rest or after exercise. Sampling rates are not stored in the files; the code works them out (heart rate ~1 sample per second) and they are marked [VERIFY].
- SIRP-600 (#6) and ziya07 (#7): no proof they are real. In ziya07, heart rate is the same when "Resting" (120.6) and "Running" (118.0); in real PAMAP2 data the same check gives +81 bpm. In SIRP-600 a tiny decision tree with 2 questions predicts the "injury risk" label 89.7 % of the time, and the file was created and last saved 4 seconds apart.

**Q: How can you say SIRP-600 or ziya07 might not be real?**
A: We do not claim proof. We run checks that real body data passes: heart rate rises with running, taller people weigh more, values are bell-shaped rather than evenly spread. We run the same checks on data we know is real (a "positive control") to show the checks work. ziya07 fails the heart-rate check that real PAMAP2 passes, so it is labelled "unknown" and used only for demos, never as evidence.

**Q: What is the sha256 fingerprint for?**
A: It is a unique code computed from a file's bytes. If even one byte changes, the code changes. Recording it in MANIFEST.csv proves exactly which version of each dataset our results came from, so anyone can check they have the same files.

**Q: Why did the audit not use the GPU?**
A: The audit reads files, turns text into numbers and computes fingerprints. That work is limited by the disk and CPU, and a GPU does not speed it up. Instead the audit runs files in parallel on all CPU cores, as many at once as free memory allows. The GPU is used later for training the generators and models.

**Q: Is the Kaggle runner file the same data as Lövdal et al.?**
A: Yes, proven by code. We downloaded the original from DataverseNL and the audit's mirror check found both CSV files byte-for-byte identical (same sha256 fingerprint; run `experiments/audit/2026-10-14_0502/`). So the Kaggle file shown in Review 1 is real data, but we cite and use the original source.

**Q: In the runner data, which column is the most recent day?**
A: It depends on the file, which is a classic trap. The authors' README says: in the day file, suffix `.6` is the day before the event and the unsuffixed column is seven days before (their paper first stated this the wrong way round); in the week file, the unsuffixed columns are the week before the event. Our loader puts both into oldest-to-newest order, otherwise a sequence model would read time backwards.

**Dataset roles (decided 2026-10-14, DECISIONS D-012 to D-015).** Lövdal is the only source of injury evidence. PAMAP2 and WESAD feed the IoT tiers and the generators. Sport DB 2.0 checks whether synthetic heart rate looks real. SIRP-600 and ziya07 are demos only. Kaggle runners is the same data as Lövdal. SoccerMon is cited in related work.

**Q: Your Review 1 slide used SIRP-600 for injury risk. Why did you change that?**
A: Three reasons found by the audit. Its rows are anonymous people with no athlete ID, so we cannot test on unseen people. Its origin is undocumented and a 2-question rule predicts its label 89.7 % of the time, which real injuries never allow. And fusion needs both model arms to judge the same athlete on the same day. So the snapshot arm now uses Lövdal's weekly summaries: real injuries, the same runners as the sequence arm.

**Q: Can your sensor data predict injuries?**
A: No, and we do not claim it. PAMAP2/WESAD people are not the Lövdal runners and have no injury records. The sensor data shows the IoT pipeline (edge, fog, cloud) and trains the generators; injury results come only from the Lövdal training logs. That separation is exactly the data gap our project statement describes.

**Q: Why an "ACWR-style 3-week" baseline instead of the normal ACWR?**
A: The classic acute:chronic workload ratio compares this week's load with the last 4 weeks, but the runner files only hold 3 weeks. We use an adapted 3-week version and label it as such, rather than claim the classic rule.

## Piece 2: loaders and person-level splits (built 2026-10-15)
**What it does.** Two jobs.
1. **Loaders** (`python tasks.py data`) turn each raw dataset into tidy arrays in `data/processed/` (not in git). They keep raw values: no scaling and no filling-in of gaps, because those must be learned from training people only (rule 3). Every loader first checks that the raw files it reads still match the fingerprints the audit saved in `data/MANIFEST.csv`.
2. **Splits** (`python tasks.py splits`) decide, once and for all, which people are used for training, validation and testing. They are saved in `data/splits/`, committed to git, and sealed with a fingerprint file (`SHA256SUMS`).

**Why it is needed.** Every later score depends on these two steps. If a person appeared in both training and test, or the test changed between runs, no result could be trusted.

**What the numbers mean.**
- **Lövdal runners** (`data/processed/lovdal/*_meta.json`): each row is one runner's last 7 days (day file) or last 3 weeks (week file), ordered oldest to newest.
  - Development runners 0-63: 39,722 day windows (533 injuries) and 39,783 week windows (527 injuries).
  - Locked runners 64-73: 3,044 day windows with 50 injuries (the paper's exact test set). They are kept in a separate folder.
  - Proof of the time order, run on the real file: for 40,000+ pairs of consecutive days, day d+1's window is exactly day d's window moved one day.
- **Sensor windows** (`data/processed/{pamap2,wesad}/windows_meta.json`): 5.1-second slices at 25 samples per second.
  - PAMAP2: 10,361 windows × 128 steps × 10 channels, 18 activities.
  - WESAD: 17,401 windows × 128 × 6 (baseline 6,846, stress 3,861, amusement 2,144, meditation 4,550).
  - 0 missing values in either.
- **Daily tables** (`data/processed/{soccermon,pmdata}/daily_meta.json`): one row per person per day. Repeated pain reports are grouped into episodes (21-day rule, D-018).
  - SoccerMon: 156 pain-report days become 27 episodes in 14 of 50 players.
  - PMData: 72 report days become 8 episodes in 6 of 15 people.
- **Sport DB 2.0** (`data/processed/sportdb2/hr_meta.json`): 55 heart-rate recordings, 461,744 samples. 93.91 % are plausible (30-220 bpm). Most of the rest are 0, the device's "no signal" value.
- **Splits:**
  - Lövdal: 5 development groups with 105-108 injuries each; the 10 locked runners are opened once at the end.
  - PAMAP2: 4 groups. WESAD: 5 groups. Validation rotates, so nobody validates twice (D-020).
  - SoccerMon and PMData: leave-one-person-out.

**The main result and the locked test (D-017).** The headline score will be the average ± spread over the 5 development groups: 533 injuries, so more precise. The 10 locked runners are the final check: 50 injuries, opened once after the evaluation freeze, even if Review 2 comes first.

**Q: Why not just report the score on the paper's 10 test runners? That's the fair comparison.**
A: We report both. The 10 runners have only 50 injuries, so one score on them moves a lot with luck. The 5 development groups test 533 injuries from 64 runners, so their average is much more stable. We will measure both uncertainties with a bootstrap that resamples runners (2,000 resamples, fixed in D-023) instead of quoting a guess. The locked runners stay untouched until the end, so that final number is one nobody could have tuned on. And the paper comparison happens on the same 10 runners, at the end.

**Q: Your rules say splits are created once. Why were two files re-made?**
A: An independent review found that the first PAMAP2 and WESAD files reused the same people for validation (in WESAD, 10 validation slots held only 5 people). The rule exists to stop reshuffling until results look good. No model had been trained yet, the test groups stayed identical (the code checks this), the old files are kept in `data/splits/superseded/`, and the decision is logged (D-020). Then the folder was sealed, so it can't happen again.

**Q: How do you know nobody peeked at the locked test runners?**
A: Four safeguards.
1. Their arrays are saved in a separate folder.
2. Only one function can open them (`src/data/lockbox.py`). It refuses unless:
   - the code is at the commit marked by an annotated `eval-freeze` tag that is also on GitHub;
   - no unfrozen code is lying around;
   - the lockbox was never opened at a different commit.

   Every opening is written to a log, which is committed afterwards.
3. Tests fail if any other module reads the raw or locked runner data, or if a development file ever contains a locked runner.
4. An annotated tag records who made it and when, and once it is pushed anyone can check that on GitHub.

In honesty, the raw file still contains them, so this is a guard rail plus an audit trail, not encryption. Their rows were also part of the dataset-level audit counts in piece 1, but no model has been trained or scored on them.

**Q: In the Lövdal folds, do validation runners repeat?**
A: Yes, and we say so. Each fold picks its validation group from its own non-test runners, so across 5 folds the 46 validation places are filled by 30 runners (a test pins these numbers). Inside one fold, validation and test never overlap, which is what matters for that fold's score. Because a runner can validate in one fold and be tested in another, settings must be chosen inside each fold. We never pick one setting by averaging across folds (D-021).

**Q: How do you choose the alert level, and why not use accuracy?**
A: Accuracy is useless here: only about 1.3 % of the scored events are injuries, so saying "healthy" every time scores about 98.6 %. Our main score is PR-AUC (average precision), which needs no alert level and is built for rare events. For yes/no alerts we use the paper's rule: on each round's own practice runners, pick the level where "injuries caught" is about equal to "healthy days left alone". Then we apply that level to the round's exam runners. We also report false alarms per caught injury, so nobody over-reads the result (D-022).

**Q: Why don't you scale each runner against their own history, like the paper?**
A: Because in this dataset a runner's history gives the answer away. The authors kept injury events at least 22 days away from other events, while healthy events come day after day. Just counting a runner's events in the previous 3 weeks finds injuries with ROC-AUC 0.9946 (`data/processed/lovdal/event_design_check.json`). A model using that would look brilliant but would have learned how the data was built, not anything about injuries. So we use one scaler learned from training runners only, and a test checks that no history signal is used (D-023).

**Q: Why don't you send your best-scoring model to the final locked test?**
A: Choosing the winner by its test score and then reporting that same score overstates it, because some of the win is luck. So we named the final model in advance (the fused model, D-021). On the locked runners, its chosen arms and the label-free copy of the paper's method are scored once next to it, never used to choose anything. Every model's development-round results are reported.

**Q: How do you know your results can be reproduced?**
A: We tested it. A fresh copy of the code from git, plus copies of exactly the 729 raw files listed in `data/MANIFEST.csv`, ran the whole audit, every data builder and every test. Everything came out identical to our working copy: the file fingerprints, the audit facts, every processed array and table, the data cards, and all 7 split files rebuilt from seed 0 (`experiments/repro/2026-10-22_0731/`). So anyone with the same raw files gets the same data.

**Q: Why 21 days for SoccerMon/PMData episodes?**
A: People report the same sore knee on many days in a row. Counting each report as a new injury would multiply one problem into ten. We copied the Lövdal authors' rule: reports within 3 weeks are the same problem. The result is 162 SoccerMon reports, falling on 156 report days, which become 27 episodes.

## How the code is built: AI tools and checks (from 2026-10-25)
**What it is.** Some project files are written by AI coding tools from **task cards** (`docs/handoff/`). Claude Code writes the card and the **tests** first; a builder tool (Antigravity or ChatGPT Go's Codex) writes the code until the tests pass; Claude Code reviews it and runs it on the real data before it joins the project. Leak-sensitive parts are not handed out.

**Why.** It lets the project move faster with limited tool budgets, without lowering the quality bar. The tests and guards decide what is correct, not the tool.

**The checks, in order.**
1. The task card says exactly what to build and which files may change.
2. The tests check the answers on small worked examples. For WP-01 they were themselves tested: all 22 pass on a reference version, and three deliberately broken versions each fail.
3. A git hook blocks commits that touch tests, task cards, configs, data, results or the rulebook.
4. GitHub re-runs all tests on every push.
5. Claude Code reviews every change and runs it on the real data.

**Q: Did you use AI tools to write the code?**
A: Yes, and we say so openly. Claude Code helped design, write and review the project, and some files were written by other AI tools from task cards. Every use is logged in `docs/AI_LOG.md` with the tool and model, and that log becomes the paper's AI-use statement. Every number in the project still comes from code we can run, and we can explain every part.

**Q: How do you know code written by an AI tool is correct?**
A: We don't trust it, we test it. The tests were written first, with answers worked out by hand, and we checked that the tests catch broken code. The tool cannot change the tests (a git hook blocks it), GitHub re-runs everything on each push, and a separate review checks for leaks and rule breaks before anything is merged.

**Q: Why not let one AI write the whole project?**
A: Because the risky parts are about honest evaluation, not just code. One leak (for example, scaling with a runner's own history) can make a useless model look excellent. So the parts where a leak could hide stay with careful review, and only well-specified, testable files are handed out.

**Q: Did the review of AI-written code ever find a real problem?**
A: Yes, in the very first task card (WP-01, the score calculator). The builder's code passed all of its tests, but our checks found three issues. A fit stopped slightly too early, so a slope was a little off. The calibration error changed with row order when many events had the same risk. Calibration functions accepted numbers that are not probabilities. The independent reviewer also showed that 10 plausible wrong versions passed the original 22 tests. We fixed the code, recorded the rule changes (D-025), and grew the tests to 51; they now catch all 10 wrong versions. That is why every builder change is reviewed before it joins the project.

**Q: What is the runner bootstrap, and why resample runners instead of days?**
A: To show how sure we are about a score, we re-draw the test runners 2,000 times with replacement (some runners appear twice, some not at all), recompute the score each time, and report the middle 95% as the range. We draw whole runners because one runner's days are related: the same body, the same habits. Drawing single days would treat them as independent and make the range look narrower than the evidence allows. This is the cluster bootstrap (Field & Welsh 2007), fixed in D-024.

**Q: When you compare two models, why "paired"?**
A: In every re-draw both models are scored on exactly the same runners. Then luck in which runners were drawn affects both models equally and cancels out in the difference, so the comparison is fairer and tighter than scoring them on different re-draws.

**Q: What did the review of the second task card (WP-02, the bootstrap) find?**
A: The code was correct. Compared with a private reference version on random cases, the only differences were in the last decimal digit, because the builder typed 2.5 where the card's formula gives 2.500000000000002 in computer arithmetic. The review also removed an unfinished leftover function and merged a loop that had been copied three times into one. After that, the results matched the reference exactly.

## Piece 3: first real models (all runs done 2026-10-28)
**What it does.** Each of the 5 development rounds trains a model on that round's training runners, tunes it on that round's validation runners only, and scores it once on that round's test runners. The round builder (`src/models/data.py`) puts every runner in exactly one role, fits the z-score scaling on training rows only, and refuses inputs that are IDs, dates or a runner's own history. Results are written by `python tasks.py baselines` into `experiments/baselines/2026-10-26_020012/` (metrics.json, summary.md, run_info.json with the commit and data fingerprint).

**First numbers** (mean over the 5 rounds; `experiments/baselines/2026-10-26_020012/summary.md`):
- Leak canary passed: logistic regression on shuffled labels scored ROC-AUC 0.498 (day) and 0.513 (week), i.e. chance.
- Constant risk: PR-AUC 0.013, which equals the injury rate (the chance level). ROC-AUC 0.500.
- ACWR-style rule (week km jump): ROC-AUC 0.540, only a little above chance.
- Logistic regression: day file PR-AUC 0.022, ROC-AUC 0.626; week file 0.020 / 0.621.

**Q: Your ROC-AUC is about 0.62, the paper reports 0.724. Did you do something wrong?**
A: We expected lower numbers. The paper scaled each runner's data using that runner's healthy days, which needs the labels, including the test runners' labels, and it chose its alert level on training runners. We do neither (D-021, D-023). Also, these are only the simplest models; the stronger ones come next. A correct lower number is worth more than a leaky higher one.

**Q: Why is PR-AUC so small (0.022)?**
A: Only about 1.3 % of events are injuries, so a random guess gets PR-AUC 0.013. 0.022 is about 1.6 times chance. PR-AUC is always read next to the injury rate.

**Q: What is the leak canary, and what can it NOT catch?**
A: We train logistic regression on training labels that were shuffled, so there is nothing real to learn; its test score must be at chance (within 0.05 of 0.5). It passed. It catches pipeline bugs where the test labels reach the predictions. It cannot catch inputs that were themselves built using the labels (like the paper's label-reading scaling), because a model trained on shuffled labels cannot learn from them. Those are blocked by other guards: the declared-input check, the tests that the scaler sees training rows only, and the rule to stop if a score looks too good. A stronger version (shuffle all labels before the whole pipeline) has been proposed to Naman.

**Q: Does a passed leak alarm prove there is no leak?**
A: No, and we never claim that. Each check covers certain kinds of leak, so we use layers: people never cross between train, validation and test (tests on the split files); scaling is fitted on training runners only (tested); inputs are only the declared 70/69 features, never an ID, date or the runner's own history (input guard; that history predicts injury almost perfectly in this dataset because of how the authors removed days near injuries, so it is banned); a result that looks too good stops the work; an independent reviewer checks the code; and the leak alarm checks that a model trained on shuffled labels scores at chance. The alarm catches bugs where test answers reach the predictions, not inputs built from labels and not dataset-design leaks.

**Q: The networks were written by an AI tool without a private answer copy. How do you know they are right?**
A: Three checks. (1) The tests check network sizes against formulas, causality of the TCN, exact predictions, and that early stopping keeps the best version. (2) A surprise check planted 21 deliberate mistakes in the tool's code; the first tests caught 14, the other 6 showed gaps in our tests (not bugs in the code), so we added tests until all 21 were caught. (3) A sanity run on the real PAMAP2 data trained in seconds and scored far above chance on people it never saw.

**All piece-3 results** (mean ± SD over the 5 development rounds; summary.md in `experiments/baselines/2026-10-26_020012`, `experiments/baselines/2026-10-26_022902`, `experiments/baselines/2026-10-28_024255`; chance PR-AUC = injury rate ≈ 0.013):

| Model | Day PR-AUC | Day ROC-AUC | Week PR-AUC | Week ROC-AUC |
|---|---|---|---|---|
| Constant risk | 0.013 | 0.500 | 0.013 | 0.500 |
| ACWR-style rule | – | – | 0.015 | 0.540 |
| Logistic regression | 0.022 | 0.626 | 0.020 | 0.621 |
| Random forest | 0.019 | 0.603 | 0.022 | 0.614 |
| CatBoost | 0.021 | 0.598 | 0.019 | 0.606 |
| GRU (day only) | 0.023 | 0.631 | – | – |
| Lövdal method, label-free | 0.017 | 0.584 | 0.021 | 0.589 |

Paper-copy check (exact copy incl. its label-reading scaling, trained on runners 0-53, tested on stand-in runners 54-63): ROC-AUC day 0.617 (95 % runner bootstrap 0.573 to 0.664), week 0.613 (0.546 to 0.675): both PASSED (lower bound above 0.5). PAMAP2 activity recognition (4 rounds, test macro-F1): 1D-CNN 0.632 ± 0.097, GRU 0.566 ± 0.054 (chance about 0.06).

**Q: Which model is best, and is it much better than the others?**
A: All learned models sit close together: ROC-AUC about 0.58 to 0.63 and PR-AUC about 1.3 to 1.8 times chance. The GRU and logistic regression are slightly ahead, but the differences are small compared with the round-to-round spread (SD about 0.01 to 0.07), so we do not claim a winner from piece 3. The model sent to the locked test was declared in advance (the fused model, piece 7), not picked from this table.

**Q: Did you rebuild the paper's method correctly? Why is your copy 0.62, not their 0.724?**
A: Our exact copy (same settings, same label-reading scaling) passed the pre-registered check: its ROC-AUC interval lies above 0.5 on both files. 0.724 was measured on the other 10 runners (64-73), which stay locked until the end, so the two numbers are not on the same people and 0.724 was never a target. Without its label-reading step (the label-free variant) the same model scores 0.584 / 0.589 on the 5 rounds.

**Q: Why is PAMAP2 macro-F1 only about 0.6?**
A: Each test person was never seen in training, and people move differently; 18 activities include similar ones (e.g. walking vs Nordic walking, ironing vs standing). Scoring on new people is the honest setting; mixing one person's clips into training and test would give much higher, misleading numbers.

## Pieces 4-10: rules, generators, tiers, fused model, explanations, dashboard, paper (2026-10-28)
**What is built (code tested; results arriving from runs):**
- **Piece 4, rules first (D-027):** pass marks for generators (real-vs-fake accuracy within 0.15 of 50 %, train-on-synthetic within 0.15 of real, shape errors below 0.10, at most 5 % too-close copies) and for explanations (deletion 1.5x random, randomisation correlation below 0.5, stability above 0.8), written before any run.
- **Piece 5, synthetic data (`src/gen/`):** TimeGAN, WGAN-GP and TimeVAE learn from each round's training people only and make clips for a chosen activity; classic augmentation as the simple comparison. Judged on real people they never saw.
- **Piece 6, tiers (`src/pipeline/`):** edge, fog and cloud as three separate programs passing messages; we count bytes sent and time to a decision, tiered vs cloud-only.
- **Piece 7, fused model (`src/models/fused.py`):** the best day-by-day model (GRU/LSTM/TCN) and the best weekly-totals model (RF/CatBoost) per round, joined by a small logistic regression that only ever sees predictions made without that runner (out-of-fold).
- **Piece 8, explanations (`src/xai/`):** SHAP, integrated gradients, an attention view and Grad-CAM, each checked by deletion, randomisation and stability tests.
- **Piece 9, dashboard (`src/dashboard/`):** 8 pages in pipeline order; data prepared in advance so the demo is instant.
- **Piece 10:** IEEE paper draft and project report (numbers inserted by code), Review 2 deck made by code (`slides/review2.pptx`).
- **Track C (exploratory):** SoccerMon LR ROC-AUC 0.764 on 27 episodes; PMData 8 episodes, about chance (`experiments/track_c/2026-10-28_055034`).

**Q: Why do you train generators only on training people?**
A: If a generator saw a test person, its fake clips could copy that person, and a model trained on them would look good on that person for the wrong reason. So generators learn from each round's training people only, and synthetic data is used only for training.

**Q: What does "tested on an idle machine" mean for the tiers?**
A: The speed test measures milliseconds per message. If heavy training runs at the same time, every message waits longer and the system looks slower than it is, so the speed test runs when nothing else runs. The byte counts do not depend on load.

**Q: How do you stop the combined model from cheating?**
A: Its combiner is trained on arm predictions made by models that never trained on that runner (5-fold out-of-fold, grouped by runner), and a test proves no row is scored by a model that saw its runner.

**Q: What problems did you catch while building?**
A: Two: a real-vs-fake checker that could be under-trained (it would make every generator look perfect; now it always trains enough, proven by a test), and TimeVAE requesting 9 GB on a 6 GB GPU (fixed by letting cuDNN choose a fitting algorithm; 2x faster).

**Combined (fused) model result** (`experiments/fused/2026-10-28_060435/summary.md`): ROC-AUC 0.609 ± 0.024, PR-AUC 0.022 over the 5 development rounds; its chosen day-by-day arm alone 0.604, its weekly-totals arm alone 0.612.

**Q: Did combining the two models help?**
A: Not clearly. The fused model scores about the same as its two parts, and the differences are smaller than the round-to-round spread. We report that honestly: on this data, the day-by-day view and the weekly-totals view carry largely the same information. The fused model was still declared in advance as the one sent to the final locked test, so we did not switch to whichever model looked best.

**Q: Did anything go wrong while running the experiments?**
A: Yes, and we fixed it openly (D-029). The synthetic-data training first ran far too slowly (a strict reproducibility mode and too many jobs on one laptop GPU), then ran out of memory (old models were not freed), and once crashed because results were saved only at the end. Each fault was measured, fixed, tested and logged; the runs now save after every step and resume after a crash. None of this changes any result, it only cost time.


## Results of the night of 28-29 Oct: generators, explanations, freeze and the locked test

**Synthetic data, activity clips (PAMAP2, 4 rounds; `experiments/generators/2026-10-28_172535/metrics_pamap2.json`).** A real-vs-fake checker tells synthetic clips from real ones far too easily for every generator (accuracy TimeGAN 0.956, WGAN-GP 0.898, TimeVAE 0.936; the pass mark is within 0.15 of 0.5). Training only on synthetic clips (TSTR) comes close enough to real training for WGAN-GP, TimeVAE and classic augmentation, not for TimeGAN (macro-F1 0.270). Adding synthetic clips to real ones changes little (about 0.64 vs 0.632 real-only).

**Same generators on WESAD (5 folds, settings copied from PAMAP2, no tuning on WESAD; `metrics_followup.json`).** No generator meets the realism criteria; only TimeGAN meets the privacy criterion.

**Synthetic injury examples (Lövdal, 5 rounds, added to training rows only).** GRU ROC-AUC with TimeGAN examples 0.633 ± 0.006, with classic augmentation 0.617 ± 0.015, real-only 0.631 ± 0.009: no real gain.

**Explanations (`experiments/explain/2026-10-28_205117/metrics.json`).** SHAP (tabular arm) and integrated gradients (sequence arm) pass all three checks in 5 of 5 rounds: deletion (removing the top inputs lowers the risk more than removing random ones), model randomisation (a random-weight model gives different explanations), stability (near-identical inputs give near-identical explanations). Grad-CAM on PAMAP2 also passes randomisation and deletion.

**Q: Your generators fail most realism tests. Is that a failure of the project?**
A: No. The pass marks were written before any run (D-027), and we report the results as they came out. A real-vs-fake checker beats every generator, so the synthetic clips are not realistic enough to stand in for real data. That is a useful, honest finding, and it is why synthetic data is never used as evidence about real athletes.

**Q: Did synthetic injury examples improve injury prediction?**
A: No. With TimeGAN examples the GRU scores 0.633 against 0.631 real-only, well inside the round-to-round spread; classic augmentation scores lower (0.617). We say "no measurable gain".

**Q: What do the explanation checks prove?**
A: That the explanations follow the model rather than noise: hiding the inputs an explanation calls important changes the prediction more than hiding random inputs; a model with random weights gets different explanations; tiny input changes do not reshuffle them. Attention weights are never shown alone (invariant 12).

**The freeze and the locked test.** Before the 10 locked runners (64-73) were opened, an independent reviewer (eval-auditor) checked the final-test program. It found no leakage but two must-fix problems in how results were saved: raw scores were computed into numbers before being saved, and the run record was incomplete. Both were fixed (D-031), plus overnight safeguards (D-032): every lockbox condition is checked before training, the locked rows are loaded only after all training, raw scores are saved before any number is computed, a short network drop is retried, and the laptop is kept awake. A dry run on development runners gave exactly the same numbers as before the fixes, so the fixes changed no result. Then the code was frozen with an annotated `eval-freeze` tag (commit 7a710c0), pushed to GitHub, and the final model was trained and scored once.

**Q: How do you know the locked-test result was not tuned on the test runners?**
A: The model, its settings search and every threshold were fixed and written down (D-021, D-023, D-030) before the tag; the program can open the locked runners only at the tagged commit with no changes, and every opening is logged in `experiments/lockbox_access.log` with the commit. The training never sees the locked rows: they are loaded only after training ends.

**Q: Why is the locked test reported next to the 5-round average and not instead of it?**
A: Ten runners is a small group, so one number on them is noisy (its bootstrap interval even understates the uncertainty). The 5-round development average uses 64 runners and is the main result; the locked test is a final check on runners nobody looked at.

**Locked final test result** (`experiments/lockbox/2026-10-29_010127/metrics.json`): on the 10 locked runners (2,994 shared days, 48 injuries, 7 injured runners) the declared fused model reached ROC-AUC 0.605 (95 % runner bootstrap 0.535-0.699) and PR-AUC 0.030 (0.016-0.056). The 5-round development average was 0.609 ± 0.024. Secondary results (scored once, never used to choose anything): the GRU arm on the day file 0.609, the CatBoost arm on the week file 0.567, and the Lövdal-style model without its label peek, trained on runners 0-63, 0.649 (day) and 0.602 (week).

**Q: What does the locked test tell you?**
A: That our development results were honest. A model that had been over-tuned would drop on runners nobody looked at; ours scored 0.605 against 0.609 in development. It also confirms the main finding: training logs give only a weak warning before an injury.

**Q: The Lövdal-style model scored 0.649 on the final test, higher than your fused model's 0.605. Why not switch to it?**
A: Because choosing the model after seeing test results is exactly the cheating the lockbox exists to stop: the fused model was declared before the freeze (D-021, D-023, D-030). The two numbers are also not a fair comparison: they were scored on different rows (day file vs shared days), trained on different runners (64 vs 51), and their intervals overlap heavily (0.56-0.75 vs 0.535-0.699). On the larger, fair test (5 development rounds, 64 runners) the fused model was higher (0.609 vs 0.584). So we say: no model is clearly best; 10 runners are too few to rank models.

**Paired comparisons** (`experiments/comparisons/2026-10-29_020232`; runner bootstrap within the 5 rounds, 2,000 resamples). Against the Lövdal label-free variant on the same file, only logistic regression (day, ROC-AUC +0.042 [+0.008, +0.065]) and the GRU (day, +0.047 [+0.013, +0.071]) are clearly better; the others' intervals include zero. The fused model vs its day-sequence arm: ROC-AUC +0.005 [+0.00003, +0.009] (just above zero; PR-AUC no difference); vs its weekly arm: -0.003 [-0.022, +0.012].

**Q: What does "clearly better" mean here?**
A: We resample runners 2,000 times and recompute the difference; if the middle 95 % of those differences stays above zero, the gain is unlikely to be luck of which runners were tested. If the interval includes zero, we do not claim a difference.

**Tiers benchmark** (`experiments/tiers/2026-10-29_025032`, idle laptop): tiered sends 2,932 bytes into the cloud vs 1,739,186 cloud-only for PAMAP2 (593x fewer) and 1,600 vs 135,619 for Sport DB 2.0 (85x fewer). Median decision time 4.25 ms vs 2.77 ms (PAMAP2) and 0.78 vs 1.00 ms (Sport DB).

**Q: Why is the tiered decision slower than cloud-only for PAMAP2?**
A: Everything runs on one laptop, so there is no network delay: cloud-only gets the raw window almost instantly. In a real deployment the raw stream would cross a network, which the edge tier avoids; the honest claim is the 593x data saving, not speed.

**Q: You changed the latency measurement. Is that allowed?**
A: Yes, it was fixed before the first official run and logged (D-034). The first trial showed about 3 seconds per window, which was not decision time but windows waiting while the other processes were still starting. Now every tier starts first, then windows arrive at a steady pace; a test proves the old method would report 1,181 ms of start-up wait where the real decision takes under 1 ms.

**Q: One GitHub check is red. Why?**
A: When the freeze tag was pushed, GitHub's checkout tool turned the annotated tag into a plain one on its test machine (a known bug, actions/checkout #290), so our freeze test correctly complained. We reproduced it, fixed the workflow (D-033), and did not move the tag; the real tag on GitHub is annotated.
