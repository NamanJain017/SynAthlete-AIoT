# VIVA NOTES

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
A: Choosing the winner by its test score and then reporting that same score overstates it, because some of the win is luck. So we named the final model in advance (the fused model, D-021) and report every other model honestly next to it.

**Q: How do you know your results can be reproduced?**
A: We tested it. A fresh copy of the code from git, plus copies of exactly the 729 raw files listed in `data/MANIFEST.csv`, ran the whole audit, every data builder and every test. Everything came out identical to our working copy: the file fingerprints, the audit facts, every processed array and table, the data cards, and all 7 split files rebuilt from seed 0 (`experiments/repro/2026-10-22_0731/`). So anyone with the same raw files gets the same data.

**Q: Why 21 days for SoccerMon/PMData episodes?**
A: People report the same sore knee on many days in a row. Counting each report as a new injury would multiply one problem into ten. We copied the Lövdal authors' rule: reports within 3 weeks are the same problem. The result is 162 SoccerMon reports, falling on 156 report days, which become 27 episodes.
