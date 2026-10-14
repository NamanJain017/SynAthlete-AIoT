# VIVA NOTES

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
