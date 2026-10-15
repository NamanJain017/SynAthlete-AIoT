# STATE (living status; update at the end of every session)
Last updated: 2026-10-15 by Claude Code (Naman's laptop)

## Current piece
1 Download + audit all datasets: DONE (roles approved by Naman 2026-10-14). Next: 2 Loaders + person-level splits.

## Done (with evidence)
- Lean starter repo pushed to GitHub (rules, brief, dataset list, plan, task runner, guard hook).
- Piece 0 Setup (2026-10-13, commit 2f44327), output copied from the laptop:
  - `.venv\Scripts\python.exe --version` -> `Python 3.12.2`
  - `python tasks.py check` -> ruff `All checks passed!`, pytest `3 passed in 0.01s`
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` -> `2.14.1+cu132 True NVIDIA GeForce RTX 4050 Laptop GPU`
  - `requirements.lock.txt` created with `pip freeze` (70 pinned packages).
- Piece 1 Download + audit (2026-10-13 to 2026-10-14):
  - `python tasks.py check` -> ruff `All checks passed!`, pytest `23 passed`.
  - Final audit run `experiments/audit/2026-10-14_2346/` (code 26d0105, git_dirty false); its facts are identical to run `2026-10-14_0502` (only origin/role cells changed). `data/MANIFEST.csv`: sha256 of every raw file read. Table in docs/DATASETS.md.
  - Headline facts (from audit.json): Lövdal 74 athletes, day 583/42,766 injury windows (1.36 %), week 575/42,798; Kaggle runners byte-identical to Lövdal (mirror check, shared fingerprint bf5e4401...); PAMAP2 9 subjects, 10.7 h; WESAD 15 subjects, 24.13 h; Sport DB 2.0 168 recordings, 55 continuous (~127.6 h); SIRP-600 600 rows, no IDs; ziya07 500 rows, 10 IDs, HR resting 120.6 vs running 118.0.
  - Roles and protocol decisions: D-012 (role per dataset), D-013 (snapshot arm on Lövdal week features, SIRP-600 demo only), D-014 (fusion on the 42,680 shared athlete-days), D-015 (ACWR-style 3-week baseline).
  - Extra data facts checked 2026-10-14 (read-only, not in audit.json): all 583 injury windows are separate injuries (no two within 7 days; minimum gap 22 days); 11 of 74 athletes never injured; top-5 athletes hold 18.4 % of windows and 14.6 % of injury windows.

## In progress
- Nothing running. SoccerMon (#8): download page unavailable.

## Blocked / needs human decision
- Review 2 date and target conference (ask the guide). Scope risk: pieces 2-10 remain; follow the PLAN cut order if time is short.
- Licences marked [VERIFY] in configs/datasets.yaml: Kaggle runners, SIRP-600, ziya07, Sport DB 2.0, SoccerMon (check each download page).
- Lövdal paper read 2026-10-15 (details in docs/DATASETS.md Rules): AUC 0.724/0.678 verified; test = the 10 most recently joined athletes; healthy events exclude the 3 weeks around injuries (limitation for any score on this data). Plan: add their 10-athlete split in piece 2 and rebuild their method (with and without their test-athlete scaling) in piece 3.
- Optional second real dataset: SoccerMon on Zenodo (doi 10.5281/zenodo.10033832, CC BY 4.0). subjective.zip is 705.8 kB (training load, wellness, self-reported injuries: 162 reports / 306 injuries, Team A 291 vs Team B 15); GPS zips total about 99 GB. Not downloaded; decide after piece 3.

## Next three actions
1. Piece 2: loaders for every dataset (reuse src/data/readers.py) + person-level splits saved once in data/splits/ + leakage tests + data cards.
2. Piece 3: baselines on Lövdal (majority, ACWR-style 3-week, LR, RF, CatBoost, GRU).
3. Run the eval-auditor subagent before any audit or result goes into the Review 2 deck.

## Ideas for later
- Optional BiLSTM comparison (one extra row) if time allows; not an R1 commitment.
- Approved 2026-10-15 (D-016, piece 9): dashboard "Training lab" page; runs saved separately in experiments/training_lab/, never overwriting official models/results, never using the locked test.
- Beginner study guide: docs/STUDY_GUIDE.md (covers everything so far + plans; update it after each piece).

## Known issues and surprises
- requirements.lock.txt pins the CUDA build (`torch==2.14.1+cu132`). Teammates without a GPU install CPU torch first, then `pip install -r requirements.txt` (not the lock file). See docs/SETUP_WINDOWS.md.
- Runner logs (authors' README.txt): `-0.01` in the three perceived-effort columns means "rest day"; dates are a day index from the first record. GOTCHA: day file suffix `.6` = day before the event and no suffix = seven days before, while the week file's no-suffix = the week before; the loader must put both into oldest-to-newest order.
- Sensor datasets and runner logs are different people: two separate tracks (DATA_CONTRACT A/B); the demo must never present them as one athlete. PAMAP2 and WESAD subjects are not described as athletes: call them "wearable sensor data" in slides.
- Scores will be noisy with 74 athletes (~15 per test fold): report a spread across folds, not one number.
- WESAD licence is academic/non-commercial: do not publish WESAD-derived synthetic data on GitHub without checking it.
- Sport DB 2.0: sampling rates are not in the files (implied HR ~1.005 Hz, ECG 250 samples per HR sample) [VERIFY against the dataset description before the realism check]; subject folders restart per sport, so one person in two sports cannot be detected; 4 skating recordings have no subject folder.
- The audit picks its worker count from free RAM (measured peak about 1.23 GB per WESAD worker). With 2.6 GB free, WESAD ran on 1 worker and took 1,008 s instead of ~69 s: close browser tabs before long runs.
