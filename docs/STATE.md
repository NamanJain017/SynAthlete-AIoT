# STATE (living status; update at the end of every session)
Last updated: 2026-10-14 by Claude Code (Naman's laptop)

## Current piece
1 Download + audit all datasets: audit done for 7 of 8 (SoccerMon unavailable); waiting on role decisions.

## Done (with evidence)
- Lean starter repo pushed to GitHub (rules, brief, dataset list, plan, task runner, guard hook).
- Piece 0 Setup (2026-10-13, commit 2f44327), output copied from the laptop:
  - `.venv\Scripts\python.exe --version` -> `Python 3.12.2`
  - `python tasks.py check` -> ruff `All checks passed!`, pytest `3 passed in 0.01s`
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` -> `2.14.1+cu132 True NVIDIA GeForce RTX 4050 Laptop GPU`
  - `requirements.lock.txt` created with `pip freeze` (70 pinned packages).
- Piece 1 audit code + first evidence run (2026-10-14):
  - `python tasks.py check` -> ruff `All checks passed!`, pytest `22 passed`.
  - `python tasks.py audit` -> run folder `experiments/audit/2026-10-14_0433/` (code commit da3e0de, git_dirty false, 121 s), `data/MANIFEST.csv` (552 files with sha256), table in docs/DATASETS.md.
  - Headline facts (all from audit.json): Kaggle runners 74 athletes, day 583/42,766 injury windows (1.36 %); PAMAP2 9 subjects, 10.7 h; WESAD 15 subjects, 24.13 h; Sport DB 2.0 168 recordings, 55 continuous (~127.6 h); SIRP-600 600 rows, no IDs; ziya07 500 rows, 10 IDs, HR resting 120.6 vs running 118.0.

- Lövdal original (#1) downloaded 2026-10-14 (DataverseNL zip, unzipped into data/raw/lovdal with Naman's OK). Audit run `experiments/audit/2026-10-14_0502/` (code 9b4108c, git_dirty false, 99 s): mirror check "day identical, week identical" (same sha256); #1 and #2 share the data fingerprint bf5e4401...

## In progress
- Piece 1: role per dataset (DECISIONS entries) and origin labels for #2, #6, #7 need Naman. Proposal: #2 origin -> real (byte-identical copy of #1), use and cite #1 only.
- SoccerMon (#8): download page unavailable.

## Blocked / needs human decision
- Review 2 date and target conference (ask the guide).
- Roles and origin labels for the audited datasets (Naman).
- Licences marked [VERIFY] in configs/datasets.yaml: Kaggle runners, SIRP-600, ziya07, Sport DB 2.0, SoccerMon (check each download page).

## Next three actions
1. Naman decides roles and origin labels -> DECISIONS D-012+, `role:`/`origin:` in configs/datasets.yaml, re-run audit.
2. Run the eval-auditor subagent before the audit table goes into the Review 2 deck.
3. Piece 2: loaders + person-level splits (runner loader must reorder the day/week suffixes, see below).

## Ideas for later
-

## Known issues and surprises
- requirements.lock.txt pins the CUDA build (`torch==2.14.1+cu132`). Teammates without a GPU install CPU torch first, then `pip install -r requirements.txt` (not the lock file). See docs/SETUP_WINDOWS.md.
- Sport DB 2.0: sampling rates are not in the files (implied HR ~1.005 Hz, ECG 250 samples per HR sample) [VERIFY against the dataset description]; subject folders restart per sport, so one person in two sports cannot be detected; 4 skating recordings have no subject folder.
- Runner logs (authors' README.txt): `-0.01` in the three perceived-effort columns means "rest day"; dates are a day index from the first record. GOTCHA: day file suffix `.6` = day before the event and no suffix = seven days before, while the week file's no-suffix = the week before; the loader must put both into oldest-to-newest order.
- The audit picks its worker count from free RAM; with little free RAM WESAD runs on 2-3 workers (measured peak about 1.23 GB per worker).
