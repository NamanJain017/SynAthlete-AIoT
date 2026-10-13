# STATE (living status; update at the end of every session)
Last updated: 2026-10-13 by Claude Code (Naman's laptop)

## Current piece
1 Download + audit all datasets (piece 0 done).

## Done (with evidence)
- Lean starter repo pushed to GitHub (rules, brief, dataset list, plan, task runner, guard hook).
- Piece 0 Setup (2026-10-13, commit fc3e2eb + lock file), output copied from the laptop:
  - `.venv\Scripts\python.exe --version` -> `Python 3.12.2`
  - `python tasks.py check` -> ruff `All checks passed!`, pytest `3 passed in 0.01s`
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` -> `2.14.1+cu132 True NVIDIA GeForce RTX 4050 Laptop GPU`
  - `requirements.lock.txt` created with `pip freeze` (70 pinned packages).

## In progress
- Piece 1: write src/data/audit.py (plan approved 2026-10-13) and fill the audit table in docs/DATASETS.md from its output.
- Datasets 2-7 downloaded and unzipped into data/raw/<name>/. Lövdal original (#1): retry DataverseNL on 2026-10-14. SoccerMon (#8): download page unavailable.

## Blocked / needs human decision
- Review 2 date and target conference (ask the guide).

## Next three actions
1. Run `python tasks.py audit` on the real data; fill the audit table in docs/DATASETS.md.
2. Decide each dataset's role (DECISIONS entry).
3. Loaders + person-level splits (piece 2).

## Ideas for later
-

## Known issues and surprises
- requirements.lock.txt pins the CUDA build (`torch==2.14.1+cu132`). Teammates without a GPU install CPU torch first, then `pip install -r requirements.txt` (not the lock file). See docs/SETUP_WINDOWS.md.
