# STATE (living status; update at the end of every session)
Last updated: 2026-10-13 by Claude Code (cloud session)

## Current piece
0 Setup on Naman's laptop, then 1 Download + audit all 6 datasets.

## Done (with evidence)
- Lean starter repo pushed to GitHub (rules, brief, dataset list, plan, task runner, guard hook).

## In progress
- Naman: clone repo, create venv, install CUDA PyTorch + requirements, run `python tasks.py check`.
- Naman: datasets 2-7 downloaded as zips into data/raw/<name>/ (not yet unzipped). Lövdal original (#1): retry DataverseNL on 2026-10-14. SoccerMon (#8): download page unavailable.

## Blocked / needs human decision
- Review 2 date and target conference (ask the guide).

## Next three actions
1. `python tasks.py check` green on the laptop.
2. Write src/data/audit.py and fill the audit table in docs/DATASETS.md.
3. Decide each dataset's role (DECISIONS entry), then loaders + splits.

## Ideas for later
-

## Known issues and surprises
-
