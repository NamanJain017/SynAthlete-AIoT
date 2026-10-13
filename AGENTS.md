# AGENTS.md: SynAthlete-AIoT

The one rulebook for every coding agent and every human on this repo. Details live in docs/.

## Mission
Build an explainable AIoT system that predicts athlete injury risk, plus a conference paper.
Everything shown in Review 1 stays in the project (see docs/PROJECT_BRIEF.md), each piece with an honest role.
Quality bar: honest, reproducible, leakage-free. A correct negative result beats an impressive wrong one.

## The person you work with
Naman is a final-year student who must explain every part in a viva. After building each piece:
1. Show the real output (file path and numbers).
2. Explain it in simple words: what it does, why it is needed, what the numbers mean.
3. Add the explanation and 2-3 likely viva questions with answers to docs/VIVA_NOTES.md.
Build in small pieces. Do not start the next piece until Naman says the current one is clear.

## Start of every session
1. Read docs/PROJECT_BRIEF.md, docs/STATE.md, docs/PLAN.md and docs/DECISIONS.md.
2. Say which piece you are on and what will prove it is done.
3. Plan before code for anything non-trivial; wait for approval.

## Invariants (never break; if a task seems to need it, STOP and ask)
1. Never invent, estimate or hand-type a result, metric, dataset statistic or citation. Numbers come from files written by code (experiments/**/metrics.json, data cards). If missing, run the code or say it is unavailable.
2. Split by person (subject or runner), never by window or row. Folds live in data/splits/*.json; create them once, never regenerate.
3. Fit scalers, imputers, generators and resamplers on training folds only. Synthetic data never enters validation or test.
4. Do not edit data/raw, data/splits or any metrics.json by hand.
5. Choose thresholds on validation data, never on test. Log every protocol change in docs/DECISIONS.md.
6. Never weaken, skip or delete a failing test to get green. Fix the cause or ask.
7. No claim stronger than its evidence. Synthetic data is never evidence about real physiology or real injuries. Injury claims only from real injury labels.
8. Every dataset is labelled real, simulated or unknown-origin in docs/DATASETS.md. Unknown-origin data may be used for demos or sanity checks, never as evidence.
9. References only from verified sources (DOI or publisher page). If unverified, write [VERIFY] and tell Naman.
10. No raw datasets, secrets or files over 5 MB in git.
11. Every run records seed, config.yaml, git commit and data version next to its metrics.json.
12. Attention weights are never presented as an explanation on their own; show them next to a faithfulness-tested method (SHAP or integrated gradients).
13. Log substantive AI-generated work in docs/AI_LOG.md (needed for the paper's AI-use statement).

## Stop and ask when
- a split, label definition, metric, threshold or protocol would change;
- a dataset's licence or origin is unclear;
- a result looks too good (assume leakage until proven otherwise);
- a test or check seems wrong.

## Work loop
read state -> plan -> implement in small steps -> `python tasks.py check` -> run it on real data -> explain to Naman + update docs/VIVA_NOTES.md -> update docs/STATE.md -> commit and push with the evidence in the message.

## Environment
- Windows laptop, PowerShell, Python 3.12, NVIDIA RTX 4050 (about 6 GB VRAM): small models, batch size 128 or less, checkpoint long runs, keep one run under about 2 hours, always set seeds.
- Teammates have no GPU: everything except generator and model training must also run on CPU.
- Code lives on GitHub (NamanJain017/SynAthlete-AIoT). Data stays local in data/raw/.

## Repo map
- docs/: PROJECT_BRIEF, PLAN, STATE, DECISIONS, DATASETS, DATA_CONTRACT, EVAL_PROTOCOL, VIVA_NOTES, CLAIMS, AI_LOG, SETUP_WINDOWS
- data/: raw (ignored), processed (ignored), splits, MANIFEST.csv
- src/: data, gen, pipeline, models, xai, dashboard
- experiments/<phase>/<run_id>/: config.yaml + metrics.json + logs
- configs/ YAML; tests/ pytest; paper/; slides/; backups/

## Commands
`python tasks.py check | snapshot | data | baselines | gen | pipeline | train | explain | demo | paper-figures`
A task that prints "not implemented" is a gap to fill, not something to stub green.

## Style
Python 3.12 (code stays 3.11-compatible), type hints, ruff, pytest, small pure functions, short comments that explain why, config in YAML. Notebooks for exploration only; every number in the paper comes from a script.
