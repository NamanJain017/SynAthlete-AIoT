# SynAthlete-AIoT: the whole project, explained simply

*Written 29 Oct 2026 for Naman and anyone who wants to understand this project from scratch. You need no background. Every number here was produced by code and is saved in a result file; the file is named next to the number, so you can always check it.*

---

## 1. The project in one paragraph

Coaches cannot see an athlete's injury risk building up day by day. We built a complete, **honest**, **explainable** AIoT system that:

1. learns injury risk from **real** injury records of runners;
2. tests whether **synthetic (fake but realistic) sensor data** can help;
3. runs as an **edge–fog–cloud** IoT design, where the wearable does quick checks itself;
4. **explains** every risk score in plain words, with tests that prove the explanations are faithful;
5. shows everything on a **coach dashboard**.

The most important word is *honest*. Many published sports-AI results look good only because the test data leaked into training. We built the project so that cannot happen, and we report weak results as weak.

---

## 2. The problem, and the gap we found

- **Problem:** injuries come from training load building up over days and weeks. A coach looking at numbers cannot easily see which runner is drifting into danger.
- **What exists:** wearable datasets have heart rate and motion, but **no injury labels**. Injury datasets have injuries, but only **daily training logs**, not sensor streams.
- **Our corrected research gap (decision D-005):** *"No public dataset pairs high-frequency physiological streams with real injury outcomes."* That is why the project has two tracks:
  - **Track A (injury evidence):** real runners' daily logs with real injuries (Lövdal dataset). This is the **only** source of claims about injuries.
  - **Track B (IoT and sensors):** wearable streams (PAMAP2, WESAD, Sport DB 2.0). Used for the activity model, the IoT tiers and the synthetic-data generators. They are **different people** from the runners, so they are never used as injury evidence.
- **Second problem:** published accuracies are often inflated by evaluation shortcuts (leakage). We treat honest evaluation as a main contribution.

---

## 3. Our idea (the solution)

| Review 1 objective | What we built | Where |
|---|---|---|
| 1. Bridge the data gap with synthetic data | TimeGAN, WGAN-GP and TimeVAE generators, plus classic augmentation (jitter, scaling, time-warp) | `src/gen/` |
| 2. Edge–fog–cloud IoT design | Three separate programs: edge "reflex" (checks each 5-second window), fog "supervisor" (session summaries), cloud "deep" (daily injury model) | `src/pipeline/` |
| 3. A fused risk model | Sequence model (GRU/LSTM/TCN) on day-by-day data + tree model (Random Forest/CatBoost) on weekly totals, combined by a small "stacking" model; risk shown as 0–100 | `src/models/fused.py` |
| 4. Validated explanations | SHAP, integrated gradients, attention view, Grad-CAM, each checked by 3 tests | `src/xai/` |
| 5. Coach dashboard | Streamlit app, 8 pages | `src/dashboard/` |
| Honest testing | 7 baseline models, a leak alarm, a rebuild of the published method, a locked final test | `src/models/`, `src/data/lockbox.py` |

---

## 4. The datasets (each has exactly one honest role)

| Dataset | What it is | Real? | Its role in our project |
|---|---|---|---|
| **Lövdal et al. 2021** (runners) | 74 competitive runners, about 7 years of daily training logs, with real injury labels. Day file: 42,766 windows with 583 injuries (1.36 %). Week file: 42,798 windows with 575 injuries. | Real | **The only injury evidence.** Every injury result comes from here. |
| **PAMAP2** | 9 people wearing sensors (heart rate and 3 accelerometers), 10.7 hours, 18 activity types | Real | Activity recognition model, edge-tier stream, generator training |
| **WESAD** | 15 people, chest and wrist sensors, 24.13 hours | Real | Checks whether the generators work on a second dataset |
| **Sport DB 2.0** | 168 heart-rate recordings (55 continuous) | Real | Realism check for synthetic heart rate; second edge stream |
| **SoccerMon / PMData** | Football players' (50) and people's (15) self-reported daily data with a few injury episodes | Real | "Track C": a small exploratory extra, not main evidence |
| **SIRP-600** | 600 rows, origin unknown | Unknown | Demo only: shows the SHAP code runs. Never evidence. |
| **ziya07** | 500 rows of wearable-like data | Unknown | Demo stream on the dashboard only |

All datasets are listed with their origin in `docs/DATASETS.md`, and each has a data card in `docs/data_cards/`. Raw data stays on the laptop (`data/raw/`, never on GitHub).

---

## 5. The rules that keep everything honest (and why)

1. **Split by person, never by row.** A runner is in training *or* testing, never both. Otherwise the model memorises a person, and the test score looks better than reality.
2. **Five development rounds.** The 64 development runners (IDs 0–63) are split into 5 rounds. Each round has its own training, validation and test runners. The headline score is the average over the 5 rounds, because one split of about 13 runners is too noisy.
3. **A locked final test.** 10 runners (IDs 64–73, the newest) were sealed away from day one. They could be opened only **once**, by one program, after the code was frozen with a Git tag called `eval-freeze`. Every opening is logged. This proves we did not tune on them.
4. **Everything is fitted on training runners only.** This covers scalers, generators and resamplers. Settings, calibration and alert thresholds are chosen on **validation** runners, never on test runners.
5. **No hand-typed numbers.** Every number in the paper and slides is inserted by code from result files (`python tasks.py paper-figures`).
6. **Leak alarm ("canary").** We train a model on shuffled (meaningless) labels. If it scores above chance, something leaks. It scored 0.498 (day) and 0.513 (week): no leak.
7. **Pass marks written before running.** Generator and explanation criteria were fixed in advance (decision D-027), so we could not move the goalposts.
8. **Every decision is logged** in `docs/DECISIONS.md` (D-001 to D-034).

---

## 6. How scores are measured (simple meanings)

- **ROC-AUC:** "If I pick one injured case and one healthy case at random, how often does the model give the injured one a higher risk?" 0.5 = coin flip, 1.0 = perfect.
- **PR-AUC (average precision):** better for rare events. Chance level = the injury rate (about 0.013 here). So 0.023 means about 1.7 times better than chance.
- **Macro-F1:** for activity recognition. It averages how well each activity is recognised, so rare activities count equally.
- **95 % interval (runner bootstrap):** we re-draw the test runners 2,000 times and recompute. The middle 95 % of results shows how much the score could move by luck. If an interval for a *difference* includes 0, we do **not** claim one model is better.
- **"± number":** the spread (standard deviation) across the 5 development rounds.

---

## 7. What we built and ran, piece by piece

The project was built in pieces 0–10 (plan in `docs/PLAN.md`; status in `docs/STATE.md`).

### Pieces 0–2: setup, data audit, loaders and splits
- Downloaded and audited every dataset (`experiments/audit/`), with a SHA-256 fingerprint of every raw file (`data/MANIFEST.csv`).
- Made person-level splits once and sealed them with checksums (`data/splits/`).
- A fresh copy of the repository rebuilt every file **identically** (`experiments/repro/2026-10-22_0731/`).

### Piece 3: real-data baseline models (injury and activity)
Results are means over the 5 development rounds. Folders: `experiments/baselines/2026-10-26_020012`, `2026-10-26_022902`, `2026-10-28_024255`, `2026-10-28_224339`.

| Model | Why we trained it | ROC-AUC day | ROC-AUC week |
|---|---|---|---|
| Constant risk | "Do nothing" reference | 0.5 (chance) | 0.5 |
| ACWR-style rule | The classic coaching rule (acute vs chronic load) | – | 0.540 |
| Logistic regression | Simple, transparent | 0.626 | 0.621 |
| Random forest | Tree ensemble | 0.603 | 0.614 |
| CatBoost | Strong boosted trees | 0.598 | 0.606 |
| GRU (neural network) | Learns day-by-day patterns | 0.631 (PR-AUC 0.023) | – |
| Lövdal method, label-free | The published method without its "label peek" | 0.584 | 0.589 |

- **Rebuild of the published Lövdal method:** an exact copy, including its scaling step, run on stand-in runners 54–63, passed its pre-registered sanity check (day ROC-AUC 0.617, interval 0.573–0.664). Its scaling step reads the injury labels, which a real system cannot do. That is why our main comparison uses the "label-free" version.
- **Activity recognition (PAMAP2, people never seen):** CNN macro-F1 0.632; GRU 0.566.

### Piece 4: evaluation rules
All pass marks were written in advance (`docs/EVAL_PROTOCOL.md`, D-027):
- **Generators:** real-vs-fake accuracy within 0.15 of 50 %; train-on-synthetic score close to real; shape and correlation errors below 0.10; privacy (no copied windows).
- **Explanations:** deletion, randomisation and stability tests.

### Piece 5: synthetic data generators
Folder: `experiments/generators/2026-10-28_172535/`. PAMAP2, 4 rounds, about 3 h.

| Generator | Real-vs-fake accuracy (0.5 = fooled) | Train-on-synthetic F1 | Real + synthetic F1 | Pass marks met |
|---|---|---|---|---|
| TimeGAN | 0.956 | 0.270 | 0.639 | 0 of 5 |
| WGAN-GP (one fixed setting, untuned, D-029) | 0.898 | 0.564 | 0.637 | 1 of 5 |
| TimeVAE | 0.936 | 0.525 | 0.647 | 2 of 5 |
| Classic augmentation | 0.860 | 0.624 | 0.636 | 2 of 5 |

Real-only F1 is 0.632.

- **WESAD (second dataset, settings copied, no tuning):** no generator met the realism criteria; only TimeGAN met privacy.
- **Synthetic injury examples** (added to training only; follow-up stage took 3.4 h): GRU ROC-AUC 0.633 (TimeGAN) and 0.617 (classic), against 0.631 real-only. On PR-AUC both are slightly **lower** than real-only: 0.0220 and 0.0210 against 0.0232. So: **no gain.**
- **Meaning:** a real-vs-fake checker beats every generator, so the synthetic data is not realistic enough to replace real data. It is used for training only and never as evidence.

### Piece 6: edge–fog–cloud tiers
- **What it is:** three separate programs talking through queues, simulating a wearable (edge), a phone or gateway (fog), and a server (cloud).
- **Benchmark** (`experiments/tiers/2026-10-29_025032`, idle laptop, 8 minutes): we compare "tiered" (edge sends small summaries) with "cloud-only" (every raw window goes to the cloud).

| Stream | Bytes into cloud: tiered vs cloud-only | Median decision time: tiered vs cloud-only |
|---|---|---|
| PAMAP2 wearable | 2,932 vs 1,739,186 (**593× less**) | 4.25 ms vs 2.77 ms |
| Sport DB heart rate | 1,600 vs 135,619 (**85× less**) | 0.78 ms vs 1.00 ms |

- **Meaning:** edge processing cuts cloud traffic by 85–593 times. Everything ran on one laptop, so there is no network delay; decision time is similar either way. The main claim is the data saving.
- **The measurement was fixed before the official run (D-034).** The first trial "measured" about 3 seconds per window. That was really windows waiting while programs started up. Now all tiers start first and windows arrive at a steady pace.

### Piece 7: the fused model and the locked final test
- **Fused model** (`experiments/fused/2026-10-28_060435`, about 5 h for 5 rounds in parallel):
  - 44 settings tried per round, chosen on validation runners;
  - 3 random seeds;
  - risk shown as 0–100 (the percentile among validation cases).
  - **Result:** ROC-AUC 0.609 ± 0.024, PR-AUC 0.022 ± 0.003.
- **Locked final test** (`experiments/lockbox/2026-10-29_010127`, 54 min, opened once at 01:54 on 29 Oct; log in `experiments/lockbox_access.log`):
  - **Fused model on the 10 locked runners:** ROC-AUC **0.605** (interval 0.535–0.699), PR-AUC 0.030, on 2,994 days with 48 injuries (7 of the 10 runners were injured).
  - **It matches development (0.609).** This is strong evidence that our development results were honest and not over-tuned.
  - **Alert level** (chosen on validation): it catches about 60–69 % of injuries, but with about 46 false alarms per injury caught, so about half of all days get an alert. The alert is too noisy for real use.
  - **Secondary results** (scored once, never used to choose anything):
    - GRU arm on the day file: 0.609;
    - CatBoost arm on the week file: 0.567;
    - Lövdal label-free model trained on runners 0–63: 0.649 (day) and 0.602 (week).
- **Important:** the Lövdal model's 0.649 is **not** a fair head-to-head with 0.605. It was scored on different rows, trained on more runners (64 vs 51), and the intervals overlap. Switching to it after seeing the test would be cheating. The fused model was declared in advance. The honest verdict: **no model is clearly best.**
- **Before opening:** an independent reviewer checked the final-test program. Its fixes were applied (D-031, D-032). A dry run on development runners gave exactly the same numbers before and after the fixes.

### Paired comparisons (the fair way to compare models)
Folder: `experiments/comparisons/2026-10-29_020232`. Same rows, runner bootstrap, about 2.5 h.

- **Against the Lövdal label-free model:**
  - **On ROC-AUC,** logistic regression (day, +0.042) and GRU (day, +0.047) are clearly better.
  - **On PR-AUC,** CatBoost (day) is also just clearly better.
  - The others are not clearly different.
- **Fused against its own parts:** a tiny ROC-AUC gain over the day-sequence part (+0.005, interval just above 0), no PR-AUC gain, and no gain over the weekly part.
- **Note:** 24 intervals were computed without correcting for multiple comparisons, so borderline ones should be read with care.

### Piece 8: explanations
Folder: `experiments/explain/2026-10-28_205117`.

- **Methods:** SHAP (tree part), integrated gradients (sequence part), an attention view (always shown next to integrated gradients, never alone), and Grad-CAM (activity CNN).
- **Three faithfulness tests:** deletion, randomisation and stability. **All passed in 5 of 5 rounds.**
- **Honest details:**
  - **Prior list:** before running, we wrote down what we expected to raise risk: load spikes, monotony and low recovery. Load mostly agreed. "Low recovery raises risk" was mostly **not** seen.
  - **Attention disagrees with integrated gradients.** That is why attention is never shown alone.
- **Plain-language reasons:** each risk score gets a sentence, generated by code, such as *"Risk is higher mainly because … was higher than usual."*

### Piece 9: the dashboard
- **Start it:** `python tasks.py demo`. Data is prepared in advance with `python tasks.py demo --prepare`, so the demo is instant and cannot crash during a live review.
- **8 pages:** Home → Athletes → Athlete detail (risk 0–100 and "why") → Live sensor (edge stream) → Synthetic data → Results → Training lab (try settings safely; never touches official results or locked runners) → Data & honesty.
- **Use:** for a coach, it shows who is at risk and why. For examiners, it shows every part of the system working, with honest labels. The coach pages replay development runners, never the locked ones.

### Piece 10: paper, report, slides
- `paper/main.tex`: IEEE conference paper draft.
- `paper/report.tex`: project report.
- `slides/review2.pptx`: Review 2 deck, 17 slides with speaker notes. Rebuild it with `python tasks.py paper-figures --deck`.
- The conference and author names are not decided yet, so placeholders stay.

### Track C (small extra)
SoccerMon logistic regression reached ROC-AUC 0.764, but on only 27 injury episodes. PMData was about chance. It is exploratory only (`experiments/track_c/2026-10-28_055034`).

---

## 8. Final verdict

1. **Injury risk from training logs is predictable, but only weakly.** Honest models reach ROC-AUC of about 0.58–0.63, and the locked test confirmed 0.605. That is better than chance but far from a reliable alarm (about 46 false alarms per caught injury).
2. **Our evaluation is trustworthy.** The locked test matched development, the leak alarm stayed at chance, and every number is reproducible from code.
3. **The published method's high score depends on a scaling step that reads labels.** Without that step it performs at a modest level. This is a useful finding, but its exact size was not measured head-to-head.
4. **Synthetic sensor data is not realistic enough yet** and did not help injury prediction. That is an honest negative result.
5. **The IoT design works.** Edge processing sends 85–593 times less data to the cloud.
6. **The explanations are faithful by test**, but some expected patterns (low recovery) did not appear.
7. **No single model is clearly best.**

A correct negative result is worth more than an impressive wrong one. That is the project's message.

---

## 9. Where things are (folder map)

| Folder or file | What it holds |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | Rules for every AI tool and person working on the repo |
| `docs/` | Plan, status (`STATE.md`), decisions (`DECISIONS.md`), datasets, evaluation protocol, viva Q&A (`VIVA_NOTES.md`), AI-use log (`AI_LOG.md`) |
| `data/` | `raw/` (local only), `processed/` (local only), `splits/` (sealed person-level splits), `MANIFEST.csv` |
| `src/data/` | Loading, audit, splits, the lockbox guard |
| `src/models/` | Baselines, neural networks, tuning, fused model, final test (`final.py`) |
| `src/gen/` | Generators and their checks |
| `src/pipeline/` | Edge–fog–cloud tiers and benchmark |
| `src/xai/` | Explanations and their tests |
| `src/dashboard/` | Streamlit app |
| `src/paper/` | Makes paper numbers, tables and slides from result files |
| `experiments/` | Every run: `config.yaml`, `run_info.json` (commit, seeds), `metrics.json` |
| `tests/` | 289 automated tests (`python tasks.py check`) |
| `paper/`, `slides/` | Paper, report, deck |

**Useful commands:** `python tasks.py check` (all tests), `python tasks.py demo` (dashboard), `python tasks.py paper-figures --deck` (rebuild paper numbers and slides).

---

## 10. What is still open (as of 29 Oct, 03:45)

1. **Task #22, review fixes.** An independent review found no leakage and no wrong numbers. It did find 8 places where the paper and slides word or choose results too favourably. The full list, with exact fixes, is in `docs/REVIEW_2026-10-29.md`.
2. **Task #23, dashboard Results page.** Add the locked test, the comparisons and the WESAD and synthetic-injury results.
3. **Task #24, paused by Naman:**
   - verify 14 references marked [VERIFY] in `paper/refs.bib`;
   - confirm dataset licences;
   - compile the paper and report to PDF.
4. One red GitHub check, from the tag push. It is explained in D-033 and fixed for future runs.
