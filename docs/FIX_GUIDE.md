# Fix guide: paper and report wording (task #22) and the paused task (#24)

This guide is for doing the fixes yourself, in VS Code, Antigravity or a free chat. Line numbers are from 29 Oct 2026; if they have moved slightly, search for the quoted text instead.

**Three rules that prevent mistakes:**
1. **Never type a number yourself.** Numbers come from result files or from `\commands` such as `\fusedAuc{}`, which `src/paper/figures.py` fills in. If you need a number in the text, copy it from the file named here.
2. **Change one item at a time.** After each change, run:
   ```
   python tasks.py paper-figures
   python tasks.py check
   ```
   Both must finish without errors.
3. **For a wording example to copy, open `src/paper/deck.py`.** The slides were already fixed, so their sentences are the correct wording.

**Files:** `paper/main.tex` (paper), `paper/report.tex` (report), `src/paper/figures.py` (makes the numbers and tables for both).

---

## Part A: task #22, the 8 review fixes (about 30–45 min)

### A1. Lövdal claims are too strong (review item B1). Text only, easy.
Edit four places:

| Where | Find | Replace with |
|---|---|---|
| `main.tex` ~line 23–24 (abstract) | "rebuild of a published bagged-XGBoost method reproduces its behaviour, and without its label-reading normalisation it performs like simple baselines." | "rebuild of a published bagged-XGBoost method passed a pre-registered sanity check; a variant without its label-reading normalisation was scored on different runners, so the effect of that step was not measured head-to-head." |
| `main.tex` ~line 43–44 | "A faithful reproduction of a published injury-prediction method" | "A rebuild of a published injury-prediction method that passed a pre-registered sanity check" |
| `main.tex` ~line 112 and `report.tex` ~line 115 | "passed its pre-registered check" / "passed its reproduction check" | "passed its pre-registered sanity check (bootstrap lower bound above 0.5)" |
| `main.tex` ~line 133–135 (discussion) | "and selection on test data or label-reading normalisation would have hidden this." | "this is why we report every model and kept the final test locked until the end." |

### A2. Final-test reporting is incomplete (B2). Text plus `figures.py`, medium.
1. **Move the paragraph:**
   - **`main.tex`:** move the line `\textit{Final test.} \lockboxSummary{}` (~line 130) to just after the injury paragraph (after ~line 113).
   - **`report.tex` (~line 127):** move `\lockboxSummary{}` up next to the injury results (after ~line 118).
2. **In `src/paper/figures.py`,** find `n["lockboxSummary"] =` and replace that statement with:
   ```python
   if lock:
       m = lock[-1][1]
       f = m["fused"]
       r, p, a = f["roc_auc_ci"], f["pr_auc_ci"], f["alert"]
       lov = m["lovdal_label_free_paper_split"]
       n["lockboxSummary"] = _tex(
           f"On the 10 locked runners ({f['events']:,} shared days, {f['injuries']} injuries in "
           f"{f['runners_with_injury']} runners), opened once after the evaluation freeze, the "
           f"declared fused model reached ROC-AUC {r['estimate']:.3f} ({r['low']:.3f}-{r['high']:.3f}) "
           f"and PR-AUC {p['estimate']:.3f} ({p['low']:.3f}-{p['high']:.3f}; chance "
           f"{f['test_injury_rate']:.3f}); its alert caught {a['sensitivity']:.0%} of injuries with "
           f"{a['false_alarms_per_caught_injury']:.0f} false alarms per caught injury. Secondary "
           f"results, scored once and not a comparison: label-free Lövdal variant "
           f"{lov['day']['roc_auc_ci']['estimate']:.3f} (day) and "
           f"{lov['week']['roc_auc_ci']['estimate']:.3f} (week); no model was switched. With "
           "10 runners the intervals are wide and understate the uncertainty.")
   else:
       n["lockboxSummary"] = "Pending the evaluation freeze."
   ```
   Keep the line `lock = _runs("lockbox")` just above it. Check that the indentation matches the lines around it: 4 spaces inside the function.

### A3. The comparison verdict uses ROC-AUC only (B3). Text, easy.
- **`main.tex`:** after "a difference counts only when its interval excludes zero." (~line 126), add:
  "On PR-AUC, logistic regression, CatBoost and the GRU (day file) are clearly better than the label-free variant; on ROC-AUC, logistic regression and the GRU (day). The fused model's ROC-AUC gain over its sequence arm is borderline and absent on PR-AUC. The 24 intervals are not adjusted for multiple comparisons."
- **`report.tex`:** add the same sentence after "A difference between two models counts only when its bootstrap interval excludes zero."
- The source of these facts is `experiments/comparisons/2026-10-29_020232/metrics.json`.

### A4. The augmentation sentence. Already fixed.
`figures.py` already writes PR-AUC first. Just run `python tasks.py paper-figures`.

### A5. WGAN-GP is not tuned (B5). Text plus one table line, easy.
- **`main.tex` ~line 95–96 and `report.tex` ~line 95:** after "with settings chosen by validation train-on-synthetic-test-on-real (TSTR) macro-F1", add "(WGAN-GP: one declared setting, untuned, D-029)".
- **In `figures.py`, function `table_generators`:** change `f"{_tex(name)} & ..."` to `f"{_tex('wgan_gp (untuned)' if name == 'wgan_gp' else name)} & ..."`.

### A6. Explanations: add the prior check (B6). `figures.py`, medium.
In `figures.py`, find the line that ends `f"{pc['sequence']['stability']}.")` (end of `explainSummary`). After that whole statement, but still inside `if ex:`, add:
```python
    agree = {}
    for rd in ex[-1][1]["rounds"]:
        for arm in ("tabular", "sequence"):
            for prior, v in rd[arm].get("prior_agreement", {}).items():
                if isinstance(v, dict) and "share_in_expected_direction" in v:
                    a = agree.setdefault(prior, [0, 0])
                    a[0] += v["share_in_expected_direction"] >= 0.5
                    a[1] += 1
    n["explainSummary"] += " Against the expectations written before running: " + "; ".join(
        f"{k.replace('_', ' ')} agreed in {x} of {t} arm-rounds" for k, (x, t) in agree.items()
    ) + ". Attention and integrated gradients often disagree."
```

### A7. Hand-typed numbers. Slides only, already fixed.
Nothing to do in the paper.

### A8. Calibration and alert results are promised but not shown (B8). Text, easy.
The simplest honest fix is to state what is reported:
- **`main.tex` ~line 78:** replace "Calibration is reported" and the rest of that sentence with "Alert results are reported for the final locked test; per-round calibration and alert results are in the result files."
- **`report.tex` ~line 62:** the same wording.
- The alert numbers then appear through the A2 sentence.

### A9. Small extras (non-blocking, 5 min)
- **`main.tex` ~line 110:** replace "differences smaller than the spread between rounds" with "only a few pre-declared pairs differ clearly (below)".
- **`report.tex` ~line 139:** replace "the remaining work is the final locked test," with "the final locked test confirmed the development result;".
- **`report.tex` ~line 73:** replace "cloud (deep model)" with "cloud (daily injury model)".
- **Tiers text:** "\tiersReduction{}" shows PAMAP2 only. Add "(85$\times$ for the Sport DB 2.0 heart-rate stream)" after it in `main.tex` (Tiers paragraph) and `report.tex`.

**When all items are done:**
1. Run `python tasks.py paper-figures --deck` and `python tasks.py check`.
2. Commit with the message "Paper and report: apply review fixes (docs/REVIEW_2026-10-29.md)".
3. Push.

---

## Part B: task #24, the paused task

### B1. Verify the 14 references (about 45–60 min)
**File:** `paper/refs.bib`. Each unchecked entry has `note = {[VERIFY]}`. They are around lines 100–154 (ISWC 2012, NeurIPS 2019, 2017 entries, the TimeVAE arXiv 2111.08095 entry, ICMI 2017 and 2018, the TCN arXiv 1803.01271 entry, NeurIPS 2017, ICML 2017, a 2018 entry, Scientific Data 2024, and the "AIoT framework" entry in Applied Sciences 2026).

For each entry:
1. Search its title on Google Scholar or the publisher site.
2. Open the **official page**: the DOI link (doi.org/…), the publisher page, or arXiv.
3. Check that the authors, title, venue, year and pages match exactly. Fix any difference.
4. Add `doi = {…}` if there is one, then delete `note = {[VERIFY]}`.
5. **If you cannot find an official page, do not cite it.** Remove the entry and the `\cite{…}` that uses it, and rephrase that sentence.
6. **The `aiotframework2026` entry has unknown authors and title, but it supports the research-gap sentence** (`main.tex` ~line 31–32, slide 2). Find the real paper in Applied Sciences vol. 16 no. 9 article 4542 at mdpi.com. If you cannot find it, remove the citation and soften the sentence to "recent frameworks describe … but leave prediction unimplemented".

**Never let an AI chat give you a reference from memory.** Only use a reference whose page you have opened yourself.

### B2. Verify dataset licences (about 20–30 min)
**File:** `configs/datasets.yaml`. Look for `licence: "[VERIFY]"` (lines ~32, 46, 62, 78, 95, 109, 139: Kaggle runners, PAMAP2's UCI page, WESAD, Sport DB 2.0, SIRP-600, ziya07, SoccerMon).

For each one:
1. Open the dataset's download page. The URL is in the same entry in `datasets.yaml`.
2. Copy the licence name exactly, for example "CC BY 4.0", into `licence:`, and delete "[VERIFY]".
3. Also update the same dataset in `docs/DATASETS.md` and `docs/data_cards/<name>.md` (these are regenerated by `python tasks.py data`, so changing `datasets.yaml` and re-running that is enough).
4. **If a licence forbids your use** (for example, no redistribution), never publish that data or anything synthetic made from it on GitHub. WESAD is non-commercial academic use only.
5. **Sport DB 2.0 sampling rate** (line ~84): check the dataset's description page. If it confirms about 1 Hz, delete "[VERIFY]".
6. **TimeVAE venue** (`configs/gen/generators.yaml` line ~42): check arXiv 2111.08095 for a published venue, then update the comment.

### B3. Compile the paper and report to PDF (about 15–30 min)
Easiest, with nothing to install: **Overleaf** (overleaf.com, free).
1. On Overleaf: New Project → Upload Project, then upload a zip of the **`paper/`** folder. It must include `main.tex`, `report.tex`, `refs.bib` and `generated/`.
2. In Menu → Main document, choose `main.tex` and press **Recompile**. Then do the same for `report.tex`.
3. **If you see an error,** click it, and Overleaf shows the line:
   - "Undefined control sequence \something" means a `\command` is missing from `generated/numbers.tex`. Run `python tasks.py paper-figures` again and re-upload `generated/`.
   - "Missing $" usually means a `%` or `_` in text. In LaTeX write `\%` and `\_`.
4. Download the PDFs. You can keep them in `paper/` and commit them, since they are small.

### Still open, your choice
- **Conference name and author names:** these are placeholders now. Fill them into `paper/main.tex` (the title and author block) and slide 1 (`deck.py`, the line with "[Team names]") when decided.
