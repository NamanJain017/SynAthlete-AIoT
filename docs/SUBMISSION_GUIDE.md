# Submission guide: paper and report

## 1. Paper (LaTeX, Overleaf)
- **Ready-made PDF:** `paper/final_paper.pdf` (compiled locally).
- **To edit on Overleaf:**
  1. Zip these items from `paper/`:
     - `final_paper.tex`
     - `refs_final.bib`
     - the whole `generated/` folder
     - the whole `figures/` folder
  2. In Overleaf choose **New Project → Upload Project** and select the zip.
  3. Choose **Menu → Main document → final_paper.tex** and press **Recompile**. The compiler can stay pdfLaTeX.
- **Fill in before submitting:**
  - author names and the department, in the `\author{...}` block near the top;
  - the conference or journal name, if one is required.
- **Never type numbers into the text.** They come from `generated/numbers.tex`. To refresh them after any new result, run `python tasks.py paper-figures` and upload `generated/` again.

## 2. Report (Word)
- **File:** `report/SynAthlete_Report.docx`.
- **After opening it in Word:**
  1. Press **Ctrl+A**, then **F9** (choose "Update entire table" if asked). This refreshes the table of contents, the lists of figures and tables, and the page numbers.
  2. Replace every placeholder in square brackets: student names and register numbers, guide name, designation, department, specialization, signatures.
  3. Check the cover page and the bonafide certificate against your department's sample. The Head of Department name is copied from the SRM template.
- **To rebuild after a change:**
  1. Edit the text in `report/report_content.md`.
  2. Run `python -m src.paper.report_docx`.
  3. Then either repeat step 1 above in Word, or run `powershell -File report\update_fields.ps1`.

## 3. Plagiarism check (needed for both, under 10%)
- Upload the paper PDF and the report to the plagiarism tool your department uses (e.g. Turnitin, through your guide).
- The text was written for this project, not copied. References and quotations are normally excluded from the score; ask your guide to set "exclude bibliography".
- **If the paper and the report match each other,** that is expected (self-overlap). Submit them separately or ask the checker to exclude your own earlier submission.
- **Save the plagiarism report PDF** and put it in the report's Appendix C.

## 4. References rule (faculty)
- At least 20 references from ScienceDirect journals and IEEE Transactions only. The paper has **25** such articles, each checked on its DOI record (`docs/REFERENCES_VERIFIED.md`).
- The 4 dataset papers are extra references, credited because we used their data.
- To add a reference:
  1. Add its DOI to `src/paper/references.py`.
  2. Run `python -m src.paper.references`.
  3. Cite it with `\citep{key}` in the paper, or `[@key]` in `report/report_content.md`.

## 5. Proof for Review 3
- **Appendix A/B of the report:** attach the paper submission email or acceptance proof.
- **Appendix C:** attach the plagiarism report.
- **The demo:** `python tasks.py demo` (see `docs/FIGURES_GUIDE.md` and the demo steps in our chat).
