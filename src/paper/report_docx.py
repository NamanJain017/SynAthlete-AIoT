"""python -m src.paper.report_docx: the project report (report/SynAthlete_Report.docx), SRM format.

Follows the SRM "Manual for preparation of B.Tech project report": A4; margins top/bottom 28 mm,
left 38 mm, right 22 mm; Times New Roman 12 with 1.5 spacing; front matter numbered in roman,
chapters in arabic; "CHAPTER n" headings; figure captions below and table captions above,
numbered per chapter; references numbered in order of citation in the manual's style.
The text lives in report/report_content.md; every result number and results table is read from
the saved result files (figures.numbers, metrics.json), never typed (AGENTS.md invariant 1).
Tables of contents, figures and tables are Word fields: report/update_fields.ps1 fills them.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
import zipfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt

from src.dashboard.prepare import DATASETS
from src.data import loaders
from src.paper import deck, figures

ROOT = loaders.ROOT
CONTENT = ROOT / "report" / "report_content.md"
OUT = ROOT / "report" / "SynAthlete_Report.docx"
FIG = ROOT / "figures"
REFS = ROOT / "paper" / "refs_final.json"
TEMPLATE = ROOT / "EDITED-B.Tech project and internship report Manual.docx"
LOGO = ROOT / "report" / "srm_logo.jpeg"
TITLE = ("SynAthlete-AIoT: An Honest and Explainable AIoT Pipeline for Athlete Injury-Risk "
         "Prediction with Synthetic Wearable Data")
FONT = "Times New Roman"


# ------------------------------------------------------------------------------- word helpers
def _field(par, instr: str, placeholder: str = "") -> None:
    """A Word field (PAGE, TOC ...); Word fills it when fields are updated."""
    def run(tag, **attrs):
        r = OxmlElement("w:r")
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        if tag == "w:instrText":
            el.set(qn("xml:space"), "preserve")
            el.text = instr
        r.append(el)
        par._p.append(r)
    run("w:fldChar", **{"w:fldCharType": "begin"})
    run("w:instrText")
    run("w:fldChar", **{"w:fldCharType": "separate"})
    if placeholder:
        par.add_run(placeholder)
    run("w:fldChar", **{"w:fldCharType": "end"})


def _font(style, size: float, bold: bool = False, italic: bool = False) -> None:
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = None
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(k), FONT)


def _styles(doc) -> None:
    st = doc.styles
    _font(st["Normal"], 12)
    pf = st["Normal"].paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size, align in (("Heading 1", 14, WD_ALIGN_PARAGRAPH.CENTER),
                              ("Heading 2", 12, WD_ALIGN_PARAGRAPH.LEFT),
                              ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT)):
        _font(st[name], size, bold=True)
        st[name].paragraph_format.alignment = align
        st[name].paragraph_format.space_before = Pt(12)
        st[name].paragraph_format.space_after = Pt(6)
        st[name].paragraph_format.keep_with_next = True
    st["Heading 1"].paragraph_format.space_after = Pt(24)
    for name, size, bold, align in (("FrontHeading", 14, True, WD_ALIGN_PARAGRAPH.CENTER),
                                    ("FigCaption", 11, False, WD_ALIGN_PARAGRAPH.CENTER),
                                    ("TabCaption", 11, False, WD_ALIGN_PARAGRAPH.CENTER),
                                    ("Cell", 10, False, WD_ALIGN_PARAGRAPH.LEFT),
                                    ("RefItem", 11, False, WD_ALIGN_PARAGRAPH.LEFT),
                                    ("Center", 12, False, WD_ALIGN_PARAGRAPH.CENTER),
                                    ("Code", 8.5, False, WD_ALIGN_PARAGRAPH.LEFT)):
        s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st["Normal"]
        _font(s, size, bold=bold)
        s.paragraph_format.alignment = align
        if name in ("FigCaption", "TabCaption", "Cell", "RefItem", "Code"):
            s.paragraph_format.line_spacing = 1.0
        if name == "FrontHeading":
            s.paragraph_format.space_after = Pt(24)
        if name in ("FigCaption", "TabCaption"):
            s.paragraph_format.space_before = Pt(4)
            s.paragraph_format.space_after = Pt(12)
    st["TabCaption"].paragraph_format.keep_with_next = True
    code = st["Code"]
    code.font.name = "Consolas"
    code.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:ascii"), "Consolas")
    code.element.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:hAnsi"), "Consolas")
    code.paragraph_format.space_after = Pt(0)
    for name in ("List Bullet",):
        _font(st[name], 12)


def _page_numbers(section, fmt: str, start: int = 1) -> None:
    sect = section._sectPr
    pg = sect.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sect.append(pg)
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))


def _margins(section) -> None:
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.top_margin = section.bottom_margin = Mm(28)
    section.left_margin, section.right_margin = Mm(38), Mm(22)
    section.header_distance = Mm(15)


def _header_number(section) -> None:
    section.header.is_linked_to_previous = False
    p = section.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _field(p, "PAGE", "1")


# ------------------------------------------------------------------------------- content helpers
def _numbers() -> dict[str, str]:
    n = {k: deck._plain(v).replace("``", "“").replace("''", "”").replace("\\_", "_")
         for k, v in figures.numbers(figures.baselines()).items()}
    ex = figures._runs("explain")
    reasons = ex[-1][1]["rounds"][0].get("reasons", []) if ex else []
    n["realReason"] = reasons[0]["sentence"] if reasons else "n/a"
    return n


def _initials(given: str) -> str:
    parts = re.split(r"(\s+|-)", given.strip())
    return "".join(p[0] + "." if p and p.strip() and p != "-" else ("-" if p == "-" else " ")
                   for p in parts).replace(" .", "").strip()


def _ref_text(r: dict) -> list[tuple[str, bool]]:
    """One reference in the manual's style, as (text, bold) pieces."""
    names = [f"{a.get('family', '')}, {_initials(a.get('given', ''))}".strip(", ")
             for a in (r.get("author") or [])]
    who = names[0] if len(names) == 1 else (", ".join(names[:-1]) + ", and " + names[-1])
    year = (r.get("issued") or {}).get("date-parts", [[""]])[0][0]
    where = r.get("container-title") or r.get("publisher") or ""
    where = where[0] if isinstance(where, list) else where
    title = (r.get("title") or "").strip()
    pages = r.get("page") or r.get("article-number")
    out = [(f"{who}, “{title},” {where}", False)]
    if r.get("volume"):
        out += [(", ", False), (str(r["volume"]), True)]
        if r.get("issue"):
            out.append((f"({r['issue']})", False))
    if pages:
        out.append((f", pp. {pages}" if "-" in str(pages) or "–" in str(pages)
                    else f", Art. no. {pages}", False))
    out.append((f", {year}. https://doi.org/{r['DOI']}", False))
    return out


class Report:
    def __init__(self) -> None:
        self.doc = Document()
        _styles(self.doc)
        self.numbers = _numbers()
        self.refs = json.loads(REFS.read_text("utf-8"))
        self.cited: list[str] = []
        self.chapter = 0
        self.fig = self.tab = self.eq = 0

    # --- text with **bold**, [@cites] and {numbers}
    def _cite(self, m: re.Match) -> str:
        nums = []
        for key in [k.strip().lstrip("@") for k in m.group(1).split(",")]:
            if key not in self.refs:
                raise KeyError(f"unknown reference {key}")
            if key not in self.cited:
                self.cited.append(key)
            nums.append(self.cited.index(key) + 1)
        return "[" + ", ".join(str(v) for v in sorted(nums)) + "]"

    def _resolve(self, text: str) -> str:
        text = re.sub(r"\[(@[^\]]+)\]", self._cite, text)
        return re.sub(r"\{(\w+)\}", lambda m: self.numbers.get(m.group(1), m.group(0)), text)

    def para(self, text: str, style: str | None = None):
        p = self.doc.add_paragraph(style=style)
        for i, chunk in enumerate(re.split(r"\*\*", self._resolve(text))):
            if chunk:
                p.add_run(chunk).bold = i % 2 == 1
        return p

    # --- figures, tables, equations
    def figure(self, name: str, caption: str) -> None:
        self.fig += 1
        p = self.doc.add_paragraph(style="Center")
        p.paragraph_format.keep_with_next = True
        width = Cm(14.5) if name != "13_confusion_activity.png" else Cm(12.5)
        p.add_run().add_picture(str(FIG / name), width=width)
        self.doc.add_paragraph(f"Figure {self.chapter}.{self.fig} {caption}", style="FigCaption")

    def table(self, header: list[str], rows: list[list[str]], caption: str,
              widths: list[float] | None = None) -> None:
        self.tab += 1
        self.doc.add_paragraph(f"Table {self.chapter}.{self.tab} {caption}", style="TabCaption")
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, h in zip(t.rows[0].cells, header, strict=True):
            cell.paragraphs[0].style = "Cell"
            cell.paragraphs[0].add_run(h).bold = True
        for row in rows:
            cells = t.add_row().cells
            for cell, v in zip(cells, row, strict=True):
                cell.paragraphs[0].style = "Cell"
                cell.paragraphs[0].add_run(self._resolve(str(v)))
        if widths:
            for row in t.rows:
                for cell, w in zip(row.cells, widths, strict=True):
                    cell.width = Cm(w)
        self.doc.add_paragraph()

    def _box(self, lines: list[tuple[str, str, bool]]) -> None:
        """A one-cell bordered box: (text, style, bold) per line."""
        t = self.doc.add_table(rows=1, cols=1)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.rows[0].cells[0]
        cell.width = Cm(14.5)
        first = True
        for text, style, bold in lines:
            par = cell.paragraphs[0] if first else cell.add_paragraph()
            first = False
            par.style = style
            par.add_run(self._resolve(text) if style != "Code" else text).bold = bold

    def code(self, target: str, caption: str) -> None:
        """A listing taken from the source file itself, so it always matches the code."""
        mod, attr = target.rsplit(".", 1)
        src = inspect.getsource(getattr(importlib.import_module(mod), attr)).rstrip()
        self._box([(line.replace(chr(9), "    "), "Code", False) for line in src.splitlines()])
        self.fig += 1
        self.doc.add_paragraph(f"Figure {self.chapter}.{self.fig} Listing: {caption} "
                               f"({mod.replace('.', '/')}.py)", style="FigCaption")

    def algorithm(self, title: str, steps: list[str]) -> None:
        self.tab += 1
        lines = [(f"Algorithm {self.chapter}.{self.tab}: {title}", "Cell", True)]
        lines += [(f"Step {i}. {st.strip()}", "Cell", False) for i, st in enumerate(steps, 1)]
        self.doc.add_paragraph(f"Table {self.chapter}.{self.tab} Algorithm: {title}",
                               style="TabCaption")
        self._box(lines)
        self.doc.add_paragraph()

    def equation(self, text: str) -> None:
        self.eq += 1
        p = self.doc.add_paragraph(style="Center")
        p.paragraph_format.tab_stops.add_tab_stop(Cm(7.5), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(15.0), WD_TAB_ALIGNMENT.RIGHT)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.add_run("\t")
        p.add_run(text).italic = True
        p.add_run(f"\t({self.chapter}.{self.eq})")

    # --- front matter
    def front_heading(self, text: str, new_page: bool = True) -> None:
        p = self.doc.add_paragraph(text, style="FrontHeading")
        p.paragraph_format.page_break_before = new_page

    def cover(self) -> None:
        d = self.doc
        if LOGO.exists():
            d.add_paragraph(style="Center").add_run().add_picture(str(LOGO), width=Cm(9))
        lines = [(TITLE.upper(), 18, True), ("", 12, False),
                 ("21CSE292P - ARTIFICIAL INTELLIGENCE OF THINGS", 14, True),
                 ("PROJECT REPORT", 14, True), ("", 12, False),
                 ("Submitted by", 14, False),
                 ("[STUDENT 1 NAME] [REG. NO.]", 16, True),
                 ("[STUDENT 2 NAME] [REG. NO.]", 16, True),
                 ("", 12, False), ("Under the Guidance of", 14, False),
                 ("[GUIDE NAME]", 16, True), ("([Designation], [Department])", 12, False),
                 ("", 12, False), ("in partial fulfillment of the requirements for the degree of",
                                   14, False),
                 ("BACHELOR OF TECHNOLOGY", 16, True), ("in", 14, False),
                 ("COMPUTER SCIENCE ENGINEERING", 16, True),
                 ("with specialization in [SPECIALIZATION NAME]", 14, False), ("", 12, False),
                 ("DEPARTMENT OF [DEPARTMENT NAME]", 14, True),
                 ("COLLEGE OF ENGINEERING AND TECHNOLOGY", 14, True),
                 ("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", 14, True),
                 ("KATTANKULATHUR - 603 203", 14, True), ("NOVEMBER 2026", 14, True)]
        for text, size, bold in lines:
            p = d.add_paragraph(style="Center")
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(text)
            r.font.size, r.bold = Pt(size), bold

    def bonafide(self) -> None:
        d = self.doc
        p = d.add_paragraph(style="Center")
        p.paragraph_format.page_break_before = True
        if LOGO.exists():
            p.add_run().add_picture(str(LOGO), width=Cm(6))
        for text, size in (("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", 16),
                           ("KATTANKULATHUR - 603 203", 14), ("BONAFIDE CERTIFICATE", 16)):
            r = d.add_paragraph(style="Center").add_run(text)
            r.bold, r.font.size = True, Pt(size)
        body = d.add_paragraph(
            "Certified that 21CSE292P - ARTIFICIAL INTELLIGENCE OF THINGS project report titled "
            f"“{TITLE}” is the bonafide work of “[STUDENT 1 NAME] [REG. NO.], [STUDENT 2 NAME] "
            "[REG. NO.]” who carried out the project work under my supervision. Certified "
            "further, that to the best of my knowledge the work reported herein does not form "
            "any other project report or dissertation on the basis of which a degree or award "
            "was conferred on an earlier occasion on this or any other candidate.")
        body.paragraph_format.line_spacing = 2.0
        d.add_paragraph()
        t = d.add_table(rows=4, cols=2)
        for (left, right), row in zip([("SIGNATURE", "SIGNATURE"),
                                       ("[GUIDE NAME]", "Dr. M. LAKSHMI"),
                                       ("SUPERVISOR", "HEAD OF THE DEPARTMENT"),
                                       ("[Designation], [Department]",
                                        "Networking and Communications")], t.rows, strict=True):
            row.cells[0].paragraphs[0].add_run(left).bold = True
            row.cells[1].paragraphs[0].add_run(right).bold = True

    def declaration(self) -> None:
        d = self.doc
        self.front_heading("OWN WORK DECLARATION FORM")
        for label in ("Degree / Course: B.Tech, Computer Science Engineering",
                      "Student Name: [STUDENT 1 NAME], [STUDENT 2 NAME]",
                      "Registration Number: [REG. NO.], [REG. NO.]",
                      f"Title of Work: {TITLE}"):
            d.add_paragraph(label)
        d.add_paragraph("I / We hereby certify that this assessment complies with the "
                        "University's Rules and Regulations relating to Academic misconduct and "
                        "plagiarism, as listed in the University website, Regulations, and the "
                        "Education Committee guidelines. I / We confirm that all the work "
                        "contained in this assessment is my / our own except where indicated, "
                        "and that I / We have met the following conditions:")
        for item in ("Clearly referenced / listed all sources as appropriate",
                     "Referenced and put in inverted commas all quoted text (from books, web, "
                     "etc.)", "Given the sources of all pictures, data etc. that are not my own",
                     "Not made any use of the report(s) or essay(s) of any other student(s) "
                     "either past or present",
                     "Acknowledged in appropriate places any help that I have received from "
                     "others (e.g. fellow students, technicians, statisticians, external sources, "
                     "AI coding assistants)",
                     "Complied with any other plagiarism criteria specified in the Course "
                     "handbook / University website"):
            d.add_paragraph(item, style="List Bullet")
        d.add_paragraph("I understand that any false claim for this work will be penalized in "
                        "accordance with the University policies and regulations.")
        d.add_paragraph("DECLARATION: I am aware of and understand the University's policy on "
                        "Academic misconduct and plagiarism and I certify that this assessment "
                        "is my / our own work, except where indicated by referring, and that I "
                        "have followed the good academic practices noted above.")
        d.add_paragraph("[Student 1 signature, registration number, date]")
        d.add_paragraph("[Student 2 signature, registration number, date]")

    def acknowledgement(self) -> None:
        self.front_heading("ACKNOWLEDGEMENT")
        for text in (
            "We express our sincere gratitude to [GUIDE NAME], our project guide, for continuous "
            "guidance, careful reviews and encouragement throughout this project.",
            "We thank Dr. M. Lakshmi, Head of the Department, and the review panel members for "
            "their valuable feedback during the project reviews, and the SRM Institute of "
            "Science and Technology for the facilities provided.",
            "We thank the authors of the public datasets used in this work, whose openness made "
            "an honest evaluation possible. AI coding assistants were used under our direction "
            "to write code and text; every such contribution is logged in the project's AI log, "
            "and all results were checked by us.",
            "Finally, we thank our families and friends for their support.",
        ):
            self.doc.add_paragraph(text)
        self.doc.add_paragraph("[STUDENT 1 NAME]        [STUDENT 2 NAME]")

    def abstract(self) -> None:
        self.front_heading("ABSTRACT")
        p = self.doc.add_paragraph(
            "Coaches cannot easily see an athlete's injury risk building up over days, and "
            "published artificial intelligence of things (AIoT) frameworks for athlete health "
            "usually stop at the architecture. This project builds and evaluates a complete "
            "pipeline: generators of synthetic wearable data, edge, fog and cloud processing "
            "tiers, a fused sequence-plus-tabular injury-risk model, explanations tested for "
            "faithfulness, and a coach dashboard. Because no public dataset pairs body signals "
            "with real injuries, injury risk is learned only from real daily training logs of 74 "
            "competitive runners, while public wearable datasets drive the IoT tiers and the "
            "generators. An evaluation protocol written before any experiment splits all data "
            "by person, chooses every setting and threshold on validation runners only, checks "
            "for leakage, and kept ten runners sealed until the code was frozen. Learned models "
            f"predict injuries only modestly above chance (ROC-AUC {self.numbers['injuryAucRange']}"
            " over five development rounds), and the declared fused model kept this level on the "
            "sealed runners, showing that the result is not inflated. The generators did not "
            "produce realistic data and synthetic injury examples gave no gain. Processing at "
            f"the edge reduced the data sent to the cloud by {self.numbers['tiersReduction']} for "
            "the wearable stream, and the explanations passed deletion, randomisation and "
            "stability tests in every round. The project shows that a carefully tested, modest "
            "result is more useful than an impressive but untrustworthy one.")
        for r in p.runs:
            r.font.size = Pt(14)

    def lists(self) -> None:
        for title, instr in (("TABLE OF CONTENTS", 'TOC \\o "1-3" \\h \\z \\t "FrontHeading,1"'),
                             ("LIST OF FIGURES", 'TOC \\h \\z \\t "FigCaption,1"'),
                             ("LIST OF TABLES", 'TOC \\h \\z \\t "TabCaption,1"')):
            self.front_heading(title)
            _field(self.doc.add_paragraph(), instr,
                   "[Right-click and choose Update Field to fill this list]")

    def abbreviations(self) -> None:
        self.front_heading("ABBREVIATIONS")
        rows = [["ACWR", "Acute-to-chronic workload ratio"],
                ["AIoT", "Artificial intelligence of things"],
                ["AUC", "Area under the curve"], ["CNN", "Convolutional neural network"],
                ["CI", "Continuous integration / confidence interval"],
                ["GAN", "Generative adversarial network"], ["GRU", "Gated recurrent unit"],
                ["HR", "Heart rate"], ["IG", "Integrated gradients"],
                ["IMU", "Inertial measurement unit"], ["IoT", "Internet of things"],
                ["LSTM", "Long short-term memory"],
                ["PR-AUC", "Area under the precision-recall curve"],
                ["ROC", "Receiver operating characteristic"],
                ["SHAP", "SHapley Additive exPlanations"],
                ["TCN", "Temporal convolutional network"],
                ["TSTR", "Train on synthetic, test on real"], ["VAE", "Variational autoencoder"],
                ["WGAN-GP", "Wasserstein GAN with gradient penalty"],
                ["XAI", "Explainable artificial intelligence"]]
        t = self.doc.add_table(rows=0, cols=2)
        for a, b in rows:
            cells = t.add_row().cells
            cells[0].paragraphs[0].add_run(a).bold = True
            cells[1].paragraphs[0].add_run(b)

    # --- chapters from report_content.md
    def chapters(self) -> None:
        blocks = re.split(r"\n\s*\n", CONTENT.read_text("utf-8").strip())
        for block in blocks:
            block = block.strip()
            if block.startswith("# CHAPTER"):
                num, title = re.match(r"# CHAPTER (\d+) \| (.+)", block).groups()
                self.chapter, self.fig, self.tab, self.eq = int(num), 0, 0, 0
                h = self.doc.add_heading(level=1)
                h.paragraph_format.page_break_before = True
                h.add_run(f"CHAPTER {num}")
                h.add_run().add_break(WD_BREAK.LINE)
                h.add_run(title)
            elif block.startswith("### "):
                self.doc.add_heading(block[4:], level=3)
            elif block.startswith("## "):
                self.doc.add_heading(block[3:], level=2)
            elif block.startswith("[[FIG:"):
                name, cap = re.match(r"\[\[FIG:([^|]+)\|(.+)\]\]", block).groups()
                self.figure(name, cap)
            elif block.startswith("[[TABLE:"):
                key, cap = re.match(r"\[\[TABLE:([^|]+)\|(.+)\]\]", block).groups()
                header, rows, widths = TABLES[key](self)
                self.table(header, rows, cap, widths)
            elif block.startswith("[[CODE:"):
                target, cap = re.match(r"\[\[CODE:([^|]+)\|(.+)\]\]", block).groups()
                self.code(target, cap)
            elif block.startswith("[[ALGO:"):
                title, steps = re.match(r"\[\[ALGO:([^|]+)\|(.+)\]\]", block, re.S).groups()
                self.algorithm(title, [x for x in steps.split(";;") if x.strip()])
            elif block.startswith("[[EQ:"):
                self.equation(block[5:-2])
            elif block.startswith("- "):
                for line in block.splitlines():
                    self.para(line[2:].strip(), "List Bullet")
            else:
                self.para(" ".join(block.splitlines()))

    def references(self) -> None:
        self.front_heading("REFERENCES")
        for i, key in enumerate(self.cited, 1):
            p = self.doc.add_paragraph(style="RefItem")
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.9)
            p.add_run(f"[{i}]\t")
            for text, bold in _ref_text(self.refs[key]):
                p.add_run(text).bold = bold
        missing = [k for k in self.refs if k not in self.cited]
        if missing:
            raise ValueError(f"references never cited in the report: {missing}")

    def appendices(self) -> None:
        for title, text in (
            ("APPENDIX A: CONFERENCE PUBLICATION",
             "[Attach the proof of paper communication / acceptance here.]"),
            ("APPENDIX B: JOURNAL PUBLICATION",
             "[Attach the journal submission or publication proof here, if any.]"),
            ("APPENDIX C: PLAGIARISM REPORT",
             "[Attach the overall and chapter-wise plagiarism report from the software here.]"),
        ):
            self.front_heading(title)
            self.doc.add_paragraph(text)

    def build(self) -> None:
        first = self.doc.sections[0]
        _margins(first)
        first.different_first_page_header_footer = True
        _page_numbers(first, "lowerRoman")
        _header_number(first)
        self.cover()
        self.bonafide()
        self.declaration()
        self.acknowledgement()
        self.abstract()
        self.lists()
        self.abbreviations()
        main = self.doc.add_section(WD_SECTION.NEW_PAGE)
        _margins(main)
        main.different_first_page_header_footer = False
        _page_numbers(main, "decimal")
        _header_number(main)
        self.chapters()
        self.references()
        self.appendices()
        OUT.parent.mkdir(exist_ok=True)
        self.doc.save(OUT)


# ------------------------------------------------------------------------------- tables
def _fmt(v: dict | None, key: str) -> str:
    if not v or not v.get(key):
        return "n/a"
    return f"{v[key]['mean']:.3f} ± {v[key]['sd']:.3f}"


def t_srs(_r):
    rows = [["Operating system", "Windows 11 (also runs on Linux in the online test service)"],
            ["Language", "Python 3.12 (code kept compatible with 3.11)"],
            ["Deep learning", "PyTorch with CUDA; CPU-only mode for every step except training"],
            ["Machine learning", "scikit-learn, CatBoost, XGBoost"],
            ["Explanations", "SHAP, Captum"],
            ["Dashboard and graphs", "Streamlit, Matplotlib"],
            ["Documents", "LaTeX (paper), python-pptx (slides), python-docx (this report)"],
            ["Version control and testing", "Git and GitHub with automatic lint and test runs; "
                                            "pytest; ruff"],
            ["Hardware used", "Laptop with an NVIDIA RTX 4050 graphics card (about 6 GB) and "
                              "16 GB memory"]]
    return ["Item", "Requirement"], rows, [4.5, 10.0]


def t_survey(_r):
    rows = [["[@oliver2020]", "Machine learning for injury risk in elite youth football",
             "Individual prediction stayed limited; no IoT processing"],
            ["[@hildebrandt2021]", "Training load and injury risk in youth ski racing",
             "Association study, not a deployed predictor"],
            ["[@tysoe2020]", "Load measures beyond the acute-to-chronic ratio",
             "No machine-learning model or explanation"],
            ["[@stern2020,@stern2021]", "Non-linear injury dynamics; prior injury as predictor",
             "Conceptual; highlights a leakage risk for models"],
            ["[@lovdal2021]", "Injury prediction for competitive runners from daily logs",
             "Scaling step reads labels; no IoT tiers"],
            ["[@caviedes2020,@abdalalim2024]", "Wearable monitoring and stress detection",
             "No injury outcomes"],
            ["[@wang2019,@ronao2016]", "Deep activity recognition from sensors",
             "Activity, not injury risk"],
            ["[@cao2021,@aazam2018]", "Edge, edge-cloud and fog computing",
             "General systems, not athlete risk"],
            ["[@gui2023,@guo2019,@liu2024]", "Generative models and time-series augmentation",
             "Realism, usefulness and privacy rarely tested together"],
            ["[@kapoor2023,@samek2017]", "Leakage; evaluation of explanations",
             "Methods we adopt for honest testing"]]
    return ["Work", "Focus", "Gap for this project"], rows, [3.6, 5.6, 5.3]


def t_datasets(_r):
    return ["Dataset", "Origin", "Role"], [list(r) for r in DATASETS], [4.0, 2.0, 8.5]


def t_modules(_r):
    rows = [["src/data", "Read raw files, fingerprints, processed arrays, person-level splits, "
                         "sealed-test guard"],
            ["src/eval", "Metrics (PR-AUC, ROC-AUC, calibration, alerts) and runner bootstrap"],
            ["src/models", "Baselines, neural networks, tuning, fused model, final test"],
            ["src/gen", "TimeGAN, WGAN-GP, TimeVAE, classic augmentation and their checks"],
            ["src/pipeline", "Edge, fog and cloud processes and their benchmark"],
            ["src/xai", "SHAP, integrated gradients, attention, Grad-CAM and their tests"],
            ["src/dashboard", "Streamlit coach dashboard and data preparation"],
            ["src/paper", "Numbers, tables, graphs, slides and this report from result files"]]
    return ["Folder", "Responsibility"], rows, [3.2, 11.3]


def t_hyper(_r):
    rows = [["Tuning", "8 settings per model per round, seeded random search; chosen on "
                       "validation; refit with seeds 0, 1, 2"],
            ["Logistic regression", "L2; C from 1e-4 to 1e2; class weight none or balanced"],
            ["Random forest", "500 trees; depth 4-12 or unlimited; min. leaf 1-50"],
            ["CatBoost", "up to 2,000 iterations, early stop 100; learning rate 0.01-0.1; "
                         "depth 4/6/8"],
            ["GRU / LSTM", "batch 128; up to 60 epochs, patience 10; Adam; hidden 16/32/64; "
                           "1-2 layers; dropout 0-0.4; learning rate 3e-4 to 3e-3"],
            ["TCN", "batch 128; up to 60 epochs; channels 16/32/64; kernel 2/3; levels 2/3"],
            ["Fused combiner", "logistic regression, C from 0.01/0.1/1/10; 5-fold out-of-fold "
                               "by runner; 44 tries per round"],
            ["Activity CNN", "3 conv blocks; batch 128; up to 40 epochs, patience 8; "
                             "channels 32/64; kernel 5/9"],
            ["TimeGAN", "batch 128; 1,000 + 1,000 + 1,500 steps; hidden 24/48/96"],
            ["WGAN-GP", "one declared setting (untuned): channels 64, noise 64, 5 critic steps, "
                        "learning rate 2e-4, 3,000 steps"],
            ["TimeVAE", "batch 128; 3,000 steps; latent 8/16/32"]]
    return ["Component", "Settings"], rows, [3.4, 11.1]


def t_commands(_r):
    rows = [["python tasks.py check", "Lint and run all automated tests"],
            ["python tasks.py baselines", "Train and evaluate the baseline models"],
            ["python tasks.py gen", "Train and evaluate the generators"],
            ["python tasks.py train", "Train the fused model over five rounds"],
            ["python tasks.py pipeline", "Run the edge-fog-cloud benchmark"],
            ["python tasks.py explain", "Explain the fused model and test the explanations"],
            ["python tasks.py demo", "Start the coach dashboard"],
            ["python tasks.py paper-figures --deck", "Rebuild paper numbers, tables and slides"],
            ["python -m src.paper.plots", "Redraw all result graphs"],
            ["python -m src.paper.report_docx", "Rebuild this report"]]
    return ["Command", "Purpose"], rows, [6.0, 8.5]


def t_tests(_r):
    rows = [["Data", "test_audit, test_manifest, test_loaders, test_windows, test_daily, "
                     "test_reference, test_splits", "File fingerprints, loaders, windows, splits"],
            ["Final test", "test_lockbox, test_final", "Sealed-test guard, freeze record, "
                                                       "final-test summary"],
            ["Models", "test_models, handoff tests", "Metrics, bootstrap, networks, tuning, "
                                                    "fused model"],
            ["Generators and tiers", "test_gen, test_pipeline", "Generator shapes and checks; "
                                                               "tier messages and latency"],
            ["Explanations", "test_xai", "SHAP, integrated gradients, deletion, randomisation, "
                                         "stability"],
            ["Dashboard and tools", "test_dashboard, test_dashboard_pages, test_repo_tools",
             "Every page renders; repository hooks"]]
    return ["Area", "Test files", "What is checked"], rows, [3.0, 5.5, 6.0]


def t_injury(_r):
    b = figures.baselines()
    names = {"constant_risk": "Constant risk", "acwr_style": "ACWR-style rule",
             "logistic_regression": "Logistic regression", "random_forest": "Random forest",
             "catboost": "CatBoost", "gru": "GRU", "lovdal_label_free": "Lövdal, label-free"}
    rows = [[lab, k, _fmt(b[(m, k)], "pr_auc"), _fmt(b[(m, k)], "roc_auc")]
            for m, lab in names.items() for k in ("day", "week") if (m, k) in b]
    fused = figures._runs("fused")
    if fused:
        s = fused[-1][1]["fused"]["summary"]
        rows.append(["Fused (declared in advance)", "shared", _fmt(s, "pr_auc"),
                     _fmt(s, "roc_auc")])
    return ["Model", "File", "PR-AUC", "ROC-AUC"], rows, [5.5, 2.0, 3.5, 3.5]


def t_lockbox(_r):
    m = figures._runs("lockbox")[-1][1]
    f = m["fused"]

    def ci(c):
        return f"{c['estimate']:.3f} ({c['low']:.3f}-{c['high']:.3f})"
    rows = [["Fused model (primary)", "shared days", f"{f['events']:,} / {f['injuries']}",
             ci(f["roc_auc_ci"]), ci(f["pr_auc_ci"])]]
    for arm, v in m["arms_on_own_files"].items():
        rows.append([f"{arm.capitalize()} arm (secondary)", f"{v['file']} file",
                     f"{v['events']:,} / {v['injuries']}", ci(v["roc_auc_ci"]),
                     ci(v["pr_auc_ci"])])
    for kind, v in m["lovdal_label_free_paper_split"].items():
        rows.append(["Lövdal label-free (secondary)", f"{kind} file",
                     f"{v['events']:,} / {v['injuries']}", ci(v["roc_auc_ci"]),
                     ci(v["pr_auc_ci"])])
    return ["Model", "Rows", "Events / injuries", "ROC-AUC (95% CI)", "PR-AUC (95% CI)"], rows, \
        [4.2, 2.3, 2.6, 2.8, 2.6]


def t_comparisons(_r):
    comps = figures._runs("comparisons")[-1][1]["comparisons"]
    rows = []
    for c in comps:
        p, r = c["pr_auc"], c["roc_auc"]
        name = c["comparison"].replace(" - lovdal_label_free", " vs label-free")
        rows.append([name.replace("_", " ").replace(" - ", " vs "),
                     f"{p['estimate']:+.4f} [{p['low']:+.4f}, {p['high']:+.4f}]",
                     f"{r['estimate']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]"])
    return ["Comparison", "Δ PR-AUC [95% CI]", "Δ ROC-AUC [95% CI]"], rows, [6.5, 4.0, 4.0]


def t_generators(_r):
    g = figures._runs("generators", "metrics_pamap2.json")[-1][1]
    label = {"timegan": "TimeGAN", "wgan_gp": "WGAN-GP (untuned)", "timevae": "TimeVAE",
             "classic": "Classic augmentation"}
    rows = [[label.get(k, k), f"{m['discriminative_accuracy']['mean']:.3f}",
             f"{m['tstr_test_macro_f1']['mean']:.3f}",
             f"{m['real_plus_synthetic_test_macro_f1']['mean']:.3f}",
             f"{sum(g['criteria'][k].values())}/{len(g['criteria'][k])}"]
            for k, m in g["pamap2_summary"].items()]
    return ["Generator", "Real-vs-fake acc.", "TSTR F1", "Real+syn. F1", "Criteria met"], rows, \
        [4.6, 2.6, 2.3, 2.6, 2.4]


def t_tiers(_r):
    st = figures._runs("tiers")[-1][1]["streams"]
    names = {"pamap2": "PAMAP2 wearable", "sportdb2_hr": "Sport DB heart rate"}
    rows = [[names.get(s, s), mode.replace("_", "-"), f"{o[mode]['bytes_into_cloud']:,}",
             f"{o[mode]['decision_latency_ms_median']:.2f}"]
            for s, o in st.items() for mode in ("tiered", "cloud_only")]
    return ["Stream", "Design", "Bytes into cloud", "Median decision (ms)"], rows, \
        [4.5, 3.0, 3.5, 3.5]


TABLES = {"srs": t_srs, "survey": t_survey, "datasets": t_datasets, "modules": t_modules,
          "hyper": t_hyper, "commands": t_commands, "tests": t_tests, "injury": t_injury,
          "lockbox": t_lockbox, "comparisons": t_comparisons, "generators": t_generators,
          "tiers": t_tiers}


def _logo() -> None:
    """The SRM logo, taken once from the official template in the project root."""
    if LOGO.exists() or not TEMPLATE.exists():
        return
    with zipfile.ZipFile(TEMPLATE) as z:
        LOGO.write_bytes(z.read("word/media/image1.jpeg"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _logo()
    r = Report()
    r.build()
    print(f"{OUT} ({len(r.cited)} references cited)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
