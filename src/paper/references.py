"""python -m src.paper.references: build paper/refs_final.bib from the official DOI records.

Faculty rule (2026-11-13): at least 20 references, only from ScienceDirect journals and IEEE
Transactions (no conference papers). Every entry below is fetched from doi.org (the DOI registry),
so authors, title, journal, volume and pages are exactly the publisher's. The four dataset sources
are kept as extra references, labelled as data sources (D-035). Also writes
docs/REFERENCES_VERIFIED.md (one row per reference, with its venue type and the date checked).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import date

from src.data import loaders

# key -> (DOI, venue type)
JOURNAL = {
    "kapoor2023": ("10.1016/j.patter.2023.100804", "ScienceDirect (Patterns)"),
    "vancalster2016": ("10.1016/j.jclinepi.2015.12.005", "ScienceDirect"),
    "arrieta2020": ("10.1016/j.inffus.2019.12.012", "ScienceDirect"),
    "greff2017": ("10.1109/TNNLS.2016.2582924", "IEEE Transactions"),
    "he2009": ("10.1109/TKDE.2008.239", "IEEE Transactions"),
    "fawcett2006": ("10.1016/j.patrec.2005.10.010", "ScienceDirect"),
    "shwartzziv2022": ("10.1016/j.inffus.2021.11.011", "ScienceDirect"),
    "samek2017": ("10.1109/TNNLS.2016.2599820", "IEEE Transactions"),
    "montavon2018": ("10.1016/j.dsp.2017.10.011", "ScienceDirect"),
    "wang2019": ("10.1016/j.patrec.2018.02.010", "ScienceDirect"),
    "ronao2016": ("10.1016/j.eswa.2016.04.032", "ScienceDirect"),
    "wong2015": ("10.1016/j.patcog.2015.03.009", "ScienceDirect"),
    "oliver2020": ("10.1016/j.jsams.2020.04.021", "ScienceDirect"),
    "hildebrandt2021": ("10.1016/j.jshs.2020.03.009", "ScienceDirect"),
    "stern2020": ("10.1016/j.ptsp.2019.10.010", "ScienceDirect"),
    "stern2021": ("10.1016/j.ptsp.2021.01.008", "ScienceDirect"),
    "tysoe2020": ("10.1016/j.jsams.2020.01.004", "ScienceDirect"),
    "gui2023": ("10.1109/TKDE.2021.3130191", "IEEE Transactions"),
    "guo2019": ("10.1016/j.neucom.2019.06.007", "ScienceDirect"),
    "cao2021": ("10.1109/TII.2021.3073066", "IEEE Transactions"),
    "aazam2018": ("10.1109/TII.2018.2855198", "IEEE Transactions"),
    "caviedes2020": ("10.1109/TBME.2020.2971907", "IEEE Transactions"),
    "abdalalim2024": ("10.1016/j.compbiomed.2024.108918", "ScienceDirect"),
    "liu2024": ("10.1016/j.eswa.2024.124793", "ScienceDirect"),
    "an2025": ("10.1016/j.jshs.2025.101054", "ScienceDirect"),
}
DATASETS = {
    "lovdal2021": ("10.1123/ijspp.2020-0518", "data source (journal article)"),
    "lovdal2021data": ("10.34894/UWU9PV", "data source (dataset record)"),
    "reiss2012": ("10.1109/ISWC.2012.13", "data source (PAMAP2 dataset paper)"),
    "schmidt2018": ("10.1145/3242969.3242985", "data source (WESAD dataset paper)"),
}
BIB = loaders.ROOT / "paper" / "refs_final.bib"
TABLE = loaders.ROOT / "docs" / "REFERENCES_VERIFIED.md"
JSONF = loaders.ROOT / "paper" / "refs_final.json"  # the same records, for the Word report


def _fetch(doi: str, accept: str) -> str:
    req = urllib.request.Request("https://doi.org/" + doi,
                                 headers={"Accept": accept, "User-Agent": "SynAthlete-refs/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _bibtex(key: str, doi: str) -> str:
    """The registry's BibTeX with our key; month/url fields dropped to keep the list tidy."""
    raw = _fetch(doi, "application/x-bibtex").strip()
    raw = re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1).lower()}{{{key},", raw, count=1)
    raw = re.sub(r",\s*(month|url)\s*=\s*\{?[^,}]*\}?", "", raw)
    # keep the publisher's capitalisation (styles lower-case titles: "LSTM" would become "Lstm")
    raw = re.sub(r"title=\{(.*?)\}, volume|title=\{(.*?)\}, (?=\w+=)",
                 lambda m: f"title={{{{{m.group(1) or m.group(2)}}}}}, "
                 + ("volume" if m.group(1) else ""), raw, count=1)
    return raw


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    today = date.today().isoformat()
    entries, rows, records = [], [], {}
    for group, refs in (("journal", JOURNAL), ("dataset", DATASETS)):
        for key, (doi, venue) in refs.items():
            entries.append(_bibtex(key, doi))
            meta = json.loads(_fetch(doi, "application/vnd.citationstyles.csl+json"))
            records[key] = {k: meta.get(k) for k in ("title", "author", "container-title",
                                                      "publisher", "volume", "issue", "page",
                                                      "article-number", "issued", "DOI",
                                                      "type")}
            title = meta.get("title", "")
            where = meta.get("container-title") or meta.get("publisher", "")
            year = (meta.get("issued", {}).get("date-parts") or [[""]])[0][0]
            rows.append(f"| {key} | {title} | {where} | {year} | {venue} | "
                        f"https://doi.org/{doi} | {today} |")
            print(f"{group}: {key} ok")
    BIB.write_text("% Generated by src/paper/references.py from doi.org records: do not edit.\n"
                   f"% {len(JOURNAL)} ScienceDirect / IEEE Transactions journal articles plus "
                   f"{len(DATASETS)} dataset sources.\n\n" + "\n\n".join(entries) + "\n", "utf-8")
    TABLE.write_text(
        "# Verified references (generated by `python -m src.paper.references`)\n\n"
        f"{len(JOURNAL)} journal articles from ScienceDirect or IEEE Transactions (faculty rule: "
        f"at least 20), plus {len(DATASETS)} dataset sources kept for honest credit (D-035). Each "
        "entry was fetched from its DOI record on the date shown.\n\n"
        "| Key | Title | Journal / source | Year | Venue type | DOI | Checked |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n", "utf-8")
    JSONF.write_text(json.dumps(records, indent=1, ensure_ascii=False), "utf-8")
    print(f"wrote {BIB}, {JSONF.name} and {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
