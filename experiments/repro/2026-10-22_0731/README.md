# Fresh-clone reproducibility check (2026-10-22)

What it does (`repro.sh`): clones the repository at commit 28dba97, copies only the 729 raw files
listed in data/MANIFEST.csv (`copy_raw.py`), runs the full audit, all data builders and all tests in
the clone, then compares the clone with the main working copy (`compare_repro.py`): MANIFEST rows,
audit facts and fingerprints, every processed array/metadata/table, the data cards, and all 7 split
files rebuilt in memory from seed 0 (never written).

Result (`log.txt`): ALL IDENTICAL; 103 tests passed in the clone; 144 s in total.
Paths in repro.sh point to this laptop's temporary folder; change SP/MAIN to rerun elsewhere.
