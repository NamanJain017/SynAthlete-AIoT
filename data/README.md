# data/

- `raw/`: original downloads, one folder per dataset (see docs/DATASETS.md). Git-ignored. Never edited.
- `processed/`: arrays and tables produced by code from `raw/`. Git-ignored; rebuild with `python tasks.py data`.
- `splits/`: person-level fold assignments (JSON). Committed. Created once by code, never regenerated.
- `MANIFEST.csv`: one row per downloaded file: dataset, file, source URL, licence, access date, SHA-256. Filled by code.
