# DATA CONTRACT (freeze in Phase 0; changes need a DECISIONS entry)
Draft defaults; confirm after the dataset audit (docs/DATASETS.md). Sport DB 2.0 and SIRP-600 formats are added once audited.

## Physiology windows (Track B)
- Sampling: resample to 25 Hz. Window: 128 steps (~5.1 s). Stride: 64.
- Array: float32, shape (N, 128, C). Channel order fixed per dataset in configs/channels_<dataset>.yaml.
- PAMAP2 start set: heart rate + accelerometers (about 10 channels). WESAD start set: wrist BVP, EDA, temperature, accelerometer.
- Labels: int codes (activity id; baseline/stress/amusement). Metadata: subject_id, window_start, dataset.
- Do not concatenate channels across datasets (different people, devices and protocols).

## Workload windows (Track A, Lövdal runner logs)
- Use the dataset authors' day-approach (7-day sequence, features per day) and week-approach definitions and labels.
- Arrays (D-019): day X float32 (N, 7, 10) + rest_day mask (N, 7); week X float32 (N, 3, 22) + extra (N, 3) = the 3 relative mileage changes. Axis 1 runs oldest -> newest (last step = just before the event day).
- Exact column order (authors' README v2): day file suffix .6 = the day before the event and no suffix = 7 days before (order kept); week file no suffix = the week before and .2 = three weeks before (order reversed). Verified on the data and pinned by tests.
- -0.01 in the three perceived columns = rest day: the value is kept and rest_day marks it.
- Metadata: athlete_id, date (day index from the first record). Label: injury (1) / no injury (0) as defined by the authors; document any deviation.
- Development runners 0-63: data/processed/lovdal/. Locked runners 64-73: data/processed/lovdal_lockbox/, read only through src/data/lockbox.py (D-017: annotated, pushed eval-freeze tag; one frozen commit only; access log). Use loaders.load_dev() for all development work; it also refuses data from another raw-data version than the splits. data/processed/lovdal/facts.json holds the whole-dataset counts quoted in the docs.

## Daily tables (Track C, SoccerMon and PMData; exploratory, D-018)
- One row per person per calendar day (UTC dates), saved as data/processed/<dataset>/daily.parquet. Missing days stay empty: no imputation.
- Labels per day: pain_report, episode_start (positive event), healthy_event (21-day rule, D-018). PMData p08 has no labels (missing injury file).

## Splits
- data/splits/<dataset>_folds.json: {fold: {train_ids, val_ids, test_ids}} by person. Created once (D-017).
- Lövdal also has lockbox_ids, final, paper_split and balance; Sport DB has test_ids + test_recordings only; leave-one-person-out files (<dataset>_lopo.json) have train_ids + test_ids; SIRP-600 has train_rows + test_rows (demo only).
- Every file's meta records seed, method, library version, git commit, data fingerprint and decision.
- data/splits/SHA256SUMS pins every file; data/splits/superseded/ keeps the first PAMAP2/WESAD versions (D-020). Nothing in data/splits/ changes after sealing.
- Normalisation statistics computed on training folds only and stored with each run: one scaler per round, no per-runner statistics (a runner's own event history reveals the label in the Lövdal event design; D-023).

## Data versions
- Every builder hashes the raw files it reads, stops if they differ from (or are missing in) data/MANIFEST.csv, and records {files_read, fingerprint} as "source" in its metadata (D-019).

## Synthetic data
- Stored as data/processed/synthetic/<generator>/<run_id>/ with a datasheet: generator, seed, training fold, conditioning, fidelity report path.

## Pipeline interfaces
- Message: {athlete_id, ts, window(shape per above), source: real|replay|synthetic}. Alert: {athlete_id, ts, tier, risk, explanation_ref}.
