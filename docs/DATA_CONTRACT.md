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
- Array: float32, shape (N, 7, F). Metadata: runner_id, window_end_date.
- Label: injury (1) / no injury (0) as defined by the authors; document any deviation.

## Splits
- data/splits/<dataset>_folds.json: {fold: {train_ids, val_ids, test_ids}} by person. Created once.
- Normalisation statistics computed on training folds only and stored with each run.

## Synthetic data
- Stored as data/processed/synthetic/<generator>/<run_id>/ with a datasheet: generator, seed, training fold, conditioning, fidelity report path.

## Pipeline interfaces
- Message: {athlete_id, ts, window(shape per above), source: real|replay|synthetic}. Alert: {athlete_id, ts, tier, risk, explanation_ref}.
