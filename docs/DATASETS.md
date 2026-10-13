# DATASETS: download list and audit

Step 1 of the project: download all six, then audit each one (fill the audit table from code output, not by hand).
Put each dataset in its own folder under `data/raw/` (git-ignored). Record each download in `data/MANIFEST.csv`.

## Download list
| # | Dataset | Where to get it | Folder | Notes |
|---|---|---|---|---|
| 1 | Lövdal et al. runner logs (Replication Data for: Injury Prediction In Competitive Runners With Machine Learning) | DataverseNL, DOI [10.34894/UWU9PV](https://doi.org/10.34894/UWU9PV) | `data/raw/lovdal/` | CC0. 74 runners, 2012-2019, daily (7-day x 10 features) and weekly (22 features) files. |
| 2 | Injury Prediction for Competitive Runners (Kaggle, shashwatwork) | kaggle.com/datasets/shashwatwork/injury-prediction-for-competitive-runners (needs Kaggle login) | `data/raw/kaggle_runners/` | Reported to be a re-upload of #1. Audit: compare row counts and values with #1. |
| 3 | PAMAP2 Physical Activity Monitoring | UCI ML Repository, dataset 231: archive.ics.uci.edu/dataset/231 | `data/raw/pamap2/` | 9 subjects, 18 activities, 3 IMUs + heart rate at 100 Hz. |
| 4 | WESAD (Wearable Stress and Affect Detection) | UCI ML Repository, dataset 465: archive.ics.uci.edu/dataset/465 (DOI 10.24432/C57K5T) | `data/raw/wesad/` | 15 subjects, chest + wrist, baseline / stress / amusement. Large (about 2 GB). Check licence terms on the UCI page. |
| 5 | Collective Sports Sensor DB = Sport DB 2.0 | Mendeley Data: data.mendeley.com/datasets/kzkjkt7mx2/2 | `data/raw/sportdb2/` | 168 cardiorespiratory recordings, 130 athletes, 11 sports (ECG, HR, breathing rate). No injury labels. |
| 6 | SIRP-600 (Sports Injury Risk Prediction) | [VERIFY] source not found by web search; Naman to provide the exact Kaggle link used for Review 1 | `data/raw/sirp600/` | Possibly a simulated Kaggle dataset; label origin in the audit. |

## Audit table (fill from scripts in src/data/audit.py)
| # | Real / simulated / unknown | Subjects | Rows or hours | Channels / features | Labels | Injury labels? | Time series? | Licence | Role decided |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 3 | | | | | | | | | |
| 4 | | | | | | | | | |
| 5 | | | | | | | | | |
| 6 | | | | | | | | | |

## Rules
- Never merge channels from different datasets into one person (different people, devices, protocols).
- Unknown-origin data: demo or sanity check only, never evidence.
- One data card per dataset in `docs/data_cards/<name>.md`: source, licence, counts, missingness, imbalance, known issues.
