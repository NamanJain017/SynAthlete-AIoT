# DATASETS: download list and audit

Step 1 of the project: download, then audit each dataset with code (fill the audit table from code output, not by hand).
Put each dataset in its own folder under `data/raw/` (git-ignored). Record each download in `data/MANIFEST.csv`.

## Download list
| # | Dataset | Download from | Folder | What we know (verify in the audit) |
|---|---|---|---|---|
| 1 | Lövdal et al. runner logs (Replication Data for: Injury Prediction In Competitive Runners With Machine Learning) | DataverseNL: https://doi.org/10.34894/UWU9PV (original source; server returned HTTP 500 on 2026-10-13, retry later. Until then use #2 and record the source in MANIFEST.csv) | `data/raw/lovdal/` | Real. 74 runners, 2012-2019, real injuries. Day approach (7 days x 10 features) and week approach (22 features). Paper: Int J Sports Physiol Perform 16(10):1522-1531, doi 10.1123/ijspp.2020-0518. |
| 2 | Injury Prediction for Competitive Runners (Kaggle, shashwatwork) | https://www.kaggle.com/datasets/shashwatwork/injury-prediction-for-competitive-runners | `data/raw/kaggle_runners/` | Reported mirror of #1 (shown in Review 1). Small; downloaded only so the audit can prove it is the same data. |
| 3 | PAMAP2 Physical Activity Monitoring | UCI: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | `data/raw/pamap2/` | Real. 9 subjects, 18 activities, 3 IMUs + heart rate. IMUs 100 Hz; heart rate only about 9 Hz (resampling interpolates it). |
| 4 | WESAD (Wearable Stress and Affect Detection) | UCI: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection (same file as https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) | `data/raw/wesad/` | Real. 15 subjects, chest + wrist. Labels: neutral/baseline, stress, amusement. About 2.5 GB. Academic, non-commercial use; follow Siegen acknowledgement policy. |
| 5 | Sport DB 2.0 (slide name: "Collective Sports Sensor DB") | Mendeley Data: https://data.mendeley.com/datasets/kzkjkt7mx2/2 (doi 10.17632/kzkjkt7mx2.2) | `data/raw/sportdb2/` | Real. 168 cardiorespiratory recordings, 130 athletes, 11 sports (ECG, HR, breathing rate). No injury labels. |
| 6 | SIRP-600 (Sports Injury Risk Prediction) | Kaggle: https://www.kaggle.com/datasets/yuanchunhong/sirp-600-sports-injury-risk-prediction-dataset | `data/raw/sirp600/` | Shown in Review 1. Origin undocumented; likely synthetic. Demo / sanity check only unless the audit shows otherwise. |
| 7 | Wearable Sports Health Monitoring (Kaggle, ziya07) | https://www.kaggle.com/datasets/ziya07/wearable-sports-health-monitoring-dataset | `data/raw/ziya07_wearable/` | Optional. Reported simulated (about 500 records). Demo only. |
| 8 | SoccerMon (Midoglu et al., Scientific Data 11:553, 2024, doi 10.1038/s41597-024-03386-x) | https://datasets.simula.no/soccermon/ | `data/raw/soccermon/` | Real. 2 elite women's teams, 2020-2021: self-reported wellness, training load and injuries (PmSys) + GPS/heart-rate wearables (STATSports). Download page reported unavailable by Naman on 2026-10-13; not downloaded. Still cited in related work (paper is published). |

Not a download: b2find.eudat.eu only mirrors the metadata of #1.

## Audit table (fill from scripts in src/data/audit.py)
| # | Real / simulated / unknown | Subjects | Rows or hours | Channels / features | Labels | Injury labels? | Time series? | Licence | Role decided |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| 3 | | | | | | | | | |
| 4 | | | | | | | | | |
| 5 | | | | | | | | | |
| 6 | | | | | | | | | |
| 7 | | | | | | | | | |
| 8 | | | | | | | | | |

## Rules
- Never merge channels from different datasets into one person (different people, devices, protocols).
- Simulated or unknown-origin data: demo or sanity check only, never evidence.
- One data card per dataset in `docs/data_cards/<name>.md`: source, licence, counts, missingness, imbalance, known issues.
- Published baseline to compare against for #1: bagged XGBoost ROC AUC 0.724 (day) and 0.678 (week), from the paper above [VERIFY numbers against the paper PDF].
