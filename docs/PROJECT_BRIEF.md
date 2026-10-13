# PROJECT BRIEF: SynAthlete-AIoT

**Title:** SynAthlete-AIoT: An Explainable AIoT Framework for Athlete Injury-Risk Prediction Using GAN-Synthesized Wearable Data
**Course:** 21CSE292P Artificial Intelligence of Things (PBL-I), SRM IST. Lead builder: Naman Jain.
**Status:** Review 1 done (19.08.2026). Review 2 expected in late October 2026 (exact date to be confirmed with the guide).

## 1. The problem in one paragraph
Coaches cannot see injury risk building up over days. Recent AIoT athlete-health frameworks (Appl. Sci. 16(9):4542, 2026) design a wearable + edge-fog-cloud system but leave anomaly detection, injury-risk prediction and explainability unimplemented. Most injury-risk ML uses one-time snapshots, not day-by-day data. And no public dataset pairs high-frequency physiological streams (heart rate, motion) with real injury outcomes.

## 2. What Review 1 showed (keep all of it; see slides/PBL_1_AIOT.pdf if added)
| Slide | Content |
|---|---|
| Problem | 3 gaps: design without implementation; snapshots not timelines; black-box bottleneck. Core gap sentence (needs correction, see section 3). |
| Literature | MDPI Appl. Sci. May 2026 AIoT framework; BMC Sports Sci Jan 2026 SHAP + RF; TimeGAN / DoppelGANger lineage; edge-AI health surveys. |
| Objectives | 1 bridge the data gap with TimeGAN / WGAN-GP; 2 simulated edge-fog-cloud tiers; 3 fused model (LSTM/GRU/TCN + Random Forest); 4 SHAP + attention explanations validated against known risk factors; 5 coach dashboard. |
| Datasets | Physiology (no injury labels): PAMAP2, WESAD. Injury outcome (no time series): SIRP-600, Injury Prediction (Runners), Collective Sports Sensor DB. |
| Pipeline | Data foundation -> GAN synthesis -> edge-fog-cloud (simulated) -> fused modelling -> explainability -> dashboard. |
| Architecture | Pile A sensor data + Pile B injury records -> GAN -> synthetic combined dataset -> reflex layer (quick check) -> supervisor layer (session) -> deep layer (long-term) -> time-series model + snapshot model -> fusion -> risk score 0-100 -> explainability -> coach dashboard. |
| Novelty | 1 implements the missing AI layer; 2 snapshot -> timeline; 3 bridges fragmented data; 4 explainability validated, not assumed. |

## 3. Correction since Review 1
Lövdal, den Hartigh and Azzopardi (2021) published 7 years of daily training logs from 74 competitive runners **with real injuries** (DataverseNL, DOI 10.34894/UWU9PV, CC0). So "no dataset joins continuous vitals with real injury outcomes" is too strong. Corrected gap sentence:
> No public dataset pairs **high-frequency physiological streams** with real injury outcomes.
The Kaggle "Injury Prediction for Competitive Runners" set shown in Review 1 is reported to be a re-upload of this same data; confirm in the dataset audit.

**Second check needed (open):** SoccerMon (Midoglu et al., Scientific Data 2024) pairs GPS + heart-rate wearables with self-reported injuries from two elite women's soccer teams, and SoccerGuard (arXiv 2411.08901) already runs injury-prediction ML on it. So even the corrected sentence may be too strong. Safer wording until the audit is done:
> Few public datasets pair high-frequency physiological streams with real injury outcomes; the closest (SoccerMon) relies on sparse, self-reported injuries from two teams.
Also: SIRP-600 (and other Kaggle sets) are likely synthetic, so the Review 1 dataset slide title "Two real, verified halves" must change in Review 2.

## 4. How every Review 1 item is included (as part of the bigger system)
| Review 1 item | Role in the build |
|---|---|
| PAMAP2 | Real wearable stream for the reflex/edge tier; generator training and fidelity tests. |
| WESAD | Second wearable dataset: generator generalisation + stress signal; role fixed after the audit. |
| Injury Prediction (Runners) / Lövdal | Real injury labels with daily time series: the evidence for injury-risk claims. |
| SIRP-600 | Snapshot-style injury risk factors for the tabular ("snapshot") arm; role and evidence status fixed after the audit (origin must be checked). |
| Collective Sports Sensor DB (Sport DB 2.0) | Real athlete cardiorespiratory sessions: extra edge-tier stream and a realism check for synthetic physiology; role fixed after the audit. |
| TimeGAN, WGAN-GP | Generator comparison, together with TimeVAE and classic augmentation as baselines. |
| Reflex / supervisor / deep layers | Edge / fog / cloud tiers as separate processes, with measured latency and bytes sent. |
| LSTM / GRU / TCN + Random Forest | Sequence models + tabular models, fused by stacking into a calibrated 0-100 risk score. |
| SHAP + attention (+ CAM) | An explanation comparison: TreeSHAP (tabular), integrated gradients (sequence), attention and 1D Grad-CAM as extra views; all checked with deletion tests and against known risk factors. |
| Coach dashboard | Streamlit: athlete list, athlete detail with plain-language reasons, system view, alert log. |

## 5. Review 2 rubric (20 marks)
| Part | Marks | What we show |
|---|---|---|
| Methodology | 5 | Final pipeline diagram, dataset audit, data roles, evaluation rules. |
| Implementation | 10 | Working code with real results: data pipeline, baselines, generator, tiers, explanations, dashboard demo. |
| Paper draft + communication to a conference | 5 | IEEE-style conference draft; conference chosen with the guide. |

## 6. Honesty rules that protect the viva
- Test only on people the model never saw.
- Injury claims only from real injury labels.
- Synthetic data is labelled as synthetic everywhere.
- Every number comes from a file written by code.
