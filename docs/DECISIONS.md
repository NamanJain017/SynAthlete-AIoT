# DECISIONS (append only; one entry per decision)
Format: ID | date | decision | reason | alternatives rejected | who

- D-001 | 2026-10-06 | Real injury evidence comes only from real injury labels (Lövdal runner logs); wearable datasets support generators, fidelity and the IoT stream | real injury labels exist only at daily-load level; synthetic coupling of physiology and injuries cannot be validated | claiming one merged real+synthetic dataset as evidence | team
- D-002 | 2026-10-06 | Split by person; scalers, resamplers and generators fitted on training folds only | prevents leakage, the main failure mode in sports ML | window-level random split | team
- D-003 | 2026-10-06 | Compare GANs with a non-GAN generator (TimeVAE) and classic augmentation | a GAN-only comparison is weak evidence | GAN only | team
- D-004 | 2026-10-06 | Pre-register the evaluation protocol before generator runs | prevents post-hoc metric selection | choose metrics after seeing results | team
- D-005 | 2026-10-06 | Correct the Review 1 gap sentence to "No public dataset pairs high-frequency physiological streams with real injury outcomes" | Lövdal et al. 2021 has daily logs with real injuries | keep original wording | team
- D-006 | 2026-10-13 | Keep every Review 1 item (6 datasets, TimeGAN/WGAN-GP, tiers, LSTM/GRU/TCN + RF, SHAP + attention, dashboard), each with a defined role; datasets' final roles decided after an audit | Review 1 commitments must stay visible; the audit keeps the use honest | dropping datasets up front | Naman
- D-007 | 2026-10-13 | Code on GitHub (main); data stays local; development with Claude Code on the Windows RTX 4050 laptop | single source of truth, backups, GPU access | local-only git with zip snapshots | Naman
- D-008 | 2026-10-13 | Tiers run as plain Python processes, not Docker | less setup, same measurable latency/bandwidth claim | Docker Compose + MQTT | Naman + Claude
- D-009 | 2026-10-13 | Paper target: IEEE-style conference draft for Review 2; journal later | Review 2 rubric asks for communication to a conference | journal first | Naman
