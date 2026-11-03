# How to present the 13 figures (`figures/`)

## The story order, about 10 minutes
Tell one story. Don't show the figures in file order. A good sequence is: **does it predict injuries → is that result trustworthy → the other objectives → what the model looks like inside.**

| Step | Figures | Time |
|---|---|---|
| 1. Injury prediction works, but only modestly | 1, 8 | 2 min |
| 2. The result is honest and holds on unseen runners | 2, 11, 3 | 2.5 min |
| 3. What an alert means in practice | 9, 10 | 1.5 min |
| 4. The other objectives: synthetic data, IoT, explanations | 4, 7, 5, 6 | 3 min |
| 5. Extras if asked | 12, 13 | as needed |

If time is short, show only **1, 2, 9, 5, 6**.

**For every figure, use three steps:**
1. Say what it measures: one sentence about the axes.
2. Point to the one number that matters.
3. Say what it means, honestly.

---

## 1. `1_injury_models.png`: every injury model
- **What it shows:** each bar is one model's score on runners it never trained on (mean of 5 rounds; the line on each bar is the spread). Left is ROC-AUC (0.5 = guessing), right is PR-AUC.
- **Say:** "All learned models score between 0.58 and 0.63 ROC-AUC: above guessing, but only modestly. The red bar is our fused model, at 0.609. It's red because we declared it in advance, not because it won."
- **Likely question:** *"Why is no model clearly best?"* **Answer:** daily training logs carry only a weak warning signal before an injury, so every method hits the same ceiling. The honest test shows this; leaky tests in other papers hide it.

## 2. `2_locked_final_test.png`: the sealed final test
- **What it shows:** dots are scores, lines are the uncertainty range. The top row is development (5 rounds), the red row is the sealed final test on 10 runners nobody looked at, and the grey rows are secondary results.
- **Say:** "Ten runners were locked away from day one and opened once, after the code was frozen. The model scored 0.605, against 0.609 in development. Because it holds on unseen people, our results were not over-tuned."
- **Likely question:** *"The Lövdal model scored 0.649 there, so why not use it?"* **Answer:** choosing a model after seeing the test would be cheating. It was also scored on different rows and trained on more runners, and the intervals overlap heavily. We say no model is clearly best.

## 3. `3_paired_comparisons.png`: which differences are real
- **What it shows:** each row compares a model with the Lövdal label-free baseline on the same rows. A dot is the difference, and the line is its 95% range. Green means clearly better (the range is entirely above 0), red means clearly worse, grey means not clear.
- **Say:** "On the day file, logistic regression and the GRU are clearly better than the baseline, and CatBoost is too on PR-AUC. Constant risk and the ACWR coaching rule are clearly worse. Most other differences are not clear."
- **Likely question:** *"Why not just compare averages?"* **Answer:** averages hide luck. Resampling runners 2,000 times shows whether a gap could be chance. We also say openly that 24 comparisons were not adjusted for multiple testing.

## 4. `4_generators.png`: synthetic wearable data
- **What it shows:** on the left, how often a detector can tell fake clips from real ones (0.5 = fooled; the green band is the pass zone). On the right, activity-recognition scores when training on synthetic only, or on real plus synthetic, against real only (red line).
- **Say:** "Every generator's clips are easy to spot: 0.86–0.96 accuracy, far from 0.5. Adding synthetic data to real data changes little. Our pass marks were written before the runs, so this is an honest negative result."
- **Likely question:** *"So the GANs failed?"* **Answer:** they didn't reach realism on this small dataset, and we report that rather than hiding it. It's why synthetic data is only used for training, never as evidence.

## 5. `7_injury_augmentation.png`: fake injury examples
- **What it shows:** GRU scores with real data only, then with TimeGAN injury examples added, then with classic augmentation added.
- **Say:** "Adding synthetic injury examples gave no gain. PR-AUC is even slightly lower (0.022 and 0.021, against 0.023 real-only)."

## 6. `5_edge_fog_cloud.png`: the IoT benefit
- **What it shows:** on the left, bytes sent to the cloud on a log scale (tiered vs cloud-only). On the right, decision time in milliseconds.
- **Say:** "Processing at the edge sends 593 times less data for the wearable stream and 85 times less for the heart-rate stream. Decisions take a few milliseconds either way."
- **Likely question:** *"Why is tiered slower for PAMAP2?"* **Answer:** everything ran on one laptop, so there is no network delay to save. The real benefit is bandwidth and battery: 593× less data.

## 7. `6_explanations.png`: explanations tested, not assumed
- **What it shows:** on the left, how many of 5 rounds each explanation passed the three faithfulness tests. On the right, how often the explanations matched what we expected **before** running.
- **Say:** "The explanations passed all three tests in all 5 rounds. Recent training load raised risk, as expected, in 10 of 10 cases. But 'low recovery raises risk' was mostly **not** seen, only 2 of 10. We report that too."
- **Likely question:** *"What is the deletion test?"* **Answer:** we hide the inputs the explanation says matter most, then hide random inputs. If hiding the "important" ones changes the risk more, the explanation is faithful.

## 8. `8_roc_pr_curves.png`: curves behind the scores
- **What it shows:** in ROC (left), the higher the curve above the diagonal, the better. In precision–recall (right), the dashed line is chance (injury rate 0.013).
- **Say:** "All curves stay close to the diagonal and to the chance line. Precision stays around 0.02 to 0.03, so most alerts are false. That's the same modest result, shown completely."
- **Note:** the curves pool all 5 rounds, so the numbers in figure 1 (per-round means) are the official ones.

## 9. `9_confusion_injury.png`: what an alert means
- **What it shows:** four boxes: healthy and no alert (correct), healthy with an alert (false alarm), injury with no alert (missed), injury with an alert (caught). The alert level was chosen on validation runners, never on test.
- **Say:** "On the sealed test the model caught 33 of 48 injuries (69%), but it also raised 1,519 false alarms, flagging about half of all healthy days. As a warning system it is too noisy for real use, and we say so."
- **Likely question:** *"Why not raise the threshold?"* **Answer:** fewer false alarms means more missed injuries. The threshold rule was fixed in advance on validation; changing it after seeing the test would be cheating.

## 10. `10_calibration_and_risk_score.png`: risk score quality
- **What it shows:** on the left, calibration. If the dots sit on the dashed line, a "2% risk" really means 2%. On the right, the 0–100 risk score for injured days (red) and healthy days (blue).
- **Say:** "Predicted risk roughly matches reality, but the highest-risk group is over-predicted. Injured days score higher (median 68 against 54), but the two groups overlap a lot, so the score is a hint, not a diagnosis."

## 11. `11_per_round_scores.png`: the spread behind "±"
- **What it shows:** every model's score in each of the 5 rounds.
- **Say:** "Scores move by several points from round to round. The ranking changes too, which is why we report means with spreads and never one lucky split."

## 12. `12_fused_vs_arms.png`: did combining help?
- **What it shows:** the fused model next to its day-by-day part and its weekly-totals part.
- **Say:** "Combining gives almost the same score as either part (0.609 against 0.604 and 0.612). Both views of the data carry the same weak signal."

## 13. `13_confusion_activity.png`: activity recognition
- **What it shows:** rows are true activities, columns are predictions; a darker diagonal is better. It shows round 0's two test people.
- **Say:** "Clear movements are recognised 94–99% of the time (walking, running, cycling, Nordic walking). Quiet activities get confused with each other, because they look alike to body sensors (sitting, standing, computer work, folding laundry, house cleaning). This re-run reproduces our recorded score exactly (macro-F1 0.754)."

---

## Phrases that keep you safe
- "Above chance, but modest." Never say "accurate" or "high accuracy" about injuries.
- "Declared in advance": that's why the fused model is highlighted.
- "Not a comparison": use it for the secondary results on the sealed test.
- "Every number comes from a result file written by code."
- "A correct negative result is worth more than an impressive wrong one."
