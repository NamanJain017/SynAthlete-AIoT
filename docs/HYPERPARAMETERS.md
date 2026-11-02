# Model settings (hyperparameters)

Copied from the config files (`configs/`) and from the final run's record (`experiments/lockbox/2026-10-29_010127/`).

## Rules shared by every tuned model
- **Settings tried:** 8 per model per round, by seeded random search over the grid. The grid was written before any run. Round k uses search seed 1000 + k; the final model used 1005.
- **How the best setting is chosen:** on validation runners only, by validation PR-AUC (macro-F1 for activities). Early stopping also watches validation.
- **Seeds:** settings are tried with seed 0. The chosen setting is retrained with seeds 0, 1 and 2, and the results are averaged.
- **Scaling:** one z-score scaler, fitted on training runners only. Platt calibration and the alert level are set on validation.

## Injury models (`configs/baselines/lovdal.yaml`)
| Model | Fixed | Grid searched |
|---|---|---|
| Logistic regression | L2, lbfgs, max_iter 2000 | C 1e-4 to 1e2 (log scale); class_weight none or balanced |
| Random forest | 500 trees | max_depth 4/6/8/12/none; min_samples_leaf 1/5/20/50; max_features sqrt/0.3/0.5; class_weight none or balanced_subsample |
| CatBoost | up to 2,000 iterations, early stopping after 100, eval metric PR-AUC | learning_rate 0.01–0.1; depth 4/6/8; l2_leaf_reg 1–10; class weights none or Balanced |
| GRU (7 days × 10 features) | batch 128, up to 60 epochs, patience 10, Adam | hidden 16/32/64; layers 1/2; dropout 0/0.2/0.4; lr 3e-4 to 3e-3; weight_decay 0 or 1e-4; pos_weight 1 or balanced |
| Lövdal bagged XGBoost (paper settings, not tuned) | 9 bags, 2,048 samples per class, learning rate 0.01, depth 2 or 3, 256 or 512 trees | – |
| Constant risk, ACWR rule | no training settings | – |

## Fused model (`configs/fused_candidates.yaml`): 44 tries per round
- **Sequence part:** GRU or LSTM (GRU grid), or TCN (batch 128, up to 60 epochs, patience 10, Adam). TCN grid: channels 16/32/64; kernel 2/3; levels 2/3; dropout 0/0.2; lr 3e-4 to 3e-3.
- **Tabular part:** random forest or CatBoost (grids above).
- **Combiner:** logistic regression, C from 0.01/0.1/1/10. It is trained on 5-fold out-of-fold predictions, grouped by runner.

**Final model, as chosen for the locked test:**
- **GRU:** hidden 32, 2 layers, dropout 0.4, lr 0.00166, weight_decay 1e-4, pos_weight balanced, batch 128, up to 60 epochs, patience 10.
- **CatBoost:** depth 4, learning_rate 0.0194, l2_leaf_reg 8.52, no class weights, up to 2,000 iterations, early stopping after 100.
- **Combiner C per seed:** 0.1, 0.01 and 1.0.

## Activity models (PAMAP2, `configs/baselines/pamap2.yaml`)
- **1D CNN:** 3 conv blocks, batch 128, up to 40 epochs, patience 8, Adam. Grid: channels 32/64; kernel 5/9; dropout 0.2/0.5; lr 3e-4 to 3e-3.
- **GRU:** batch 128, up to 40 epochs, patience 8, Adam. Grid: hidden 32/64/128; layers 1/2; dropout 0/0.3; lr range in the config.
- **Input:** windows of 128 steps × 10 channels at 25 Hz.

## Generators (`configs/gen/generators.yaml`): seeds 3000 + round, 8 settings each
- **TimeGAN:** batch 128; 1,000 embedding steps, 1,000 supervised steps, 1,500 joint steps; gamma 1. Grid: hidden 24/48/96; layers 2/3; lr 5e-4 to 2e-3; noise_dim 8/16.
- **WGAN-GP (untuned, one declared setting):** channels 64, noise_dim 64, n_critic 5, lr 2e-4; batch 128, 3,000 steps, gradient-penalty λ 10, Adam betas (0.5, 0.9).
- **TimeVAE:** batch 128, 3,000 steps. Grid: latent 8/16/32; channels 32/64/128; KL weight 0.5/1/2; lr 3e-4 to 3e-3.
- **Classic augmentation:** jitter sd 0.03, scaling sd 0.1, time-warp with 4 knots and sd 0.2.
- **How settings are chosen:** a CNN is trained on synthetic data only, then scored on the real validation person (TSTR macro-F1).

## Tiers benchmark
One window every 0.1 s; 3 repetitions; sessions of 12 windows.
