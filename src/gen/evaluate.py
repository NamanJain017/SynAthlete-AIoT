"""Piece-5 generator checks (EVAL_PROTOCOL "Generator criteria", D-027).

All comparisons use REAL windows of people the generator never saw (the round's test people),
except privacy, which by definition compares with the generator's own training windows.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.eval.metrics import macro_f1
from src.models import nets


def discriminative_accuracy(real: np.ndarray, syn: np.ndarray, seed: int,
                            device: torch.device, epochs: int = 10,
                            min_steps: int = 600) -> float:
    """Post-hoc GRU classifier: real vs synthetic, trained on 70 %, accuracy on held-out 30 %.
    0.5 = indistinguishable (the TimeGAN paper's discriminative score is |acc - 0.5|).
    At least `min_steps` updates: an under-trained classifier would score every generator
    near 0.5 and so flatter it."""
    rng = np.random.default_rng(seed)
    n = min(len(real), len(syn))
    X = np.concatenate([real[rng.choice(len(real), n, replace=False)],
                        syn[rng.choice(len(syn), n, replace=False)]]).astype(np.float32)
    y = np.r_[np.ones(n), np.zeros(n)].astype(np.int64)
    order = rng.permutation(2 * n)
    cut = int(0.7 * 2 * n)
    tr, te = order[:cut], order[cut:]
    nets.seed_everything(seed)
    model = nets.GRUNet(X.shape[2], 1, hidden=32, layers=1, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.as_tensor(X[tr], device=device), torch.as_tensor(y[tr], device=device).float()
    gen = torch.Generator().manual_seed(seed)
    model.train()
    steps = max(epochs * -(-len(tr) // 128), min_steps)
    for _ in range(steps):
        idx = torch.randint(0, len(tr), (min(128, len(tr)),), generator=gen).to(device)
        loss = loss_fn(model(Xt[idx])[:, 0], yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = nets.predict_scores(model, X[te], device=device)
    return float(np.mean((p >= 0.5) == (y[te] == 1)))


def _acf(X: np.ndarray, lags: int) -> np.ndarray:
    """Mean autocorrelation per channel and lag (1..lags), averaged over windows."""
    Xc = X - X.mean(axis=1, keepdims=True)
    var = (Xc ** 2).sum(axis=1)
    out = np.empty((lags, X.shape[2]))
    for lag in range(1, lags + 1):
        num = (Xc[:, lag:] * Xc[:, :-lag]).sum(axis=1)
        out[lag - 1] = np.nanmean(np.where(var > 0, num / np.where(var > 0, var, 1), np.nan),
                                  axis=0)
    return out


def acf_error(real: np.ndarray, syn: np.ndarray, lags: int = 10) -> float:
    return float(np.nanmean(np.abs(_acf(real, lags) - _acf(syn, lags))))


def correlation_error(real: np.ndarray, syn: np.ndarray) -> float:
    """Mean absolute difference of the channel-correlation matrices (off-diagonal)."""
    def corr(X):
        c = np.corrcoef(X.reshape(-1, X.shape[2]), rowvar=False)
        return np.nan_to_num(c)
    diff = np.abs(corr(real) - corr(syn))
    off = ~np.eye(diff.shape[0], dtype=bool)
    return float(diff[off].mean())


def _nearest(a: np.ndarray, b: np.ndarray, device: torch.device, chunk: int = 2048) -> np.ndarray:
    """Euclidean distance from each row of a to its nearest row of b (flattened windows)."""
    bt = torch.as_tensor(b.reshape(len(b), -1), device=device)
    out = []
    for s in range(0, len(a), chunk):
        at = torch.as_tensor(a[s:s + chunk].reshape(-1, bt.shape[1]), device=device)
        out.append(torch.cdist(at, bt).min(dim=1).values.cpu().numpy())
    return np.concatenate(out)


def privacy(train: np.ndarray, test: np.ndarray, syn: np.ndarray,
            device: torch.device) -> dict:
    """Share of synthetic windows closer to a training window than the 5th percentile of
    real test-to-training nearest distances (a memorising generator gives a high share)."""
    p5 = float(np.percentile(_nearest(test, train, device), 5))
    d_syn = _nearest(syn, train, device)
    return {"real_test_p5_distance": p5, "share_synthetic_closer": float(np.mean(d_syn < p5))}


def cnn_macro_f1(X_train, y_train, r, setting: dict, seed: int, device,
                 score_test: bool = True) -> tuple[float, float | None]:
    """The round's piece-3 CNN setting trained on (X_train, y_train) with early stopping on the
    REAL validation person; returns (validation macro-F1, test macro-F1 or None).
    score_test=False while choosing settings, so test people are not even looked at then."""
    nets.seed_everything(seed)
    classes = np.unique(y_train)
    model = nets.build_model("cnn1d", X_train.shape[2], len(classes), setting)
    res = nets.train_model(model, "multiclass", X_train, y_train, r.val.X, r.val.y,
                           lr=setting["lr"], batch_size=setting["batch_size"],
                           max_epochs=setting["max_epochs"], patience=setting["patience"],
                           seed=seed, device=device)
    if not score_test:
        return float(res["best_score"]), None
    test = macro_f1(r.test.y, nets.predict_labels(res["model"], r.test.X, res["classes"],
                                                  device=device))
    return float(res["best_score"]), float(test)
