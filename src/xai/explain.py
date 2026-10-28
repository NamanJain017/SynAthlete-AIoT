"""Piece-8 explanations (PLAN piece 8, D-027(3), invariant 12).

- TreeSHAP on the tabular arm (random forest or CatBoost, week features).
- Integrated gradients (captum) on the sequence arm (GRU, LSTM or TCN, 7 days x 10 features).
- Attention view: a small GRU with attention pooling, trained on the same round data for
  explanations only (never part of the risk score), always shown next to integrated gradients.
- 1D Grad-CAM on the PAMAP2 activity CNN (the convolutional model of the project).
Baseline for "removing" an input = 0, which is the training mean after the round's scaler.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------- attributions


def tree_shap(model, X: np.ndarray) -> np.ndarray:
    """TreeSHAP values for class 1, shape (n, features)."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def integrated_gradients(model: nn.Module, X: np.ndarray, device, steps: int = 32,
                         batch: int = 256) -> np.ndarray:
    """IG of the risk logit w.r.t. the inputs, baseline 0 (training mean), shape like X."""
    from captum.attr import IntegratedGradients
    model = model.to(device).eval()
    ig = IntegratedGradients(lambda x: model(x)[:, 0])
    out = []
    with torch.backends.cudnn.flags(enabled=False):  # RNN backward needs training-mode cuDNN
        for s in range(0, len(X), batch):
            x = torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device)
            out.append(ig.attribute(x, baselines=torch.zeros_like(x), n_steps=steps)
                       .detach().cpu().numpy())
    return np.concatenate(out)


class AttentionGRU(nn.Module):
    """GRU -> one attention score per time step -> softmax -> weighted sum -> Linear."""

    def __init__(self, n_channels: int, n_outputs: int, hidden: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        self.rnn = nn.GRU(n_channels, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_outputs)
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        w = torch.softmax(self.score(h)[:, :, 0], dim=1)
        self.last_weights = w.detach()
        return self.head(self.drop((w[:, :, None] * h).sum(dim=1)))


def attention_weights(model: AttentionGRU, X: np.ndarray, device, batch: int = 1024):
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            model(torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device))
            out.append(model.last_weights.cpu().numpy())
    return np.concatenate(out)


def grad_cam_1d(model: nn.Module, X: np.ndarray, device, batch: int = 256) -> np.ndarray:
    """1D Grad-CAM on the last convolution of a CNN1D for its predicted class: one importance
    value per time step of that layer, upsampled to the input length, scaled to [0, 1]."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    store = {}
    h1 = convs[-1].register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = convs[-1].register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    model = model.to(device).eval()
    out = []
    for s in range(0, len(X), batch):
        x = torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device)
        logits = model(x)
        model.zero_grad()
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
        act, grad = store["act"], store["grad"]
        cam = torch.relu((grad.mean(dim=2, keepdim=True) * act).sum(dim=1))  # (n, t')
        cam = nn.functional.interpolate(cam[:, None], size=X.shape[1], mode="linear")[:, 0]
        cam = cam / cam.amax(dim=1, keepdim=True).clamp_min(1e-12)
        out.append(cam.detach().cpu().numpy())
    h1.remove()
    h2.remove()
    return np.concatenate(out)


# ---------------------------------------------------------------- plain language


def reason_sentence(names: list[str], x_row: np.ndarray, attr_row: np.ndarray, top: int = 3) -> str:
    """'Risk is higher mainly because <feature> was high/low, ...' from the largest positive
    contributions; x is scaled, so its sign says above or below the training average."""
    order = [i for i in np.argsort(-attr_row) if attr_row[i] > 0][:top]
    if not order:
        return "No input raises this risk above the average."
    parts = [f"{names[i]} was {'higher' if x_row[i] > 0 else 'lower'} than usual"
             for i in order]
    return "Risk is higher mainly because " + "; ".join(parts) + "."
