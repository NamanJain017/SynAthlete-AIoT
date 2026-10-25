"""Neural networks for pieces 3 and 7, and one training loop for all of them.

Task card: docs/handoff/WP-03-nets.md. Tests: tests/handoff/test_wp03_nets.py.
- PAMAP2 activity recognition (piece 3): CNN1D and GRUNet on 128 x 10 sensor clips, many classes.
- Injury risk (piece 3): GRUNet on 7 days x 10 features, one output (binary).
- Fused model (piece 7, declared in configs/fused_candidates.yaml): GRUNet, LSTMNet and TCN.
Every network takes x of shape (batch, time, channels) and returns logits (batch, n_outputs).
This file never reads data: it receives arrays that were already split and scaled elsewhere
(src/models, written by Claude Code), so nothing here can see validation or test people.
"""

from __future__ import annotations

import copy
import math
import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from src.eval.metrics import average_precision, macro_f1

ArrayLike = Sequence | np.ndarray

# Set to True only when everything below is written; the tests skip while it is False.
IMPLEMENTED = True

NETWORKS = ("cnn1d", "gru", "lstm", "tcn")
# Setting names that belong to training, not to the network; build_model ignores them.
TRAINING_KEYS = frozenset({"lr", "weight_decay", "pos_weight", "batch_size", "max_epochs",
                           "patience", "optimizer"})


def seed_everything(seed: int) -> None:
    """Make runs repeatable.

    Seeds Python's `random`, NumPy's global generator and torch (CPU and every GPU) with `seed`;
    sets torch.backends.cudnn.deterministic = True and torch.backends.cudnn.benchmark = False;
    calls torch.use_deterministic_algorithms(True, warn_only=True); and sets the environment
    variable CUBLAS_WORKSPACE_CONFIG to ":4096:8" if it is not set yet (PyTorch reproducibility
    notes: needed for deterministic GPU matrix maths).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def pick_device(prefer_gpu: bool = True) -> torch.device:
    """torch.device("cuda") if prefer_gpu and torch.cuda.is_available(), else "cpu"."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNN1D(nn.Module):
    """Small 1D convolutional network for sensor clips.

    Block b (b = 0 .. conv_blocks-1), in order:
      Conv1d(in = n_channels if b == 0 else channels, out = channels, kernel_size = kernel,
             padding = "same", bias = True) -> BatchNorm1d(channels) -> ReLU -> MaxPool1d(2)
    then the mean over the time axis -> Dropout(dropout) -> Linear(channels, n_outputs).
    forward(x): x (batch, time, n_channels) -> logits (batch, n_outputs).
    ValueError in forward if time < 2 ** conv_blocks (the pooling would run out of steps).
    """

    def __init__(self, n_channels: int, n_outputs: int, channels: int, kernel: int,
                 dropout: float, conv_blocks: int = 3) -> None:
        super().__init__()
        self._conv_blocks = conv_blocks
        blocks = []
        for b in range(conv_blocks):
            in_ch = n_channels if b == 0 else channels
            blocks.extend([
                nn.Conv1d(in_ch, channels, kernel_size=kernel, padding="same", bias=True),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
        self.conv = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_time = 2 ** self._conv_blocks
        if x.shape[1] < min_time:
            raise ValueError(
                f"time dimension {x.shape[1]} is too small; need at least {min_time}"
            )
        # (batch, time, channels) -> (batch, channels, time) for Conv1d
        out = self.conv(x.transpose(1, 2))
        # mean over the time axis
        out = out.mean(dim=2)
        out = self.drop(out)
        return self.head(out)


class GRUNet(nn.Module):
    """GRU (Cho et al. 2014) that reads the steps in order.

    nn.GRU(n_channels, hidden, num_layers = layers, batch_first = True,
           dropout = dropout if layers > 1 else 0.0)
    -> the output at the LAST time step -> Dropout(dropout) -> Linear(hidden, n_outputs).
    forward(x): x (batch, time, n_channels) -> logits (batch, n_outputs).
    """

    def __init__(self, n_channels: int, n_outputs: int, hidden: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        # PyTorch warns if dropout > 0 with num_layers=1
        self.rnn = nn.GRU(n_channels, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        # last time step
        out = out[:, -1, :]
        out = self.drop(out)
        return self.head(out)


class LSTMNet(nn.Module):
    """Exactly like GRUNet, with nn.LSTM in place of nn.GRU (same arguments, same head)."""

    def __init__(self, n_channels: int, n_outputs: int, hidden: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.drop(out)
        return self.head(out)


class _CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding so the output length equals the input length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int) -> None:
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              dilation=dilation, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad on the left only
        x = nn.functional.pad(x, (self._pad, 0))
        return self.conv(x)


class _ResidualBlock(nn.Module):
    """One TCN residual block: two causal convolutions with ReLU+Dropout, plus a shortcut."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            _CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1x1 shortcut when widths differ
        self.shortcut = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                         if in_channels != out_channels else None)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.branch(x)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return self.relu(branch + shortcut)


class TCN(nn.Module):
    """Temporal convolutional network (Bai, Kolter & Koltun 2018), WITHOUT weight normalisation.

    Residual block i (i = 0 .. levels-1) with dilation d = 2 ** i and input width
    w = n_channels if i == 0 else channels:
      branch = two times, in order: [causal Conv1d(in = w for the first conv else channels,
               out = channels, kernel_size = kernel, dilation = d, bias = True) -> ReLU ->
               Dropout(dropout)], where "causal" means (kernel - 1) * d zeros are added on the
               LEFT of the time axis only, so the output at step t uses only steps <= t and the
               length stays the same
      shortcut = Conv1d(w, channels, kernel_size = 1) if w != channels, else the input itself
      block output = ReLU(branch + shortcut)
    then the LAST time step -> Linear(channels, n_outputs).
    The last output therefore sees exactly the last 1 + 2 * (kernel - 1) * (2 ** levels - 1) steps.
    forward(x): x (batch, time, n_channels) -> logits (batch, n_outputs).
    """

    def __init__(self, n_channels: int, n_outputs: int, channels: int, kernel: int,
                 levels: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        for i in range(levels):
            in_ch = n_channels if i == 0 else channels
            dilation = 2 ** i
            blocks.append(_ResidualBlock(in_ch, channels, kernel, dilation, dropout))
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, channels) -> (batch, channels, time) for Conv1d
        out = self.network(x.transpose(1, 2))
        # last time step
        out = out[:, :, -1]
        return self.head(out)


# Keys each network constructor expects (excluding n_channels and n_outputs).
_NET_KEYS: dict[str, tuple[set[str], set[str]]] = {
    "cnn1d": ({"channels", "kernel", "dropout"}, {"conv_blocks"}),
    "gru": ({"hidden", "layers", "dropout"}, set()),
    "lstm": ({"hidden", "layers", "dropout"}, set()),
    "tcn": ({"channels", "kernel", "levels", "dropout"}, set()),
}


def build_model(name: str, n_channels: int, n_outputs: int, params: dict) -> nn.Module:
    """Create one network from a settings dict (one draw of the random search).

    name "cnn1d" -> CNN1D(channels, kernel, dropout, conv_blocks [optional, default 3]);
    "gru" -> GRUNet(hidden, layers, dropout); "lstm" -> LSTMNet(hidden, layers, dropout);
    "tcn" -> TCN(channels, kernel, levels, dropout).
    Keys in TRAINING_KEYS are ignored. ValueError for an unknown name, a missing key or any
    other unknown key (a typo must never be silently ignored).
    """
    if name not in _NET_KEYS:
        raise ValueError(f"unknown network name: {name!r}")
    required, optional = _NET_KEYS[name]
    net_params = {k: v for k, v in params.items() if k not in TRAINING_KEYS}
    unknown = set(net_params) - required - optional
    if unknown:
        raise ValueError(f"unknown keys for {name}: {unknown}")
    missing = required - set(net_params)
    if missing:
        raise ValueError(f"missing keys for {name}: {missing}")
    constructors = {"cnn1d": CNN1D, "gru": GRUNet, "lstm": LSTMNet, "tcn": TCN}
    return constructors[name](n_channels, n_outputs, **net_params)


class EarlyStopping:
    """Remembers the best validation score so far (higher is better).

    EarlyStopping(patience): ValueError if patience < 1.
    update(score) records one epoch's score and returns True if it is a new best: the first
    score always is; later ones only if STRICTLY greater than the best so far (a tie keeps the
    earlier epoch). ValueError if score is not a finite number.
    Attributes: best_score (None before the first update), best_epoch (0-based; -1 before the
    first update), should_stop (True once `patience` updates in a row were not a new best).
    """

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self._patience = patience
        self.best_score: float | None = None
        self.best_epoch: int = -1
        self.should_stop: bool = False
        self._epoch: int = -1  # current epoch counter
        self._wait: int = 0  # epochs since last improvement

    def update(self, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score}")
        self._epoch += 1
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = self._epoch
            self._wait = 0
            return True
        self._wait += 1
        if self._wait >= self._patience:
            self.should_stop = True
        return False


def positive_weight(y: ArrayLike, setting: float | str) -> float:
    """Weight of the positive class for binary training (the grid's pos_weight).

    setting "balanced" -> (number of 0s) / (number of 1s) in y; a number -> float(setting).
    ValueError if y is not all 0/1, or for "balanced" when y has no 1s.
    """
    arr = np.asarray(y)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("y must contain only 0s and 1s")
    if setting == "balanced":
        n_pos = int(np.sum(arr == 1))
        if n_pos == 0:
            raise ValueError("y has no positive examples for balanced weighting")
        return float(np.sum(arr == 0)) / float(n_pos)
    return float(setting)


def make_loss(task: str, pos_weight: float = 1.0) -> nn.Module:
    """task "binary" -> nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weight]));
    task "multiclass" -> nn.CrossEntropyLoss() (pos_weight unused). ValueError otherwise."""
    if task == "binary":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown task: {task!r}")


def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def predict_scores(model: nn.Module, X: ArrayLike, batch_size: int = 1024,
                   device: str | torch.device = "cpu") -> np.ndarray:
    """Scores for every row, in row order, as a float64 NumPy array.

    Puts the model in eval mode (it stays in eval mode) and uses torch.no_grad(); moves the
    model to `device`; works through X in batches of `batch_size`.
    Model with 1 output -> shape (N,): sigmoid probabilities of class 1.
    Model with k > 1 outputs -> shape (N, k): softmax probabilities (each row sums to 1).
    ValueError if X is not 3-D or contains NaN or infinity.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D, got {X.ndim}-D")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinity")

    model = model.to(device)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(1)
            else:
                probs = torch.softmax(logits, dim=1)
            parts.append(probs.cpu().numpy())

    return np.concatenate(parts).astype(np.float64)


def predict_labels(model: nn.Module, X: ArrayLike, classes: ArrayLike, batch_size: int = 1024,
                   device: str | torch.device = "cpu") -> np.ndarray:
    """classes[argmax of predict_scores(model, X, ...)] for a multi-output model.

    `classes` are the original labels (e.g. PAMAP2 activity ids such as 1, 2, ..., 24), in the
    order of the model's outputs. Ties go to the lowest output index (numpy.argmax).
    ValueError for a model with 1 output or if len(classes) differs from the number of outputs.
    """
    scores = predict_scores(model, X, batch_size=batch_size, device=device)
    if scores.ndim == 1:
        raise ValueError("predict_labels requires a multi-output model (got 1 output)")
    classes = np.asarray(classes)
    if len(classes) != scores.shape[1]:
        raise ValueError(
            f"len(classes)={len(classes)} differs from number of outputs={scores.shape[1]}"
        )
    return classes[np.argmax(scores, axis=1)]


def train_model(model: nn.Module, task: str, X_train: ArrayLike, y_train: ArrayLike,
                X_val: ArrayLike, y_val: ArrayLike, *, lr: float, batch_size: int,
                max_epochs: int, patience: int, seed: int, weight_decay: float = 0.0,
                pos_weight: float = 1.0, device: str | torch.device = "cpu") -> dict:
    """Train with early stopping on the validation score; keep the best epoch's weights.

    task "multiclass": classes = numpy.unique(y_train) (sorted); the model must have
      len(classes) outputs; the loss uses each label's position in classes; validation score =
      src.eval.metrics.macro_f1(y_val, predict_labels(model, X_val, classes, device=device)).
      Validation may contain labels missing from training (they count in macro-F1).
    task "binary": y_train and y_val must be 0/1 and y_val must contain both; the model must
      have 1 output; validation score = src.eval.metrics.average_precision(y_val,
      predict_scores(model, X_val, device=device)).
    Each epoch: model.train(); shuffle the training rows with ONE torch.Generator created at the
    start and seeded with `seed` (torch.randperm(n, generator=...)); go through them in batches
    of batch_size (the last, smaller batch is used too); loss from make_loss(task, pos_weight),
    optimizer from make_optimizer(model, lr, weight_decay), created once before the first epoch.
    Then score validation and call EarlyStopping(patience).update(score); on a new best, keep a
    copy of the weights; stop when should_stop or after max_epochs. Finally load the best weights.
    Returns {"model": the same model object with the best weights, on `device`,
             "classes": the classes array (multiclass) or None (binary),
             "best_epoch": int (0-based), "best_score": float, "epochs_run": int,
             "history": [{"epoch": int, "train_loss": float, "val_score": float}, ...]}
    train_loss is the mean loss over the epoch's training rows. Plain Python ints and floats.
    ValueError: unknown task; X not 3-D or not finite; X and y lengths differ; wrong number of
    model outputs; binary labels not 0/1 or y_val with one class only; batch_size, max_epochs
    or patience < 1.
    """
    # --- validation of inputs ---
    if task not in ("binary", "multiclass"):
        raise ValueError(f"unknown task: {task!r}")
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val)
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3-D, got {X_train.ndim}-D")
    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train contains NaN or infinity")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths differ")
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val lengths differ")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")

    # Determine number of model outputs by running a probe
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        probe = torch.tensor(X_train[:1], dtype=torch.float32, device=device)
        n_outputs = model(probe).shape[1]

    classes = None
    if task == "multiclass":
        classes = np.unique(y_train)
        if n_outputs != len(classes):
            raise ValueError(
                f"model has {n_outputs} outputs but training data has {len(classes)} classes"
            )
        # Map labels to positions
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        y_train_idx = np.array([label_to_idx[label] for label in y_train])
    elif task == "binary":
        if n_outputs != 1:
            raise ValueError(f"binary task requires 1 output, model has {n_outputs}")
        if not np.all((y_train == 0) | (y_train == 1)):
            raise ValueError("binary labels must be 0 or 1")
        if not np.all((y_val == 0) | (y_val == 1)):
            raise ValueError("binary labels must be 0 or 1")
        if len(np.unique(y_val)) < 2:
            raise ValueError("y_val must contain both classes (0 and 1)")

    # --- setup ---
    loss_fn = make_loss(task, pos_weight).to(device)
    optimizer = make_optimizer(model, lr, weight_decay)
    es = EarlyStopping(patience)
    gen = torch.Generator()
    gen.manual_seed(seed)

    n = len(X_train)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    if task == "binary":
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
    else:
        y_train_t = torch.tensor(y_train_idx, dtype=torch.long)

    best_weights = None
    history = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        total_samples = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            logits = model(xb)
            if task == "binary":
                # targets must be (batch, 1) float for BCEWithLogitsLoss
                loss = loss_fn(logits, yb.unsqueeze(1))
            else:
                loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_len = len(idx)
            total_loss += loss.item() * batch_len
            total_samples += batch_len

        train_loss = total_loss / total_samples

        # Validation score
        if task == "multiclass":
            preds = predict_labels(model, X_val, classes, device=device)
            val_score = macro_f1(y_val, preds)
        else:
            scores = predict_scores(model, X_val, device=device)
            val_score = average_precision(y_val, scores)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_score": float(val_score),
        })

        is_best = es.update(val_score)
        if is_best:
            best_weights = copy.deepcopy(model.state_dict())

        if es.should_stop:
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model = model.to(device)

    return {
        "model": model,
        "classes": classes,
        "best_epoch": int(es.best_epoch),
        "best_score": float(es.best_score),
        "epochs_run": int(len(history)),
        "history": history,
    }
