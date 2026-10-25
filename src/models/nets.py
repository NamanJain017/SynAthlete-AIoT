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

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

ArrayLike = Sequence | np.ndarray

# Set to True only when everything below is written; the tests skip while it is False.
IMPLEMENTED = False

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
    raise NotImplementedError


def pick_device(prefer_gpu: bool = True) -> torch.device:
    """torch.device("cuda") if prefer_gpu and torch.cuda.is_available(), else "cpu"."""
    raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LSTMNet(nn.Module):
    """Exactly like GRUNet, with nn.LSTM in place of nn.GRU (same arguments, same head)."""

    def __init__(self, n_channels: int, n_outputs: int, hidden: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(name: str, n_channels: int, n_outputs: int, params: dict) -> nn.Module:
    """Create one network from a settings dict (one draw of the random search).

    name "cnn1d" -> CNN1D(channels, kernel, dropout, conv_blocks [optional, default 3]);
    "gru" -> GRUNet(hidden, layers, dropout); "lstm" -> LSTMNet(hidden, layers, dropout);
    "tcn" -> TCN(channels, kernel, levels, dropout).
    Keys in TRAINING_KEYS are ignored. ValueError for an unknown name, a missing key or any
    other unknown key (a typo must never be silently ignored).
    """
    raise NotImplementedError


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
        raise NotImplementedError

    def update(self, score: float) -> bool:
        raise NotImplementedError


def positive_weight(y: ArrayLike, setting: float | str) -> float:
    """Weight of the positive class for binary training (the grid's pos_weight).

    setting "balanced" -> (number of 0s) / (number of 1s) in y; a number -> float(setting).
    ValueError if y is not all 0/1, or for "balanced" when y has no 1s.
    """
    raise NotImplementedError


def make_loss(task: str, pos_weight: float = 1.0) -> nn.Module:
    """task "binary" -> nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weight]));
    task "multiclass" -> nn.CrossEntropyLoss() (pos_weight unused). ValueError otherwise."""
    raise NotImplementedError


def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)."""
    raise NotImplementedError


def predict_scores(model: nn.Module, X: ArrayLike, batch_size: int = 1024,
                   device: str | torch.device = "cpu") -> np.ndarray:
    """Scores for every row, in row order, as a float64 NumPy array.

    Puts the model in eval mode (it stays in eval mode) and uses torch.no_grad(); moves the
    model to `device`; works through X in batches of `batch_size`.
    Model with 1 output -> shape (N,): sigmoid probabilities of class 1.
    Model with k > 1 outputs -> shape (N, k): softmax probabilities (each row sums to 1).
    ValueError if X is not 3-D or contains NaN or infinity.
    """
    raise NotImplementedError


def predict_labels(model: nn.Module, X: ArrayLike, classes: ArrayLike, batch_size: int = 1024,
                   device: str | torch.device = "cpu") -> np.ndarray:
    """classes[argmax of predict_scores(model, X, ...)] for a multi-output model.

    `classes` are the original labels (e.g. PAMAP2 activity ids such as 1, 2, ..., 24), in the
    order of the model's outputs. Ties go to the lowest output index (numpy.argmax).
    ValueError for a model with 1 output or if len(classes) differs from the number of outputs.
    """
    raise NotImplementedError


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
    raise NotImplementedError
