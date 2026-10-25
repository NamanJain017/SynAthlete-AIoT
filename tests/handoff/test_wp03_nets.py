"""Answer checker for task card WP-03 (docs/handoff/WP-03-nets.md): src/models/nets.py.

Written by Claude Code before the code exists; builders must not edit this file.
Network sizes are checked against parameter-count formulas from the card's layer lists;
training is checked on small made-up data (CPU, a few seconds). No real data is used.
"""

import copy
import inspect
import math
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from src.eval.metrics import macro_f1
from src.models import nets

# ---------- parameter counts written out from the card's layer lists ----------

def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cnn_count(n_ch, n_out, c, k, blocks=3):
    first = n_ch * c * k + c + 2 * c  # conv weights + bias, batch-norm weight + bias
    later = c * c * k + c + 2 * c
    return first + (blocks - 1) * later + c * n_out + n_out


def rnn_count(gates, n_ch, n_out, h, layers):
    layer = lambda inp: gates * (h * inp + h * h + 2 * h)  # noqa: E731  (PyTorch has 2 biases)
    return layer(n_ch) + (layers - 1) * layer(h) + h * n_out + n_out


def tcn_count(n_ch, n_out, c, k, levels):
    first = (n_ch * c * k + c) + (c * c * k + c) + ((n_ch * c + c) if n_ch != c else 0)
    return first + (levels - 1) * 2 * (c * c * k + c) + c * n_out + n_out


# ---------- small made-up data ----------

def easy_multiclass(n, seed, t=16, classes=(3, 7, 24)):
    """The class decides which channel is shifted up by 2: easy to learn."""
    rng = np.random.default_rng(seed)
    y = rng.choice(classes, size=n)
    X = rng.normal(size=(n, t, 3)).astype(np.float32)
    for j, label in enumerate(classes):
        X[y == label, :, j] += 2.0
    return X, y


def easy_binary(n, seed, t=16):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    y[:2] = [0, 1]  # both classes always present
    X = rng.normal(size=(n, t, 3)).astype(np.float32)
    X[y == 1, :, 0] += 2.0
    return X, y


def fit(model, task, X, y, Xv, yv, **overrides):
    settings = dict(lr=3e-3, batch_size=32, max_epochs=15, patience=5, seed=0)
    settings.update(overrides)
    return nets.train_model(model, task, X, y, Xv, yv, **settings)


class FirstStep(nn.Module):
    """Fixed 'model' whose logits are the first time step: makes predictions exact to check."""

    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        return x[:, 0, : self.width]


# ---------- networks ----------

def test_network_names():
    assert nets.NETWORKS == ("cnn1d", "gru", "lstm", "tcn")


@pytest.mark.parametrize("n_out", [1, 4])
@pytest.mark.parametrize("model", [
    lambda o: nets.CNN1D(3, o, channels=8, kernel=5, dropout=0.2),
    lambda o: nets.GRUNet(3, o, hidden=8, layers=2, dropout=0.2),
    lambda o: nets.LSTMNet(3, o, hidden=8, layers=1, dropout=0.0),
    lambda o: nets.TCN(3, o, channels=8, kernel=3, levels=2, dropout=0.2),
])
def test_output_shape(model, n_out):
    out = model(n_out).eval()(torch.randn(5, 16, 3))
    assert tuple(out.shape) == (5, n_out)


def test_cnn_size_matches_the_card():
    cnn = nets.CNN1D(10, 18, channels=32, kernel=5, dropout=0.2)
    assert n_params(cnn) == cnn_count(10, 18, 32, 5)
    assert n_params(nets.CNN1D(10, 12, channels=64, kernel=9, dropout=0.5, conv_blocks=2)) == \
        cnn_count(10, 12, 64, 9, blocks=2)


@pytest.mark.parametrize("layers", [1, 2])
def test_gru_and_lstm_sizes_match_the_card(layers):
    assert n_params(nets.GRUNet(10, 1, hidden=32, layers=layers, dropout=0.2)) == \
        rnn_count(3, 10, 1, 32, layers)
    assert n_params(nets.LSTMNet(10, 18, hidden=64, layers=layers, dropout=0.0)) == \
        rnn_count(4, 10, 18, 64, layers)


def test_tcn_size_matches_the_card():
    assert n_params(nets.TCN(10, 1, channels=16, kernel=2, levels=3, dropout=0.2)) == \
        tcn_count(10, 1, 16, 2, 3)
    # input width equal to channels: no 1x1 shortcut convolution
    assert n_params(nets.TCN(16, 2, channels=16, kernel=3, levels=2, dropout=0.0)) == \
        tcn_count(16, 2, 16, 3, 2)


@pytest.mark.parametrize("kernel,levels", [(2, 2), (3, 2), (2, 3)])
def test_tcn_is_causal_and_sees_exactly_its_receptive_field(kernel, levels):
    torch.manual_seed(0)
    field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)
    t = field + 5
    model = nets.TCN(3, 2, channels=8, kernel=kernel, levels=levels, dropout=0.0).eval()
    x = torch.randn(4, t, 3)
    inside, outside = x.clone(), x.clone()
    inside[:, t - field, :] += 5.0       # the oldest step the last output may see
    outside[:, : t - field, :] += 5.0    # everything older than that
    with torch.no_grad():
        base = model(x)
        assert not torch.allclose(model(inside), base)
        assert torch.allclose(model(outside), base, atol=1e-6)


def test_rnns_read_the_whole_sequence():
    torch.manual_seed(0)
    for model in (nets.GRUNet(3, 2, hidden=8, layers=1, dropout=0.0),
                  nets.LSTMNet(3, 2, hidden=8, layers=2, dropout=0.0)):
        model.eval()
        x = torch.randn(4, 12, 3)
        changed = x.clone()
        changed[:, 0, :] += 5.0  # the very first step
        with torch.no_grad():
            assert not torch.allclose(model(changed), model(x))


def _head_input(model, x):
    """What the network's single Linear head receives (found by layer type, not by name)."""
    heads = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(heads) == 1
    seen = {}
    hook = heads[0].register_forward_pre_hook(lambda m, inp: seen.setdefault("x", inp[0]))
    with torch.no_grad():
        model(x)
    hook.remove()
    return seen["x"]


def _layer_output(model, kind, x):
    """Output of the LAST layer of type `kind` (the recurrent layer, or the last max-pool)."""
    layers = [m for m in model.modules() if isinstance(m, kind)]
    seen = {}
    hook = layers[-1].register_forward_hook(lambda m, inp, out: seen.__setitem__("out", out))
    with torch.no_grad():
        model(x)
    hook.remove()
    return seen["out"]


@pytest.mark.parametrize("net,rnn", [(lambda: nets.GRUNet(3, 2, hidden=8, layers=2, dropout=0.0),
                                      nn.GRU),
                                     (lambda: nets.LSTMNet(3, 2, hidden=8, layers=1, dropout=0.0),
                                      nn.LSTM)])
def test_rnn_head_reads_the_last_time_step(net, rnn):
    torch.manual_seed(0)
    model, x = net().eval(), torch.randn(4, 9, 3)
    out = _layer_output(model, rnn, x)[0]  # (batch, time, hidden)
    assert torch.allclose(_head_input(model, x), out[:, -1, :], atol=1e-6)


def test_cnn_head_reads_the_mean_over_time():
    torch.manual_seed(0)
    model, x = nets.CNN1D(3, 2, channels=8, kernel=3, dropout=0.0).eval(), torch.randn(4, 16, 3)
    pooled = _layer_output(model, nn.MaxPool1d, x)  # (batch, channels, time)
    assert torch.allclose(_head_input(model, x), pooled.mean(dim=2), atol=1e-6)


def test_single_layer_rnn_gets_no_inner_dropout():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # PyTorch warns if dropout is set with one layer
        nets.GRUNet(3, 2, hidden=8, layers=1, dropout=0.4)
        nets.LSTMNet(3, 2, hidden=8, layers=1, dropout=0.4)


def test_cnn_needs_enough_time_steps():
    model = nets.CNN1D(3, 2, channels=8, kernel=3, dropout=0.0).eval()
    assert tuple(model(torch.randn(2, 8, 3)).shape) == (2, 2)
    with pytest.raises(ValueError):
        model(torch.randn(2, 7, 3))


def test_build_model():
    cnn = nets.build_model("cnn1d", 10, 18, {"channels": 32, "kernel": 5, "dropout": 0.2,
                                              "lr": 1e-3, "batch_size": 128, "max_epochs": 40,
                                              "patience": 8, "optimizer": "adam"})
    assert isinstance(cnn, nets.CNN1D) and n_params(cnn) == cnn_count(10, 18, 32, 5)
    two = nets.build_model("cnn1d", 10, 18, {"channels": 32, "kernel": 5, "dropout": 0.2,
                                              "conv_blocks": 2})
    assert n_params(two) == cnn_count(10, 18, 32, 5, blocks=2)
    gru = nets.build_model("gru", 10, 1, {"hidden": 16, "layers": 2, "dropout": 0.4,
                                          "weight_decay": 1e-4, "pos_weight": "balanced"})
    assert isinstance(gru, nets.GRUNet) and n_params(gru) == rnn_count(3, 10, 1, 16, 2)
    assert isinstance(nets.build_model("lstm", 10, 1, {"hidden": 16, "layers": 1, "dropout": 0.0}),
                      nets.LSTMNet)
    tcn = nets.build_model("tcn", 10, 1, {"channels": 16, "kernel": 2, "levels": 3, "dropout": 0.2})
    assert isinstance(tcn, nets.TCN) and n_params(tcn) == tcn_count(10, 1, 16, 2, 3)


def test_build_model_refuses_mistakes():
    with pytest.raises(ValueError):
        nets.build_model("transformer", 10, 1, {"hidden": 16})
    with pytest.raises(ValueError):  # missing dropout
        nets.build_model("gru", 10, 1, {"hidden": 16, "layers": 1})
    with pytest.raises(ValueError):  # typo must not be ignored
        nets.build_model("gru", 10, 1, {"hidden": 16, "layers": 1, "dropout": 0.0, "hiden": 32})


# ---------- small training pieces ----------

def test_early_stopping_worked_example():
    es = nets.EarlyStopping(patience=2)
    assert es.best_score is None and es.best_epoch == -1 and es.should_stop is False
    assert [es.update(s) for s in [0.1, 0.3, 0.3]] == [True, True, False]  # a tie is no gain
    assert es.best_epoch == 1 and es.best_score == 0.3 and es.should_stop is False
    assert es.update(0.2) is False and es.should_stop is True


def test_early_stopping_counter_restarts_after_a_new_best():
    es = nets.EarlyStopping(patience=2)
    for s in [0.5, 0.4, 0.6, 0.6]:
        es.update(s)
    assert es.best_epoch == 2 and es.should_stop is False
    es.update(0.1)
    assert es.should_stop is True
    one = nets.EarlyStopping(patience=1)
    assert one.update(-3.0) is True and one.should_stop is False  # first score is always best
    assert one.update(-3.0) is False and one.should_stop is True


def test_early_stopping_refuses_bad_input():
    with pytest.raises(ValueError):
        nets.EarlyStopping(patience=0)
    es = nets.EarlyStopping(patience=2)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            es.update(bad)


def test_positive_weight():
    assert nets.positive_weight([0, 0, 0, 1], "balanced") == pytest.approx(3.0)
    assert nets.positive_weight(np.array([1, 0, 1, 0, 0]), "balanced") == pytest.approx(1.5)
    assert nets.positive_weight([0, 1], 1) == 1.0
    assert isinstance(nets.positive_weight([0, 1], 1), float)
    assert nets.positive_weight([0, 1], 2.5) == 2.5
    with pytest.raises(ValueError):
        nets.positive_weight([0, 0, 0], "balanced")
    with pytest.raises(ValueError):
        nets.positive_weight([0, 1, 2], 1)


def test_make_loss():
    binary = nets.make_loss("binary", 3.0)
    value = binary(torch.zeros(2, 1), torch.tensor([[1.0], [0.0]]))
    assert value.item() == pytest.approx((3 * math.log(2) + math.log(2)) / 2)
    multi = nets.make_loss("multiclass")
    assert multi(torch.zeros(2, 4), torch.tensor([0, 3])).item() == pytest.approx(math.log(4))
    with pytest.raises(ValueError):
        nets.make_loss("regression")


def test_make_optimizer():
    opt = nets.make_optimizer(nn.Linear(2, 1), lr=0.01, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 0.01 and opt.param_groups[0]["weight_decay"] == 1e-4


def test_seed_everything(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    nets.seed_everything(3)
    first = (torch.rand(3), np.random.rand(3))
    nets.seed_everything(3)
    again = (torch.rand(3), np.random.rand(3))
    assert torch.equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    nets.seed_everything(3)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"  # an existing value is kept


def test_pick_device():
    assert nets.pick_device(prefer_gpu=False) == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert nets.pick_device().type == expected


# ---------- prediction ----------

def test_predict_scores_and_labels_are_exact():
    X = np.array([[[0.0, 5.0, 1.0]], [[2.0, 0.0, 2.0]]], dtype=np.float32)
    probs = nets.predict_scores(FirstStep(3), X)
    e = np.exp(X[:, 0, :] - X[:, 0, :].max(axis=1, keepdims=True))
    assert probs.dtype == np.float64 and probs.shape == (2, 3)
    assert np.allclose(probs, e / e.sum(axis=1, keepdims=True), atol=1e-6)
    # row 0 -> output 1 (label 7); row 1 ties outputs 0 and 2 -> lowest index (label 3)
    assert nets.predict_labels(FirstStep(3), X, np.array([3, 7, 24])).tolist() == [7, 3]
    one = nets.predict_scores(FirstStep(1), np.array([[[0.0]], [[2.0]]], dtype=np.float32))
    assert one.shape == (2,) and np.allclose(one, [0.5, 1 / (1 + math.exp(-2))], atol=1e-6)


def test_predict_uses_batches_and_keeps_row_order():
    X = np.random.default_rng(0).normal(size=(10, 1, 3)).astype(np.float32)
    whole = nets.predict_scores(FirstStep(3), X, batch_size=1024)
    assert np.allclose(nets.predict_scores(FirstStep(3), X, batch_size=3), whole, atol=1e-7)


def test_predict_refuses_mistakes():
    with pytest.raises(ValueError):
        nets.predict_scores(FirstStep(3), np.zeros((4, 3), dtype=np.float32))  # not 3-D
    bad = np.zeros((2, 1, 3), dtype=np.float32)
    bad[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        nets.predict_scores(FirstStep(3), bad)
    X = np.zeros((2, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        nets.predict_labels(FirstStep(1), X[:, :, :1], np.array([0]))  # 1 output
    with pytest.raises(ValueError):
        nets.predict_labels(FirstStep(3), X, np.array([1, 2]))  # wrong number of classes


def test_predict_is_in_eval_mode():
    torch.manual_seed(0)
    model = nets.CNN1D(3, 4, channels=8, kernel=3, dropout=0.5)
    X = np.random.default_rng(1).normal(size=(6, 16, 3)).astype(np.float32)
    first = nets.predict_scores(model, X)
    assert model.training is False
    assert np.array_equal(first, nets.predict_scores(model, X))  # no dropout, fixed batch-norm
    assert np.allclose(first.sum(axis=1), 1.0)


# ---------- training ----------

def test_learns_an_easy_multiclass_task():
    X, y = easy_multiclass(240, seed=1)
    Xv, yv = easy_multiclass(90, seed=2)
    nets.seed_everything(0)
    res = fit(nets.CNN1D(3, 3, channels=16, kernel=3, dropout=0.0), "multiclass", X, y, Xv, yv)
    assert res["classes"].tolist() == [3, 7, 24]
    assert res["best_score"] >= 0.9
    labels = nets.predict_labels(res["model"], Xv, res["classes"])
    assert set(labels.tolist()) <= {3, 7, 24} and macro_f1(yv, labels) >= 0.9


def test_learns_an_easy_binary_task():
    X, y = easy_binary(240, seed=3)
    Xv, yv = easy_binary(90, seed=4)
    nets.seed_everything(0)
    res = fit(nets.GRUNet(3, 1, hidden=16, layers=1, dropout=0.0), "binary", X, y, Xv, yv,
              max_epochs=25, patience=8)
    assert res["classes"] is None and res["best_score"] >= 0.9
    scores = nets.predict_scores(res["model"], Xv)
    assert scores.shape == (90,) and scores.min() >= 0.0 and scores.max() <= 1.0


def test_result_and_history_are_consistent():
    X, y = easy_multiclass(120, seed=5)
    Xv, yv = easy_multiclass(60, seed=6)
    nets.seed_everything(0)
    model = nets.GRUNet(3, 3, hidden=8, layers=1, dropout=0.0)
    res = fit(model, "multiclass", X, y, Xv, yv, max_epochs=6)
    assert res["model"] is model
    h = res["history"]
    assert res["epochs_run"] == len(h) and [r["epoch"] for r in h] == list(range(len(h)))
    for r in h:
        assert set(r) == {"epoch", "train_loss", "val_score"}
        assert type(r["epoch"]) is int
        assert type(r["train_loss"]) is float and type(r["val_score"]) is float
        assert math.isfinite(r["train_loss"]) and r["train_loss"] > 0 and 0 <= r["val_score"] <= 1
    scores = [r["val_score"] for r in h]
    assert type(res["best_epoch"]) is int and type(res["best_score"]) is float
    assert res["best_epoch"] == scores.index(max(scores)) and res["best_score"] == max(scores)


def test_training_is_repeatable_and_uses_the_seed():
    X, y = easy_multiclass(120, seed=7)
    Xv, yv = easy_multiclass(60, seed=8)

    def run(seed):
        nets.seed_everything(0)
        model = nets.GRUNet(3, 3, hidden=8, layers=1, dropout=0.2)
        res = fit(model, "multiclass", X, y, Xv, yv, max_epochs=3, patience=3, seed=seed)
        return res["history"], nets.predict_scores(res["model"], Xv)

    (h1, p1), (h2, p2), (h3, _) = run(0), run(0), run(1)
    assert h1 == h2 and np.array_equal(p1, p2)
    assert [r["train_loss"] for r in h1] != [r["train_loss"] for r in h3]  # other shuffle order


def test_best_weights_are_restored():
    # random labels: the validation score wanders, so training stops early, after the best epoch
    rng = np.random.default_rng(9)
    X, Xv = (rng.normal(size=(n, 16, 3)).astype(np.float32) for n in (120, 60))
    y, yv = rng.choice([0, 1, 2], 120), rng.choice([0, 1, 2], 60)

    def run(max_epochs, patience):
        nets.seed_everything(0)
        return fit(nets.CNN1D(3, 3, channels=8, kernel=3, dropout=0.0), "multiclass", X, y, Xv, yv,
                   lr=1e-2, max_epochs=max_epochs, patience=patience)

    a = run(40, 2)
    assert a["epochs_run"] < 40 and a["epochs_run"] == a["best_epoch"] + 1 + 2
    again = macro_f1(yv, nets.predict_labels(a["model"], Xv, a["classes"]))
    assert again == pytest.approx(a["best_score"])
    b = run(a["best_epoch"] + 1, 1000)  # the same run, cut off right at the best epoch
    assert np.allclose(nets.predict_scores(a["model"], Xv), nets.predict_scores(b["model"], Xv),
                       atol=1e-6)


def test_pos_weight_and_weight_decay_are_used():
    X, y = easy_binary(96, seed=10)
    Xv, yv = easy_binary(40, seed=11)

    def run(**kw):
        nets.seed_everything(0)
        res = fit(nets.GRUNet(3, 1, hidden=8, layers=1, dropout=0.0), "binary", X, y, Xv, yv,
                  max_epochs=2, patience=5, **kw)
        return res["history"][0]["train_loss"], nets.predict_scores(res["model"], Xv)

    base_loss, base_pred = run()
    assert run(pos_weight=10.0)[0] != pytest.approx(base_loss)
    assert not np.allclose(run(weight_decay=0.5)[1], base_pred)


def test_every_training_row_is_used_once_per_epoch_and_the_loss_is_per_row():
    X, y = easy_binary(33, seed=17)  # batches of 32 and 1: the last, small batch must count
    nets.seed_everything(0)
    model = nets.GRUNet(3, 1, hidden=4, layers=1, dropout=0.0)
    start = copy.deepcopy(model)
    rows = []
    hook = model.register_forward_hook(
        lambda m, inp, out: rows.append(len(inp[0])) if m.training else None)
    res = fit(model, "binary", X, y, X, y, lr=0.0, max_epochs=1)  # lr 0: weights stay put
    hook.remove()
    assert sum(rows) == 33
    with torch.no_grad():  # mean loss per ROW over the epoch, not the mean of batch means
        target = torch.tensor(y, dtype=torch.float32)[:, None]
        whole = nets.make_loss("binary")(start(torch.tensor(X)), target)
    assert res["history"][0]["train_loss"] == pytest.approx(whole.item(), rel=1e-5)


def test_validation_may_have_labels_missing_from_training():
    X, y = easy_multiclass(90, seed=12)
    Xv, yv = easy_multiclass(30, seed=13)
    yv[:3] = 9  # an activity nobody did in training
    nets.seed_everything(0)
    res = fit(nets.CNN1D(3, 3, channels=8, kernel=3, dropout=0.0), "multiclass", X, y, Xv, yv,
              max_epochs=3)
    assert res["classes"].tolist() == [3, 7, 24] and 0 <= res["best_score"] <= 1


def test_training_refuses_mistakes():
    X, y = easy_multiclass(40, seed=14)
    Xb, yb = easy_binary(40, seed=15)
    multi = lambda: nets.GRUNet(3, 3, hidden=4, layers=1, dropout=0.0)  # noqa: E731
    single = lambda: nets.GRUNet(3, 1, hidden=4, layers=1, dropout=0.0)  # noqa: E731
    nan_X = X.copy()
    nan_X[0, 0, 0] = np.nan
    bad_calls = [
        lambda: fit(multi(), "regression", X, y, X, y),
        lambda: fit(multi(), "multiclass", X[:, :, 0], y, X, y),           # not 3-D
        lambda: fit(multi(), "multiclass", nan_X, y, X, y),                # NaN
        lambda: fit(multi(), "multiclass", X, y[:-1], X, y),               # lengths differ
        lambda: fit(nets.GRUNet(3, 4, hidden=4, layers=1, dropout=0.0), "multiclass", X, y, X, y),
        lambda: fit(multi(), "binary", Xb, yb, Xb, yb),                    # 3 outputs for binary
        lambda: fit(single(), "binary", Xb, yb + 1, Xb, yb),               # labels not 0/1
        lambda: fit(single(), "binary", Xb, yb, Xb, np.zeros_like(yb)),    # one class in val
        lambda: fit(single(), "binary", Xb, yb, Xb, np.ones_like(yb)),     # (both ways round)
        lambda: fit(multi(), "multiclass", X, y, X, y, batch_size=0),
        lambda: fit(multi(), "multiclass", X, y, X, y, max_epochs=0),
        lambda: fit(multi(), "multiclass", X, y, X, y, patience=0),
    ]
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU (CI and teammates)")
def test_trains_and_predicts_on_the_gpu():
    X, y = easy_binary(64, seed=16)
    nets.seed_everything(0)
    res = fit(nets.GRUNet(3, 1, hidden=8, layers=1, dropout=0.0), "binary", X, y, X, y,
              max_epochs=2, device="cuda")
    assert next(res["model"].parameters()).device.type == "cuda"
    assert nets.predict_scores(res["model"], X, device="cuda").shape == (64,)


def test_never_reads_files():
    source = inspect.getsource(nets)
    for banned in ("open(", "np.load", "read_csv", "data/", "src.data", "Path("):
        assert banned not in source, banned
