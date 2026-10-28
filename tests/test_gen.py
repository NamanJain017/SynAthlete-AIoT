"""Piece-5 generators and checks on small made-up data (CPU, seconds)."""

import numpy as np
import pytest
import torch

from src.gen import evaluate as ge
from src.gen import models

CPU = torch.device("cpu")
TINY = {
    "timegan": dict(batch_size=16, embed_steps=3, supervised_steps=3, joint_steps=2, gamma=1.0,
                    hidden=8, layers=2, lr=1e-3, noise_dim=4),
    "wgan_gp": dict(batch_size=16, steps=2, gp_lambda=10.0, betas=[0.5, 0.9], channels=8,
                    noise_dim=8, n_critic=2, lr=1e-4),
    "timevae": dict(batch_size=16, steps=3, latent=4, channels=8, kl_weight=1.0, lr=1e-3),
}


def toy(n=40, t=16, c=3, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    X = rng.normal(size=(n, t, c)).astype(np.float32) + y[:, None, None]
    return X, y


@pytest.mark.parametrize("name", list(TINY))
def test_generators_make_windows_of_the_right_shape_and_are_seeded(name):
    X, y = toy()
    a = models.TRAINERS[name](X, y, 3, TINY[name], seed=0, device=CPU)(np.array([0, 2, 1, 1]))
    b = models.TRAINERS[name](X, y, 3, TINY[name], seed=0, device=CPU)(np.array([0, 2, 1, 1]))
    assert a.shape == (4, 16, 3) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.allclose(a, b, atol=1e-5)  # same seed, same synthetic windows


def test_classic_augmentation_keeps_shape_and_class_source():
    X, y = toy()
    out = models.classic_sampler(X, y, {"jitter_sd": 0.03, "scaling_sd": 0.1,
                                        "time_warp_knots": 4, "time_warp_sd": 0.2}, 0)(
        np.array([2, 2, 2]))
    assert out.shape == (3, 16, 3)
    assert abs(out.mean() - X[y == 2].mean()) < 1.0  # copies of class-2 windows, not class 0


def test_fidelity_measures_on_identical_and_different_data():
    X, _ = toy(n=60)
    assert ge.acf_error(X, X) == pytest.approx(0.0)
    assert ge.correlation_error(X, X) == pytest.approx(0.0)
    p = ge.privacy(X[:30], X[30:], X[:30].copy(), CPU)  # copies of training = memorised
    assert p["share_synthetic_closer"] == 1.0
    acc = ge.discriminative_accuracy(X, X + 5.0, 0, CPU, epochs=5)  # trivially different
    assert acc > 0.9


def test_retry_recovers_from_a_brief_out_of_memory(monkeypatch):
    from src.gen import run
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    calls = []

    def flaky(x):
        calls.append(x)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return x * 2

    assert run._retry(flaky, 21) == 42 and len(calls) == 2


def test_retry_gives_up_after_the_last_attempt(monkeypatch):
    from src.gen import run
    monkeypatch.setattr(run.time, "sleep", lambda s: None)

    def always(x):
        raise torch.OutOfMemoryError("simulated")

    with pytest.raises(torch.OutOfMemoryError):
        run._retry(always, 1)


def test_checkpoint_is_written_whole(tmp_path):
    from src.gen import run
    p = tmp_path / "pamap2_r0_timegan.json"
    run._save_json(p, {"tstr": 0.5})
    assert p.exists() and not p.with_suffix(".tmp").exists()
    assert __import__("json").loads(p.read_text())["tstr"] == 0.5
