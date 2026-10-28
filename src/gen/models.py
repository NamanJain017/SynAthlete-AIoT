"""Class-conditional time-series generators for piece 5 (D-003, D-027).

Every generator learns from ONE round's training windows only and makes new windows for a
requested class. Shapes: X (n, time, channels) float32, scaled; y = class index 0..K-1.
- TimeGAN: Yoon, Jarrett & van der Schaar (2019), embedding/recovery/generator/supervisor/
  discriminator GRUs trained in three phases; the class is a one-hot input to every network.
- WGAN-GP: Gulrajani et al. (2017), 1D convolutional generator and critic, gradient penalty.
- TimeVAE: Desai et al. (2021), convolutional variational autoencoder (the non-GAN baseline).
- classic: jitter, scaling and time-warp of real training windows (Um et al. 2017).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

Sampler = Callable[[np.ndarray], np.ndarray]  # class indices -> windows


class _FastCudnn:
    """While a generator trains: cuDNN benchmark mode on (TimeVAE without it asked for a 9 GB
    workspace and fell back to a very slow algorithm), strict deterministic mode off, and TF32
    matrix maths. Restores the previous settings after."""

    def __enter__(self):
        self.prev = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                     torch.are_deterministic_algorithms_enabled())
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False
        # the TSTR/discriminator checks switch strict deterministic mode on (seed_everything);
        # it made generator training about 1.5x slower, so it is off while a generator trains
        torch.use_deterministic_algorithms(False)
        self.precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")  # TF32 matrix maths on this GPU

    def __exit__(self, *exc):
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = self.prev[:2]
        torch.use_deterministic_algorithms(self.prev[2], warn_only=True)
        torch.set_float32_matmul_precision(self.precision)


def _fast(train):
    def wrapped(*args, **kwargs):
        with _FastCudnn():
            return train(*args, **kwargs)
    return wrapped


def _onehot(y: torch.Tensor, k: int) -> torch.Tensor:
    return F.one_hot(y, k).float()


def _tensors(X: np.ndarray, y: np.ndarray, device: torch.device):
    return (torch.as_tensor(X, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device))


def _batches(X: torch.Tensor, y: torch.Tensor, batch: int, gen: torch.Generator):
    """Endless random mini-batches (with replacement), from one seeded generator."""
    n = len(X)
    while True:
        idx = torch.randint(0, n, (batch,), generator=gen, device=X.device)
        yield X[idx], y[idx]


def _sample_in_chunks(make: Callable[[torch.Tensor], torch.Tensor], y: np.ndarray,
                      device: torch.device, chunk: int = 1024) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(y), chunk):
            out.append(make(torch.as_tensor(y[s:s + chunk], device=device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------- TimeGAN


class _RNN(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int, layers: int, act: str | None) -> None:
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_out)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.out(self.rnn(x)[0])
        return torch.sigmoid(h) if self.act == "sigmoid" else h


def train_timegan(X: np.ndarray, y: np.ndarray, k: int, s: dict, seed: int,
                  device: torch.device) -> Sampler:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)  # on the GPU: no copies
    T, C, H, L, Z = X.shape[1], X.shape[2], s["hidden"], s["layers"], s["noise_dim"]
    emb, rec = _RNN(C + k, H, H, L, "sigmoid").to(device), _RNN(H, H, C, L, None).to(device)
    gnet, sup = _RNN(Z + k, H, H, L, "sigmoid").to(device), _RNN(H + k, H, H, max(L - 1, 1),
                                                                    "sigmoid").to(device)
    disc = _RNN(H + k, H, 1, L, None).to(device)
    Xt, yt = _tensors(X, y, device)
    it = _batches(Xt, yt, s["batch_size"], gen)
    opt = lambda *m: torch.optim.Adam([p for x in m for p in x.parameters()], lr=s["lr"])  # noqa: E731
    o_er, o_gs, o_d = opt(emb, rec), opt(gnet, sup), opt(disc)
    o_er_joint = opt(emb, rec)

    def cond(x, c):  # repeat the one-hot class along time
        return torch.cat([x, c[:, None, :].expand(-1, x.shape[1], -1)], dim=2)

    def noise(n):
        return torch.rand(n, T, Z, generator=gen, device=device)

    for _ in range(s["embed_steps"]):  # phase 1: autoencoder
        xb, yb = next(it)
        c = _onehot(yb, k)
        loss = F.mse_loss(rec(emb(cond(xb, c))), xb)
        o_er.zero_grad()
        loss.backward()
        o_er.step()
    for _ in range(s["supervised_steps"]):  # phase 2: next-step supervisor on real embeddings
        xb, yb = next(it)
        c = _onehot(yb, k)
        h = emb(cond(xb, c)).detach()
        loss = F.mse_loss(sup(cond(h[:, :-1], c)), h[:, 1:])
        o_gs.zero_grad()
        loss.backward()
        o_gs.step()
    bce = nn.BCEWithLogitsLoss()
    for _ in range(s["joint_steps"]):  # phase 3: joint adversarial training
        for _ in range(2):  # generator twice per step, as in the paper's code
            xb, yb = next(it)
            c = _onehot(yb, k)
            h = emb(cond(xb, c))
            e_hat = gnet(cond(noise(len(xb)), c))
            h_hat = sup(cond(e_hat, c))
            x_hat = rec(h_hat)
            y_fake, y_fake_e = disc(cond(h_hat, c)), disc(cond(e_hat, c))
            g_adv = bce(y_fake, torch.ones_like(y_fake)) + s["gamma"] * bce(
                y_fake_e, torch.ones_like(y_fake_e))
            g_sup = F.mse_loss(sup(cond(h, c))[:, :-1], h.detach()[:, 1:])
            g_mom = (torch.abs(x_hat.std(0) - xb.std(0)).mean()
                     + torch.abs(x_hat.mean(0) - xb.mean(0)).mean())
            loss = g_adv + 100 * torch.sqrt(g_sup + 1e-12) + 100 * g_mom
            o_gs.zero_grad()
            loss.backward()
            o_gs.step()
            h = emb(cond(xb, c))  # embedder/recovery keep reconstructing
            e_loss = F.mse_loss(rec(h), xb)
            e_loss = 10 * torch.sqrt(e_loss + 1e-12) + 0.1 * F.mse_loss(
                sup(cond(h, c))[:, :-1], h.detach()[:, 1:])
            o_er_joint.zero_grad()
            e_loss.backward()
            o_er_joint.step()
        xb, yb = next(it)
        c = _onehot(yb, k)
        h = emb(cond(xb, c)).detach()
        e_hat = gnet(cond(noise(len(xb)), c)).detach()
        h_hat = sup(cond(e_hat, c)).detach()
        y_real, y_fake, y_fake_e = disc(cond(h, c)), disc(cond(h_hat, c)), disc(cond(e_hat, c))
        d_loss = (bce(y_real, torch.ones_like(y_real)) + bce(y_fake, torch.zeros_like(y_fake))
                  + s["gamma"] * bce(y_fake_e, torch.zeros_like(y_fake_e)))
        if d_loss.item() > 0.15:  # train the discriminator only when it is not winning
            o_d.zero_grad()
            d_loss.backward()
            o_d.step()

    for m in (emb, rec, gnet, sup, disc):
        m.eval()

    def make(yb: torch.Tensor) -> torch.Tensor:
        c = _onehot(yb, k)
        return rec(sup(cond(gnet(cond(noise(len(yb)), c)), c)))

    return lambda y_new: _sample_in_chunks(make, y_new, device)


# ---------------------------------------------------------------- WGAN-GP


def _up_stack(ch: int, T: int, n_out: int) -> nn.Module:
    """Linear seed of length T/8, then three x2 transposed convolutions."""
    layers = []
    for _ in range(3):
        layers += [nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1), nn.BatchNorm1d(ch),
                   nn.ReLU()]
    return nn.Sequential(*layers, nn.Conv1d(ch, n_out, 5, padding=2))


class _ConvGen(nn.Module):
    def __init__(self, n_in: int, ch: int, T: int, C: int) -> None:
        super().__init__()
        self.ch, self.t0 = ch, T // 8
        self.seed = nn.Linear(n_in, ch * self.t0)
        self.up = _up_stack(ch, T, C)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.seed(z).view(len(z), self.ch, self.t0)
        return self.up(h).transpose(1, 2)  # (n, T, C)


class _ConvCritic(nn.Module):
    def __init__(self, C: int, k: int, ch: int, T: int, n_out: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C + k, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2))
        self.head = nn.Linear(ch * (T // 8), n_out)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        xc = torch.cat([x, c[:, None, :].expand(-1, x.shape[1], -1)], dim=2).transpose(1, 2)
        return self.head(self.net(xc).flatten(1))


def train_wgan_gp(X: np.ndarray, y: np.ndarray, k: int, s: dict, seed: int,
                  device: torch.device) -> Sampler:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)  # on the GPU: no copies
    T, C, Z = X.shape[1], X.shape[2], s["noise_dim"]
    g = _ConvGen(Z + k, s["channels"], T, C).to(device)
    d = _ConvCritic(C, k, s["channels"], T).to(device)
    og = torch.optim.Adam(g.parameters(), lr=s["lr"], betas=tuple(s["betas"]))
    od = torch.optim.Adam(d.parameters(), lr=s["lr"], betas=tuple(s["betas"]))
    it = _batches(*_tensors(X, y, device), s["batch_size"], gen)

    def fake(c):
        return g(torch.cat([torch.randn(len(c), Z, generator=gen, device=device), c], dim=1))

    for _ in range(s["steps"]):
        for _ in range(s["n_critic"]):
            xb, yb = next(it)
            c = _onehot(yb, k)
            xf = fake(c).detach()
            eps = torch.rand(len(xb), 1, 1, generator=gen, device=device)
            xi = (eps * xb + (1 - eps) * xf).requires_grad_(True)
            grad = torch.autograd.grad(d(xi, c).sum(), xi, create_graph=True)[0]
            gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
            loss = d(xf, c).mean() - d(xb, c).mean() + s["gp_lambda"] * gp
            od.zero_grad()
            loss.backward()
            od.step()
        _, yb = next(it)
        c = _onehot(yb, k)
        loss = -d(fake(c), c).mean()
        og.zero_grad()
        loss.backward()
        og.step()
    g.eval()
    return lambda y_new: _sample_in_chunks(lambda yb: fake(_onehot(yb, k)), y_new, device)


# ---------------------------------------------------------------- TimeVAE


class _VAE(nn.Module):
    def __init__(self, C: int, k: int, ch: int, T: int, latent: int) -> None:
        super().__init__()
        self.enc = _ConvCritic(C, k, ch, T, n_out=2 * latent)
        self.dec = _ConvGen(latent + k, ch, T, C)
        self.latent = latent

    def forward(self, x, c):
        mu, logvar = self.enc(x, c).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(torch.cat([z, c], dim=1)), mu, logvar


def train_timevae(X: np.ndarray, y: np.ndarray, k: int, s: dict, seed: int,
                  device: torch.device) -> Sampler:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)  # on the GPU: no copies
    T, C = X.shape[1], X.shape[2]
    vae = _VAE(C, k, s["channels"], T, s["latent"]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=s["lr"])
    it = _batches(*_tensors(X, y, device), s["batch_size"], gen)
    for _ in range(s["steps"]):
        xb, yb = next(it)
        x_hat, mu, logvar = vae(xb, _onehot(yb, k))
        recon = F.mse_loss(x_hat, xb, reduction="sum") / len(xb)
        kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(xb)
        loss = recon + s["kl_weight"] * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    vae.eval()

    def make(yb: torch.Tensor) -> torch.Tensor:
        z = torch.randn(len(yb), vae.latent, generator=gen, device=device)
        return vae.dec(torch.cat([z, _onehot(yb, k)], dim=1))

    return lambda y_new: _sample_in_chunks(make, y_new, device)


# ---------------------------------------------------------------- classic augmentation


def classic_augment(X: np.ndarray, rng: np.random.Generator, s: dict) -> np.ndarray:
    """Jitter + per-channel scaling + smooth time-warp of real training windows (Um et al.)."""
    n, T, C = X.shape
    out = X * rng.normal(1.0, s["scaling_sd"], (n, 1, C))
    knots = np.linspace(0, T - 1, s["time_warp_knots"] + 2)
    grid = np.arange(T)
    for i in range(n):
        shift = rng.normal(0, s["time_warp_sd"] * T / (len(knots) - 1), len(knots))
        warp = np.interp(grid, knots, knots + shift)
        warp = np.clip(np.sort(warp), 0, T - 1)
        out[i] = np.stack([np.interp(warp, grid, out[i, :, c]) for c in range(C)], axis=1)
    return (out + rng.normal(0, s["jitter_sd"], out.shape)).astype(np.float32)


def classic_sampler(X: np.ndarray, y: np.ndarray, s: dict, seed: int) -> Sampler:
    """Augmented copies of real training windows of the requested classes."""
    rng = np.random.default_rng(seed)

    def sample(y_new: np.ndarray) -> np.ndarray:
        idx = np.array([rng.choice(np.flatnonzero(y == c)) for c in y_new])
        return classic_augment(X[idx], rng, s)

    return sample


TRAINERS = {"timegan": _fast(train_timegan), "wgan_gp": _fast(train_wgan_gp),
            "timevae": _fast(train_timevae)}
