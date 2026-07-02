"""deep.py — Mult-VAE (Liang et al., WWW 2018), the standard deep recommender for
implicit feedback, run through this project's frozen evaluation harness.

The point is an honest, apples-to-apples deep-learning comparison: same per-user
split, same metrics as ALS / EASE / BM25. Torch is imported lazily and only used
here, so the rest of the project stays torch-free.

Architecture: a denoising variational autoencoder over each user's bag-of-items
vector — encoder MLP -> latent (mu, logvar) -> decoder MLP -> multinomial logits
over all items. Trained with the multinomial log-likelihood + KL, with KL
annealed. Recommendations = decoder logits with the user's train items masked.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(4)


class MultVAE(nn.Module):
    def __init__(self, n_items: int, hidden: int = 600, latent: int = 200, dropout: float = 0.5):
        super().__init__()
        self.latent = latent
        self.drop = nn.Dropout(dropout)
        self.enc1 = nn.Linear(n_items, hidden)
        self.enc2 = nn.Linear(hidden, latent * 2)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        for m in (self.enc1, self.enc2, self.dec1, self.dec2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        h = torch.tanh(self.enc1(h))
        stats = self.enc2(h)
        mu, logvar = stats[:, :self.latent], stats[:, self.latent:]
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        h = torch.tanh(self.dec1(z))
        return self.dec2(h), mu, logvar


def train_multvae(train: sp.csr_matrix, epochs: int = 60, batch_size: int = 256,
                  hidden: int = 600, latent: int = 200, dropout: float = 0.5,
                  lr: float = 1e-3, beta_max: float = 0.2, seed: int = 0) -> MultVAE:
    """Fit Mult-VAE on binarised user-item interactions. Deterministic given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.from_numpy((train > 0).astype(np.float32).toarray())
    n_users, n_items = X.shape
    model = MultVAE(n_items, hidden, latent, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    total_steps = epochs * max(1, n_users // batch_size)
    anneal_steps = int(0.6 * total_steps)  # ramp KL weight over the first 60%
    step = 0
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n_users), max(1, n_users // batch_size)):
            xb = X[idx]
            logits, mu, logvar = model(xb)
            log_softmax = F.log_softmax(logits, dim=1)
            neg_ll = -(log_softmax * xb).sum(1).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            beta = beta_max * min(1.0, step / max(1, anneal_steps))
            loss = neg_ll + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return model


@torch.no_grad()
def multvae_recs(model: MultVAE, train: sp.csr_matrix, user_rows: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k item indices per user, train items masked out. Same signature as models.*_recs."""
    model.eval()
    train = train.tocsr()
    user_rows = np.asarray(user_rows)
    out = np.empty((len(user_rows), k), dtype=np.int64)
    for start in range(0, len(user_rows), 512):
        rows = user_rows[start:start + 512]
        xb = torch.from_numpy((train[rows] > 0).astype(np.float32).toarray())
        logits, _, _ = model(xb)
        logits[xb > 0] = -np.inf                      # never recommend a train item
        out[start:start + len(rows)] = torch.topk(logits, k, dim=1).indices.numpy()
    return out
