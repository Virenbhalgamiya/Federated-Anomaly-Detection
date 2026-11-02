"""Variational autoencoder, training loop and evaluation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, 24)
        self.fc5 = nn.Linear(24, input_dim)

    def encode(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = F.leaky_relu(self.fc3(z), 0.2)
        h = F.leaky_relu(self.fc4(h), 0.2)
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight=0.1):
    """Per-sample mean of summed-squared reconstruction error plus a down-weighted KL term."""
    mse = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (mse + kl_weight * kld) / x.size(0)


def get_model(input_dim):
    return VariationalAutoencoder(input_dim=input_dim)


def train_epochs(model, optimizer, loader, epochs, device, prox_ref=None, prox_mu=0.0, lr=1e-3):
    """Train for `epochs` passes over `loader`. Works with or without Opacus wrapping.

    FedProx is applied as a proximal pull toward the global weights *after* each optimizer step.
    Adding the term to the loss does not work under Opacus: DP-SGD rebuilds the gradient from
    per-sample gradients and drops any loss term that does not flow through the layers
    (see checks/fedprox_under_dp.py). The pull only uses the public global weights, so it is
    post-processing and costs no privacy budget.
    """
    base = getattr(model, "_module", model)
    model.train()
    total, steps = 0.0, 0
    for _ in range(epochs):
        for (x,) in loader:
            if x.size(0) == 0:
                continue
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            if prox_ref is not None and prox_mu > 0:
                with torch.no_grad():
                    for p, g in zip(base.parameters(), prox_ref):
                        p.sub_(lr * prox_mu * (p - g))
            total += loss.item()
            steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def score_samples(model, X, device="cpu"):
    """Deterministic anomaly score: reconstruction error from the latent mean (no sampling noise)."""
    base = getattr(model, "_module", model)
    base.eval()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu, _ = base.encode(x)
    return ((base.decode(mu) - x) ** 2).mean(dim=1).cpu().numpy()


def evaluate(model, X, y, device="cpu", threshold=None):
    """ROC AUC, average precision and precision/recall/F1 at a threshold.

    If `threshold` is None it is chosen by Youden's J on (X, y); pass the validation threshold
    when scoring the test set so the test labels never influence any choice.
    """
    scores = score_samples(model, X, device)
    fpr, tpr, thr = roc_curve(y, scores)
    if threshold is None:
        threshold = float(thr[(tpr - fpr).argmax()])
    preds = (scores > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "threshold": threshold,
    }
