"""Does FedProx do anything when Opacus (DP-SGD) is active?

Compares final weights for prox_mu=0 vs prox_mu=1, same seed and batches, noise_multiplier=0
(clipping only, so runs are deterministic):

  in-loss   the original implementation: (mu/2)*||w - w_global||^2 added to the loss
  post-step the current implementation: proximal pull applied after each optimizer step

Expected: in-loss has an effect without DP but NONE with DP; post-step has an effect in both.
Run:  python checks/fedprox_under_dp.py
"""
import copy
import os
import sys

import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_model, train_epochs, vae_loss

LR = 0.01


def train_in_loss(model, opt, loader, global_model, mu):
    """The original approach, kept here only for the comparison."""
    model.train()
    for (x,) in loader:
        opt.zero_grad()
        recon, m, lv = model(x)
        loss = vae_loss(recon, x, m, lv)
        if mu > 0:
            loss = loss + (mu / 2) * sum(((a - b) ** 2).sum()
                                         for a, b in zip(model.parameters(), global_model.parameters()))
        loss.backward()
        opt.step()


def run(style, mu, with_dp):
    torch.manual_seed(0)
    X = torch.randn(256, 10)
    loader = DataLoader(TensorDataset(X), batch_size=32, shuffle=False)
    model = get_model(10)
    global_model = copy.deepcopy(model)
    for p in global_model.parameters():
        p.data += 0.5  # make the proximal term non-zero
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    if with_dp:
        model, opt, loader = PrivacyEngine().make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=0.0, max_grad_norm=1.0, poisson_sampling=False)
    torch.manual_seed(1)
    if style == "in-loss":
        train_in_loss(model, opt, loader, global_model, mu)
    else:
        ref = [p.detach().clone() for p in global_model.parameters()]
        train_epochs(model, opt, loader, 1, "cpu", prox_ref=ref, prox_mu=mu, lr=LR)
    return torch.cat([p.detach().flatten() for p in model.parameters()])


if __name__ == "__main__":
    for style in ("in-loss", "post-step"):
        for with_dp in (False, True):
            diff = (run(style, 0.0, with_dp) - run(style, 1.0, with_dp)).abs().max().item()
            verdict = "has an effect" if diff > 1e-6 else "NO effect"
            print(f"{style:9s}  DP={with_dp!s:5}  max|w(mu=0)-w(mu=1)| = {diff:.3e}  -> {verdict}")
