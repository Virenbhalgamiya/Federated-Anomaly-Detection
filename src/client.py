"""Federated client: local VAE training, optional DP-SGD (Opacus) with cumulative accounting."""
from collections import OrderedDict
from dataclasses import dataclass

import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from .model import get_model, train_epochs


@dataclass
class FLConfig:
    n_clients: int = 3
    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    noise_multiplier: float = 0.0   # 0 disables DP
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    prox_mu: float = 0.0            # 0 disables FedProx


class Client:
    """One institution. The client (and its privacy accountant) lives for the whole run,
    so epsilon accumulates across rounds instead of resetting each round."""

    def __init__(self, X, input_dim, cfg, device="cpu"):
        self.cfg, self.device = cfg, device
        dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        self.n = len(dataset)
        model = get_model(input_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.engine = None
        if cfg.noise_multiplier > 0:
            self.engine = PrivacyEngine(accountant="rdp")
            model, opt, loader = self.engine.make_private(
                module=model, optimizer=opt, data_loader=loader,
                noise_multiplier=cfg.noise_multiplier, max_grad_norm=cfg.max_grad_norm)
        self.model, self.opt, self.loader = model, opt, loader

    @property
    def base(self):
        return getattr(self.model, "_module", self.model)  # unwrap Opacus' GradSampleModule

    def get_parameters(self):
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.base.state_dict().items())

    def set_parameters(self, params):
        self.base.load_state_dict(params, strict=True)  # raises on any mismatch, never silent

    def fit(self):
        use_prox = self.cfg.prox_mu > 0
        ref = [p.detach().clone() for p in self.base.parameters()] if use_prox else None
        loss = train_epochs(self.model, self.opt, self.loader, self.cfg.local_epochs, self.device,
                            prox_ref=ref, prox_mu=self.cfg.prox_mu, lr=self.cfg.lr)
        return self.get_parameters(), self.n, loss

    def epsilon(self):
        return self.engine.get_epsilon(self.cfg.delta) if self.engine else float("inf")
