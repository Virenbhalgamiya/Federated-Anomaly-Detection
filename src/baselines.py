"""Non-federated baselines with the same total training budget as the federated runs."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import evaluate, get_model, train_epochs


def _train_standard(X, splits, cfg, epochs, seed, device):
    torch.manual_seed(seed)
    model = get_model(splits.input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
                        batch_size=cfg.batch_size, shuffle=True)
    best = {"auc": -1.0}
    for ep in range(1, epochs + 1):
        train_epochs(model, opt, loader, 1, device)
        val = evaluate(model, splits.X_val, splits.y_val, device)
        if val["auc"] > best["auc"]:
            best = {"auc": val["auc"], "threshold": val["threshold"], "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return evaluate(model, splits.X_test, splits.y_test, device, threshold=best["threshold"])


def run_centralized(splits, cfg, seed, device="cpu"):
    """All normal training data pooled in one place. Upper reference, no privacy."""
    return _train_standard(splits.X_train_normal, splits, cfg, cfg.rounds * cfg.local_epochs, seed, device)


def run_local_only(client_data, splits, cfg, seed, device="cpu"):
    """Each client trains alone on its own data; metrics are averaged over clients."""
    results = [_train_standard(X, splits, cfg, cfg.rounds * cfg.local_epochs, seed + i, device)
               for i, X in enumerate(client_data)]
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}
