"""Federated Averaging server loop (custom simulation, no external FL framework)."""
import copy
from collections import OrderedDict

import numpy as np
import torch

from .client import Client
from .model import evaluate, get_model


def fedavg(updates):
    """Sample-count-weighted average of client state dicts: [(state_dict, n_samples), ...]."""
    total = sum(n for _, n in updates)
    return OrderedDict(
        (k, sum(params[k].float() * (n / total) for params, n in updates)) for k in updates[0][0])


def run_federated(client_data, splits, cfg, seed, device="cpu"):
    """Run FedAvg (optionally FedProx and/or DP-SGD). Returns (test_metrics, per-round history).

    The round is chosen by validation AUC, the decision threshold comes from the validation set,
    and the test set is scored once, at the end.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = [Client(X, splits.input_dim, cfg, device) for X in client_data]
    global_model = get_model(splits.input_dim).to(device)

    history, best = [], {"auc": -1.0}
    for rnd in range(1, cfg.rounds + 1):
        global_params = OrderedDict((k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items())
        updates = []
        for client in clients:
            client.set_parameters(global_params)
            params, n, _ = client.fit()
            updates.append((params, n))
        global_model.load_state_dict(fedavg(updates), strict=True)

        val = evaluate(global_model, splits.X_val, splits.y_val, device)
        eps = max(c.epsilon() for c in clients)  # worst-case client, cumulative over all rounds so far
        history.append({"round": rnd, "val_auc": val["auc"], "epsilon": eps})
        if val["auc"] > best["auc"]:
            best = {"auc": val["auc"], "round": rnd, "threshold": val["threshold"],
                    "state": copy.deepcopy(global_model.state_dict())}

    global_model.load_state_dict(best["state"])
    test = evaluate(global_model, splits.X_test, splits.y_test, device, threshold=best["threshold"])
    test["best_round"] = best["round"]
    test["epsilon"] = history[-1]["epsilon"]  # total budget after all rounds (conservative)
    return test, history
