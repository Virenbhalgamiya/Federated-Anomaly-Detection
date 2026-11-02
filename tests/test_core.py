import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.client import Client, FLConfig
from src.data import make_splits, partition
from src.model import get_model, score_samples, train_epochs
from src.server import fedavg


def _toy(n=600, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.3).astype(int)
    X[y == 1] += 2.0  # anomalies are shifted
    return X, y


def test_fedavg_is_sample_weighted():
    a = {"w": torch.full((2,), 1.0)}
    b = {"w": torch.full((2,), 4.0)}
    out = fedavg([(a, 1), (b, 3)])
    assert torch.allclose(out["w"], torch.full((2,), 3.25))


@pytest.mark.parametrize("mode", ["iid", "noniid"])
def test_partition_is_disjoint_and_complete(mode):
    X = np.arange(300 * 4, dtype=float).reshape(300, 4)
    parts = partition(X, 3, mode, seed=0)
    assert sum(len(p) for p in parts) == len(X)
    rows = {tuple(r) for p in parts for r in p}
    assert len(rows) == len(X)  # no duplicates, nothing lost


def test_noniid_clients_see_different_regions():
    X, _ = _toy()
    iid = partition(X, 3, "iid", 0)
    skew = partition(X, 3, "noniid", 0)
    spread = lambda parts: np.ptp([p.mean(axis=0)[0] for p in parts])
    assert spread(skew) > spread(iid)


def test_scaler_uses_training_normals_only():
    X, y = _toy()
    s = make_splits(X, y, seed=0)
    assert np.allclose(s.X_train_normal.mean(axis=0), 0, atol=1e-8)
    assert not np.allclose(s.X_test.mean(axis=0), 0, atol=1e-3)  # test was scaled, not re-centred
    assert s.X_train_normal.shape[0] < len(X)
    assert set(np.unique(s.y_val)) <= {0, 1}


def test_set_parameters_rejects_wrong_keys():
    cfg = FLConfig()
    c = Client(np.random.randn(50, 6), 6, cfg)
    with pytest.raises(RuntimeError):
        c.set_parameters({"bogus": torch.zeros(1)})


def test_scores_are_deterministic():
    m = get_model(6)
    X = np.random.randn(20, 6)
    assert np.allclose(score_samples(m, X), score_samples(m, X))


def test_proximal_pull_moves_toward_global():
    torch.manual_seed(0)
    m = get_model(6)
    ref = [p.detach().clone() + 1.0 for p in m.parameters()]
    dist = lambda: sum(((p - r) ** 2).sum() for p, r in zip(m.parameters(), ref)).item()
    X = torch.randn(64, 6)
    loader = [(X,)]
    opt = torch.optim.SGD(m.parameters(), lr=0.0)  # lr=0 isolates the pull from the gradient step
    before = dist()
    train_epochs(m, opt, loader, 1, "cpu", prox_ref=ref, prox_mu=5.0, lr=0.05)
    assert dist() < before


def test_epsilon_accumulates_across_rounds():
    cfg = FLConfig(noise_multiplier=1.0, local_epochs=1)
    X = np.random.randn(400, 6)
    c = Client(X, 6, cfg)
    e0 = c.epsilon()
    c.fit()
    e1 = c.epsilon()
    c.fit()
    e2 = c.epsilon()
    assert e0 <= e1 < e2  # budget keeps growing; it is not reset each round


def test_no_dp_means_infinite_epsilon():
    c = Client(np.random.randn(100, 6), 6, FLConfig(noise_multiplier=0.0))
    assert c.epsilon() == float("inf")
