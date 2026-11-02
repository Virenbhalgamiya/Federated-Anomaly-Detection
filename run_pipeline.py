"""Quick demo: one federated run (FedAvg + DP-SGD) on MAGIC, evaluated on a held-out test set.

For the full study (baselines, privacy sweep, FedProx, several seeds) use run_experiments.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.client import FLConfig
from src.data import load_magic, make_splits, partition
from src.server import run_federated


def main(seed=0, noise_multiplier=1.0):
    X, y = load_magic()
    splits = make_splits(X, y, seed)
    cfg = FLConfig(noise_multiplier=noise_multiplier)
    clients = partition(splits.X_train_normal, cfg.n_clients, "iid", seed)
    print(f"{len(clients)} clients, sizes {[len(c) for c in clients]}, "
          f"DP noise multiplier {cfg.noise_multiplier}, {cfg.rounds} rounds")
    test, history = run_federated(clients, splits, cfg, seed)
    for h in history[:: max(1, len(history) // 5)]:
        print(f"  round {h['round']:2d}  val AUC {h['val_auc']:.4f}  epsilon so far {h['epsilon']:.2f}")
    print(f"\nTest ROC AUC {test['auc']:.4f} | AUPRC {test['auprc']:.4f} | F1 {test['f1']:.4f} "
          f"| precision {test['precision']:.4f} | recall {test['recall']:.4f}")
    print(f"Total privacy budget epsilon = {test['epsilon']:.2f} at delta = {cfg.delta:g}")


if __name__ == "__main__":
    main()
