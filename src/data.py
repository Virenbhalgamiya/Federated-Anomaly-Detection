"""Data loading, splitting and client partitioning for the MAGIC Gamma Telescope dataset.

MAGIC (OpenML id 1120) is real Cherenkov-telescope data: gamma showers ('g', signal) versus
hadronic showers ('h', background). We treat gamma as "normal" and hadron as the "anomaly".
Hadrons are ~35% of the data, so this is a binary-separation benchmark for an anomaly detector,
not a rare-event search.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Splits:
    X_train_normal: np.ndarray  # normal-only training data (what the clients see)
    X_val: np.ndarray           # used for model selection and the decision threshold
    y_val: np.ndarray
    X_test: np.ndarray          # touched once, at the end
    y_test: np.ndarray
    input_dim: int


def load_magic():
    """Download MAGIC from OpenML. Fails loudly: there is no silent synthetic fallback."""
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml(data_id=1120, as_frame=True, parser="auto")
    df = dataset.frame
    target_name = dataset.target_names[0] if isinstance(dataset.target_names, list) else dataset.target_names
    y = (dataset.target.astype(str) == "h").astype(int).values
    X = df.drop(columns=[target_name]).values.astype(np.float64)
    return X, y


def make_splits(X, y, seed, val_size=0.2, test_size=0.2):
    """Stratified train/val/test split. The scaler is fit on normal training data only."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size / (1 - test_size), stratify=y_tr, random_state=seed)
    X_norm = X_tr[y_tr == 0]
    scaler = StandardScaler().fit(X_norm)
    return Splits(
        X_train_normal=scaler.transform(X_norm),
        X_val=scaler.transform(X_va), y_val=y_va,
        X_test=scaler.transform(X_te), y_test=y_te,
        input_dim=X.shape[1],
    )


def partition(X, n_clients, mode, seed):
    """Split normal training data across clients.

    iid:     random equal split.
    noniid:  feature-skew. Samples are ordered along the first principal component and cut into
             contiguous slices, so each client sees a different region of the normal distribution.
    """
    rng = np.random.default_rng(seed)
    if mode == "iid":
        chunks = np.array_split(rng.permutation(len(X)), n_clients)
    elif mode == "noniid":
        Xc = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        chunks = np.array_split(np.argsort(Xc @ vt[0]), n_clients)
    else:
        raise ValueError(f"unknown partition mode: {mode}")
    return [X[c] for c in chunks]
