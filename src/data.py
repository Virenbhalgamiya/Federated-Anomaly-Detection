import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ParticleDetectorDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

def download_and_preprocess_magic_dataset():
    """
    Downloads the real MAGIC Gamma Telescope dataset from OpenML.
    This is an astrophysics dataset used to distinguish gamma rays (signal) 
    from hadronic showers (background/normal).
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml
    
    print("  [Data] Downloading true physics dataset (MAGIC Gamma Telescope)...")
    try:
        # data_id=1120 is the MAGIC Gamma Telescope dataset on OpenML
        dataset = fetch_openml(data_id=1120, parser='auto', as_frame=True)
        df = dataset.frame
        target_name = dataset.target_names[0] if isinstance(dataset.target_names, list) else dataset.target_names
    except Exception as e:
        print(f"  [Data] Failed to download from OpenML: {e}")
        # Fallback to random data if openml is down
        print("  [Data] Falling back to synthetic simulation...")
        return generate_simulated_physics_data()
        
    # Class 'h' (hadron) is the background -> mapped to class 1 (Anomaly here for test)
    # Class 'g' (gamma) is the signal -> mapped to class 0 (Normal)
    df['label'] = dataset.target.apply(lambda x: 1 if x == 'h' else 0)
    
    X = df.drop([target_name, 'label'], axis=1, errors='ignore').values
    y = df['label'].values
    
    # Shuffle the dataset
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    print(f"  [Data] Loaded {len(X)} samples with {X.shape[1]} features.")
    print(f"  [Data] Normal samples: {np.sum(y == 0)} | Anomalous samples: {np.sum(y == 1)}")
    
    return X, y

def get_federated_dataloaders(n_clients=3, batch_size=64, validation_split=0.2):
    """
    Downloads, scales, and splits real physics data among federated clients 
    and a centralized validation set.
    """
    X, y = download_and_preprocess_magic_dataset()
    
    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split off a centralized test set for final server-side evaluation or centralized testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)

    # Note: Unsupervised anomaly detection Autoencoder only trains on normal data.
    # The clients should ideally only see normal data representing standard operating conditions.
    # But for federated settings, clients just train on what they have. 
    # We will filter train data to ONLY normal samples for training the AE.
    normal_mask = (y_train_val == 0)
    X_train_normal = X_train_val[normal_mask]

    # Partition training data among clients
    partitions = np.array_split(X_train_normal, n_clients)
    
    trainloaders = []
    
    for partition in partitions:
        # AE reconstructs input, so target is input itself
        dataset = ParticleDetectorDataset(partition, partition)
        trainloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    
    # Server / Test dataset (contains both normal and anomalies to evaluate ROC AUC)
    test_dataset = ParticleDetectorDataset(X_test, y_test)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader, X_train_normal.shape[1]
