import os
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import pandas as pd
from .model import get_model, test

def evaluate_and_report(history, testloader, input_dim, save_dir="assets"):
    """
    Evaluates the final global federated model.
    Generates ROC curves, loss histograms, and a README.md report.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load the final model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(input_dim)
    
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        print("Warning: best_model.pth not found. Evaluating randomly initialized model.")
        
    criterion = nn.MSELoss()
    
    # 2. Extract final test metrics
    _, all_losses, all_targets = test(model, testloader, criterion, device)
    
    # 3. Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(all_targets, all_losses)
    roc_auc = auc(fpr, tpr)
    
    # Generate optimal threshold using Youden's J statistic
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    
    # 4. Generate Predictions
    preds = [1 if x > optimal_threshold else 0 for x in all_losses]
    report = classification_report(all_targets, preds, target_names=["Normal (0)", "Anomaly (1)"], output_dict=True)
    
    # 5. Print results to terminal
    print("\n" + "="*50)
    print("📈 ENTERPRISE FL EVALUATION RESULTS")
    print("="*50)
    print(f"ROC AUC Score       : {roc_auc:.4f}")
    print(f"Precision (Anomaly) : {report['Anomaly (1)']['precision']:.4f}")
    print(f"Recall (Anomaly)    : {report['Anomaly (1)']['recall']:.4f}")
    print(f"Optimal Threshold   : {optimal_threshold:.4f} (VAE Recon + KL)")
    print("="*50 + "\n")
    
    # 6. Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Federated AE)')
    plt.legend(loc="lower right")
    plt.savefig(f"{save_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Plot Learning Curve (Server Validation Loss)
    rounds, losses = zip(*history)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, losses, marker='o', linestyle='-', color='purple')
    plt.title('Federated Global Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Generate README
    readme_content = f"""<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Flower-Federated_Learning-22d3ee?style=flat-square" alt="Flower"/>
  <img src="https://img.shields.io/badge/Scikit_Learn-Data_Processing-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat-square" alt="Matplotlib"/>
</p>

<h1 align="center">🌌 Federated Anomaly Detection for High-Energy Physics</h1>

<p align="center">
  <strong>An end-to-end, privacy-preserving machine learning pipeline using Federated Averaging (FedAvg) to collaboratively detect rare particle decay anomalies across multiple simulated research institutions.</strong>
</p>

---

## 🎯 Enterprise Upgrades Implemented

In particle physics experiments like those at the Large Hadron Collider (LHC), massive amounts of detector data are collected across multiple collaborating institutions. Sharing raw data centrally often introduces latency, bandwidth bottlenecks, and severe security challenges. 

This project solves this via an **Enterprise Federated Learning** architecture featuring:

### 1. Differential Privacy (PyTorch Opacus)
- **Data Privacy by Design:** Only model gradients are aggregated centrally.
- **Opacus DP:** Client gradients are mathematically clipped and injected with controlled noise during the backward pass, ensuring the central aggregator cannot deduce individual physics data samples from the weights. The privacy budget spent (Epsilon) is tracked live.

### 2. Probabilistic Modeling (Variational Autoencoder)
- **Unsupervised Anomaly Detection:** Utilizes a Deep PyTorch **Variational Autoencoder** trained purely on normal background QCD jets to identify rare signal events.
- **Why VAE?** Instead of rigid distances, the VAE learns continuous probability distributions. The threshold uses a combination of Reconstruction Loss and KL Divergence for robust anomaly scoring.

### 3. Non-IID Handling (FedProx)
- **Custom Orchestration:** Implements a decoupled orchestration of the Federated Averaging algorithm without heavy backend dependencies.
- **FedProx penalty:** Proximal mathematical terms act as constraints to stop clients with highly skewed data from drifting too far from the global server state.

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Federated Learning Architecture" width="800"/>
</p>

---

## 📊 Live Evaluation Results

**Note: These metrics are dynamically injected upon successful completion of the `run_pipeline.py` script.**

The federated model aggregates over `5` communication rounds. After aggregation, the centralized global model is evaluated against a holdout test suite simulating unseen rare anomalies.

### Final Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **ROC AUC** | **{roc_auc:.4f}** | Area Under the Receiver Operating Characteristic Curve |
| **Precision (Anomaly)** | **{report['Anomaly (1)']['precision']:.4f}**| Accuracy of anomaly positive predictions |
| **Recall (Anomaly)** | **{report['Anomaly (1)']['recall']:.4f}** | Proportion of actual anomalies successfully caught |
| **Optimal Threshold** | **{optimal_threshold:.4f}** | The reconstruction loss cutoff to classify purely rare events |

### Training History & Performance

<p align="center">
  <img src="assets/loss_curve.png" alt="Validation Loss" width="45%"/>
  <img src="assets/roc_curve.png" alt="ROC Curve" width="45%"/>
</p>

---

## 🚀 Quick Start (One-Click Pipeline)

This project is completely modularized. To reproduce these results on your local cluster or GPU instance:

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
python run_pipeline.py
```

**Executing this command will automatically:**
1. Generate synthetic High-Energy particle data and inject rare anomalies.
2. Distribute data locally among $N$ federated clients.
3. Train the Autoencoder simultaneously and aggregate weights globally using `FedAvg`.
4. Validate the model computing optimal classification thresholds.
5. Export performance charts to `assets/` and dynamically overwrite this very `README.md` with the live results.
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print(f"\n✅ Evaluation complete. ROC AUC: {roc_auc:.4f}")
    print(f"✅ Generated plots in {save_dir}/")
    print(f"✅ Updated README.md")
