import flwr as fl
import os
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import get_federated_dataloaders
from src.server import simulate_fedavg
from src.report import evaluate_and_report

def main():
    print("=" * 60)
    print("🚀 Starting Federated Anomaly Detection Pipeline")
    print("=" * 60)
    
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5
    
    print("\n[1/4] Generating simulated HD physics data...")
    trainloaders, testloader, input_dim = get_federated_dataloaders(n_clients=NUM_CLIENTS)
    print(f"✅ Created {NUM_CLIENTS} client partitions. Input dim: {input_dim}")
    
    print("\n[2/4] Setting up Federated Learning infrastructure...")
    # Skipping heavy flwr Ray requirements. Running synchronously.
    
    print("\n[3/4] Starting Federated Simulation (FedAvg)...")
    history = simulate_fedavg(
        models=None,
        trainloaders=trainloaders,
        testloader=testloader,
        input_dim=input_dim,
        num_rounds=NUM_ROUNDS,
        local_epochs=1
    )
    
    print("\n[4/4] Evaluating Global Model and Generating Reports...")
    evaluate_and_report(history, testloader, input_dim)

if __name__ == "__main__":
    main()
