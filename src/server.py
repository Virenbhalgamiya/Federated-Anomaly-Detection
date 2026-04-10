import torch
import torch.nn as nn
from collections import OrderedDict
from .model import get_model, test
import os
import copy

def evaluate_global_model(model, testloader, input_dim):
    """
    Evaluates the model and returns MSE loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    loss, _, _ = test(model, testloader, criterion, device)
    return loss

def simulate_fedavg(models, trainloaders, testloader, input_dim, num_rounds=5, local_epochs=1):
    """
    Simulates Federated Averaging (FedAvg) locally without Ray.
    """
    from .client import AutoencoderClient
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize global model
    global_model = get_model(input_dim)
    global_model.to(device)
    
    best_loss = float('inf')
    loss_history = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num}/{num_rounds} ---")
        
        # 1. Distribute global weights to clients
        global_weights = global_model.state_dict()
        
        client_updates = []
        epsilons = []
        total_samples = 0
        
        for idx, trainloader in enumerate(trainloaders):
            print(f"  [Server] Dispatching global weights to Client {idx+1}/{len(trainloaders)}...")
            
            # Create client
            client = AutoencoderClient(input_dim, trainloader, device)
            
            # Send global weights
            parameters = [val.cpu().numpy() for _, val in global_weights.items()]
            client.set_parameters(parameters)
            
            # Train locally
            config = {"lr": 0.001, "local_epochs": local_epochs}
            updated_params, num_samples, metrics = client.fit(parameters, config)
            
            print(f"  [Client {idx+1}] Trained on {num_samples} samples. VAE Loss: {metrics['loss']:.6f} | Ɛ spent: {metrics['epsilon']:.4f}")
            
            # Save client updates
            client_updates.append((updated_params, num_samples))
            epsilons.append(metrics)
            total_samples += num_samples
            
        # 2. Server Aggregation (FedAvg)
        print(f"  [Server] Aggregating weights from {len(client_updates)} clients via FedAvg...")
        new_global_weights = OrderedDict()
        for idx, (updated_params, num_samples) in enumerate(client_updates):
            weight = num_samples / total_samples
            
            # Combine params
            params_dict = zip(global_model.state_dict().keys(), updated_params)
            for k, v in params_dict:
                if k not in new_global_weights:
                    new_global_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
                new_global_weights[k] += torch.tensor(v, dtype=torch.float32) * weight
                
        # 3. Update global model
        global_model.load_state_dict(new_global_weights)
        
        # 4. Evaluate centralized
        val_loss = evaluate_global_model(global_model, testloader, input_dim)
        avg_epsilon = sum([metrics['epsilon'] for metrics in epsilons]) / len(epsilons)
        
        print(f"> Round {round_num} Validation Loss: {val_loss:.6f} | Privacy Cost (Epsilon): Ɛ = {avg_epsilon:.4f}")
        loss_history.append((round_num, val_loss))
        
        # 5. Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(global_model.state_dict(), "best_model.pth")
            with open("best_model_loss.txt", "w") as f:
                f.write(str(val_loss))
                
    return loss_history

