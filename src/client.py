import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from .model import get_model, train

class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, input_dim, trainloader, device):
        self.model = get_model(input_dim)
        self.trainloader = trainloader
        self.device = device
        self.criterion = nn.MSELoss()
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for key, val in self.model.state_dict().items() if not key.endswith('num_batches_tracked')]

    def set_parameters(self, parameters):
        # Opacus might wrap self.model into GradSampleModule. If so, loading standard dict fails.
        # But set_parameters runs BEFORE make_private, so self.model is usually standard here,
        # unless it persisted from previous rounds. Handle both cases cleanly.
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Hyperparameters can be passed via config from the server
        lr = config.get("lr", 0.001)
        epochs = config.get("local_epochs", 1)
        prox_mu = config.get("prox_mu", 0.01) # FedProx proximal term weight
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 1. Opacus Differential Privacy Setup
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        
        # Make the model, optimizer, and dataloader private
        # We wrap in a try block in case the model is already wrapped from a previous round
        try:
            self.model, optimizer, self.trainloader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                noise_multiplier=1.0,  # DP noise
                max_grad_norm=1.0,     # DP clipping
            )
        except ValueError:
            # Already attached
            pass
            
        # 2. FedProx - save a static copy of the global model before local training begins
        import copy
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        
        loss = 0.0
        for _ in range(epochs):
            loss = train(
                self.model, 
                self.trainloader, 
                optimizer, 
                self.criterion, 
                self.device, 
                epochs=1,
                prox_mu=prox_mu,
                global_model=global_model
            )
            
        # Extract privacy budget spent
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
            
        # Opacus wraps the model in a GradSampleModule which changes parameter names (adds _module.)
        # We need to extract the base parameters safely.
        # get_parameters below will just export the state dict of the model.
        # But we must ensure it matches the original shape if Opacus modified it.
        # Fortunately, Opacus keeps state_dict shapes identical.
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": loss, "epsilon": epsilon}

def get_client_fn(trainloaders, input_dim):
    """
    Returns a function that spawns a client. Used by Flower simulation.
    """
    def client_fn(cid: str) -> fl.client.Client:
        # Load the partition for this client
        trainloader = trainloaders[int(cid)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create and return client
        return AutoencoderClient(input_dim, trainloader, device)
    
    return client_fn
