import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 16)
        
        # Latent space (mu and logvar)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, 24)
        self.fc5 = nn.Linear(24, input_dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x), 0.2)
        h2 = F.leaky_relu(self.fc2(h1), 0.2)
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z), 0.2)
        h4 = F.leaky_relu(self.fc4(h3), 0.2)
        return self.fc5(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss (MSE) + KL Divergence
    """
    # MSE Reconstruction Loss
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # KLD weight can be tuned, we use 0.1 for stability in AD
    return (MSE + 0.1 * KLD) / x.size(0)

def get_model(input_dim):
    return VariationalAutoencoder(input_dim=input_dim)

def train(model, trainloader, optimizer, criterion, device, epochs=1, prox_mu=None, global_model=None):
    """Trains the model for one or more epochs (with optional FedProx proximal term)."""
    model.train()
    model.to(device)
    
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # VAE Loss
        from .model import vae_loss_function
        loss = vae_loss_function(recon_batch, target, mu, logvar)
        
        # FedProx Proximal Term
        if prox_mu is not None and global_model is not None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
                proximal_term += ((local_weights - global_weights) ** 2).sum()
            loss += (prox_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(trainloader)
    return avg_loss

def test(model, testloader, criterion, device):
    """Evaluates the model and computes reconstruction errors."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    all_losses = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            
            # Anomaly score is typically reconstruction loss for VAE
            loss_per_sample = torch.mean((recon_batch - data) ** 2, dim=1)
            
            all_losses.extend(loss_per_sample.cpu().numpy())
            all_targets.extend(target.numpy()) # target is actual true label (anomaly or not)
            
            from .model import vae_loss_function
            batch_loss = vae_loss_function(recon_batch, data, mu, logvar)
            total_loss += batch_loss.item()
            
    avg_loss = total_loss / len(testloader)
    return avg_loss, all_losses, all_targets
