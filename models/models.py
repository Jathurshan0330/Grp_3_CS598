# from ast import Module
import torch
import torch.nn as nn
import math







        

class Autoencoder(nn.Module):
    def __init__(
        self, 
        num_channels = 1,
        emb_size = 256, # For SHHS as sampling frequency is 125Hz
        patch_size = 125//5, # For SHHS as sampling frequency is 125Hz
        model_type = 'CNN', # 'CNN', 'Transformer'
        sparsity = 0.1, # sparsity parameter
        ):
        super().__init__()  
        
        if 'CNN' in model_type:
            self.encoder = nn.Sequential(
                nn.Conv1d(num_channels, emb_size//4, kernel_size=patch_size, stride=patch_size),
                nn.ReLU(),
                nn.Conv1d(emb_size//4, emb_size, kernel_size=1, stride=1),
                nn.ReLU(),
            )
            
            self.decoder = nn.Sequential(
                nn.Conv1d(emb_size, emb_size//4, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.ConvTranspose1d(emb_size//4, num_channels, kernel_size=patch_size, stride=patch_size),
            )
        
        if 'Linear' in model_type:
            self.encoder = nn.Sequential(
                nn.Linear(num_channels*patch_size, emb_size//4),
                nn.LeakyReLU(),
                nn.Linear(emb_size//4, emb_size),
                nn.Tanh(),
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(emb_size, emb_size//4),
                nn.LeakyReLU(),
                nn.Linear(emb_size//4, num_channels*patch_size),
            )
        
        self.sparsity = sparsity
        
    def forward(self, x):
        #x: (batch, channels, time)
        z = self.encoder(x)
        #x: (batch, emb_size, time//patch_size)
        
        
        
        x_reconstructed = self.decoder(z)
        #x: (batch, channels, time)
        
        return x_reconstructed, z
    
    def sparse_loss(self, z):
        # sparse_loss = self.sparsity * torch.mean(torch.abs(z)) 
        # sparse loss l1 norm
        sparse_loss = self.sparsity * torch.mean(torch.abs(z))
        
        return sparse_loss
    
    def recon_loss(self, x, x_reconstructed):
        recon_loss = nn.MSELoss()(x, x_reconstructed)
        return recon_loss

    def tokenize(self, x):
        z = self.encoder(x)
        return z.permute(0,2,1)
    
    def decode(self, z):
        x_reconstructed = self.decoder(z.permute(0,2,1))
        return x_reconstructed
        
        
        
