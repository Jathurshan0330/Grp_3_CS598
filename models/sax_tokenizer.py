import torch
from torch import nn
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np



# SAX Tokenizer
# 'SAX tokenizer'
class SAX_Tokenizer(nn.Module):
    def __init__(self,
                 in_channels=1,
                 n_tokens = 32, # number of tokens or bins
                #  emb_size = 256, # embedding size
                 window_size = 125//10, # window size for SAX (0.5s window for SHHS) 0.1s is better 125//10
                 ) :
        super().__init__()
        
        self.in_channels = in_channels
        self.n_tokens = n_tokens
        # self.emb_size = emb_size
        self.window_size = window_size
        # self.embedding = nn.Embedding(n_tokens, emb_size)
        
        # create a lookup table that contains the breakpoints that divide a Gaussian distribution in an arbitrary number (from 3 to 32) of equiprobable regions
        # the area under the curve of a Gaussian distribution is divided into equiprobable regions (1/n_tokens)
        
    def gaussian_breakpoints(self,n_tokens: int):
        """
        Calculate the breakpoints that divide a Gaussian distribution
        into equiprobable regions.

        Parameters:
        n_regions (int): Number of regions to divide the distribution into.
                        There is no explicit upper or lower limit.

        Returns:
        np.ndarray: Array of breakpoints (z-scores).
        """
        if n_tokens < 1:
            raise ValueError("Number of regions must be at least 1.")

        # Calculate the breakpoints for equiprobable regions of a standard normal distribution
        probabilities = np.linspace(0, 1, n_tokens + 1)[1:-1]  # Exclude 0 and 1
        breakpoints = stats.norm.ppf(probabilities)

        return breakpoints
        
        
        
    def get_PAA(self, x, is_plot=False):
        # Function to obtain the Piecewise Aggregate Approximation (PAA) of the signal
        # 1) Normalize the signal to have zero mean and unit variance
        # 2) Use non-overlapping windows and obtain the mean of each window (PAA)
        #x: batch, 1, time
        
        # normalize each signal in the batch to have zero mean and unit variance
        x_normalized = (x - x.mean(dim=2).unsqueeze(2))/(x.std(dim=2).unsqueeze(2) + 1e-8)

        
        # Use non-overlapping windows and obtain the mean of each window
        x_normalized = x_normalized.squeeze(1)
        x = x_normalized.unfold(1, self.window_size, self.window_size).mean(dim=2)
        
        if is_plot:
            t_token = torch.linspace(0, 30, x.shape[1])
            t = torch.linspace(0, 30, x_normalized.shape[1])
            plt.figure(figsize=(20,5))
            plt.plot(t, x_normalized[0].squeeze(0).numpy(), label='original signal')
            plt.plot(t_token, x[0].squeeze(0).numpy(), label='PAA',linewidth=2)
            plt.legend()
            plt.show()
        return x
    
    def get_map_dict(self):
        # obtain dict to map tokens to center of the buckets
        break_points = self.gaussian_breakpoints(self.n_tokens)
        map_dict = {i+1: (break_points[i]+break_points[i+1])/2 for i in range(len(break_points)-1)}
        map_dict[0] = break_points[0]-(break_points[1]-break_points[0])
        map_dict[self.n_tokens-1] = break_points[-1]+(break_points[-1]-break_points[-2])
        
        return map_dict
    
    def discretize(self, x):
        # Discretize the PAA signal using the breakpoints
        #x: batch, time
        break_points = self.gaussian_breakpoints(self.n_tokens)
        x = torch.bucketize(x, torch.tensor(break_points, device=x.device))
        
        return x
    
    def forward(self, x):
        x = self.get_PAA(x)
        x = self.discretize(x)
        # x = self.embedding(x)
        return x # (batch, time//win_length, emb_size)