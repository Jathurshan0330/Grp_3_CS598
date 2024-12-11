import torch.nn.functional as F
import torch

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


def compute_fft(x):
    fft_out = torch.fft.fft(x, dim=-1)
    fft_out = torch.fft.fftshift(fft_out, dim=-1)
    # get half of the fft
    fft_out = fft_out[:, :,(fft_out.shape[-1]//2)+1:]
    fft_mag = torch.abs(fft_out)
    fft_phase = torch.angle(fft_out)
    return fft_mag, fft_phase


def augment_temporal(x,device):
    B = x.shape[0]
    T = x.shape[-1]
    # Randomly select B alphas between 0.1 and 1.5
    alpha = (torch.rand(B) * (1.5 - 0.1) + 0.1).to(device)
    
    
    # Generate random noise epsilon, same size as x
    sd = (torch.rand(B) * (0.2 - 0.001) + 0.001).to(device)
    x_aug = x.clone()
    epsilon = torch.randn_like(x_aug) * sd.unsqueeze(-1).unsqueeze(-1) # Noise epsilon
    
    x_aug = alpha.unsqueeze(-1).unsqueeze(-1) * (x_aug + epsilon)

    
    
    
    # for i in range (B):
    #     # select 0 or 1
    #     add_artifact = torch.randint(2, (1,)).item()
    #     if add_artifact:
    #             # print('Adding artifact')
    #         # Randomly select a start time index t
    #         max_t = T - int(0.8 * T)
    #         t = torch.randint(low=0, high=max_t + 1, size=(1,)).item()
    #         artifact_magnitude = torch.randn(1).item() * (1.5 - 0.5) + 0.5
    #         artifact = torch.randn_like(x[i,0, t:t+max_t]) * artifact_magnitude
    #         x_aug[i,0, t:t+max_t] += artifact
    return x_aug
