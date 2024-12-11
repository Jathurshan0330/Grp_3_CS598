import torch
import torch.nn as nn
import math
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
from models.model_blocks import PositionalEncoding
from einops import rearrange
from timm.models.layers import trunc_normal_

from models.model_utils import l2norm





    
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 emb_size = 64,
                 num_heads = 8,
                 depth = 4,
                 max_seq_len = 1024,   
                 ):
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(emb_size,max_len=max_seq_len)
        
        
        self.transformer = LinearAttentionTransformer(
            dim = emb_size,
            heads = num_heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )  
        
    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer(x)
        # x = x.mean(dim = 1)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, emb_size, code_book_size):
        super().__init__()
        
        self.code_book_size = code_book_size
        self.emb_size = emb_size
        
        self.code_book = nn.Embedding(num_embeddings=code_book_size, embedding_dim=emb_size)
        self.code_book.weight.data.uniform_(-1/code_book_size, 1/code_book_size)
    
    def forward(self, x):
        x_flattened = x.reshape(-1, self.emb_size) # x.view(-1, self.emb_size)  # (batch_size * sequence_length, emb_size)
        # Compute squared distances using matrix operations
        code_book_weights = self.code_book.weight  # (code_book_size, emb_size)
        distances = (
            torch.sum(x_flattened**2, dim=1, keepdim=True)  # (batch_size * sequence_length, 1)
            - 2 * torch.matmul(x_flattened, code_book_weights.T)  # (batch_size * sequence_length, code_book_size)
            + torch.sum(code_book_weights**2, dim=1)  # (code_book_size,)
        )
        indices = torch.argmin(distances, dim=1).reshape(x.size(0), x.size(1))  # (batch_size, sequence_length)
        quant_out_flattened = self.code_book(indices.reshape(-1))  # (batch_size * sequence_length, emb_size)
        quant_out = quant_out_flattened.reshape(x.size())  # (batch_size, sequence_length, emb_size)
        
        
        return quant_out, indices
    
    # def forward(self, x):
    #     x_flattened = x.reshape(-1, self.emb_size) # x.view(-1, self.emb_size)  # (batch_size * sequence_length, emb_size)
    #     code_book_expanded = self.code_book.weight.unsqueeze(0)  # (1, code_book_size, emb_size)
    #     distances = torch.cdist(x_flattened.unsqueeze(1), code_book_expanded) 
    #     indices = torch.argmin(distances, dim=-1).view(x.size(0), x.size(1))
        
    #     quant_out_flattened = self.code_book(indices.reshape(-1))#self.code_book(indices.view(-1))
    #     quant_out = quant_out_flattened.view(x.size()) 
        
    #     return quant_out, indices
        


 
                 
                 
        


class Freq_VQVAE(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 n_freq = 320,
                 emb_size = 64,
                 code_book_size = 4096,
                 beta = 0.2):
        super().__init__()
        
        # # freq embedding
        # self.freq_embedding = nn.Embedding(n_freq, 1)
        
        # Encoder        
        self.encoder = nn.Linear(n_freq, emb_size)
        self.trans_encoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = 12)#4)
        
        
        # Vector quantization bottleneck
        self.quantizer = VectorQuantizer(emb_size, code_book_size)
        self.beta = beta

        # Decoder
        self.trans_decoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = 3)#4)
        self.decoder = nn.Linear(emb_size, n_freq)
            
    def forward(self, x):
        #add frequency embedding
        # freq = torch.arange(x.size(1)).to(x.device)
        # freq = self.freq_embedding(freq)
        # x = x + freq
        
        x = x.permute(0, 2, 1)
        
        
        
        x = self.encoder(x)
        # print('Shape after encoder:', x.shape)
        x = self.trans_encoder(x)
        # print('Shape after transformer:', x.shape)
        
        quant_in = l2norm(x)
        
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        # print('Shape after quantization:', quant_out.shape)
        
        # Straight through estimator
        quant_out = quant_in + (quant_out - quant_in).detach()
        
        # Decoder
        x = self.trans_decoder(quant_out)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x, indices,quant_out,quant_in
    
    def tokenize(self, x):
        # #add frequency embedding
        # freq = torch.arange(x.size(1)).to(x.device)
        # freq = self.freq_embedding(freq)
        # x = x + freq
        
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.trans_encoder(x)
        
        quant_in = l2norm(x)
        
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        
        return quant_out, indices
    
    
    def vec_quantizer_loss(self, quant_in, quant_out):
        # compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_in) ** 2)
        code_book_loss = torch.mean((quant_out - quant_in.detach()) ** 2)
        
        loss = code_book_loss + self.beta * commitment_loss
        
        return loss, code_book_loss, commitment_loss




class TemporalConv1D(nn.Module):
    def __init__(self,
                in_channels = 1,
                emb_size = 64,
                kernel_size = 64, # For SHHS as sampling frequency is 125Hz
                smallest_kernel_divider = 4, # For SHHS as sampling frequency is 125Hz
                stride = 64):
        super().__init__()
        
   
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, 
                    kernel_size = kernel_size//smallest_kernel_divider, stride = stride//smallest_kernel_divider),
            nn.GELU(),
            nn.GroupNorm(emb_size//4, emb_size))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, 
                    kernel_size = smallest_kernel_divider, stride = smallest_kernel_divider),
            nn.GELU(),
            nn.GroupNorm(emb_size//4, emb_size))
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = rearrange(x, 'B C T -> B T C')
        return x   

class TemporalConv1DDecoder(nn.Module):
    def __init__(self,
                 emb_size=256,
                 out_channels=1,
                 kernel_size=64,
                 smallest_kernel_divider=4,
                 stride=64):
        super().__init__()

        # Inverse of conv2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(emb_size, emb_size,
                               kernel_size=smallest_kernel_divider, stride=smallest_kernel_divider),
            nn.GELU(),
            nn.GroupNorm(emb_size // 4, emb_size))

        # Inverse of conv1
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(emb_size, out_channels,
                               kernel_size=kernel_size // smallest_kernel_divider,
                               stride=stride // smallest_kernel_divider),
            nn.GELU())

    def forward(self, x):
        x = rearrange(x, 'B T C -> B C T')  # Rearrange to (B, C, T) for deconvolutions
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x

class Temporal_VQVAE(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 kernel_size = 1024,
                 stride = 64,
                 smallest_kernel_divider = 4,
                 emb_size = 64,
                 code_book_size = 4096,
                 beta = 0.2):
        super().__init__()
        
        
        # Encoder
        self.encoder = TemporalConv1D(in_channels = in_channels,
                                        emb_size = emb_size,
                                        kernel_size = kernel_size,
                                        smallest_kernel_divider = smallest_kernel_divider,
                                        stride = stride)
        
        self.trans_encoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = 12)#4)
        
        
        # Vector quantization bottleneck
        self.quantizer = VectorQuantizer(emb_size, code_book_size)
        self.beta = beta

        # Decoder
        self.trans_decoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = 3)#4)
        self.decoder = TemporalConv1DDecoder(emb_size = emb_size,
                                             out_channels = in_channels,
                                            kernel_size = kernel_size,
                                            smallest_kernel_divider = smallest_kernel_divider,
                                            stride = stride)
            
    def forward(self, x):
        x = self.encoder(x)
        # print('Shape after encoder:', x.shape)
        x = self.trans_encoder(x)
        # print('Shape after transformer:', x.shape)
        
        quant_in = l2norm(x)
        
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        # print('Shape after quantization:', quant_out.shape)
        
        # Straight through estimator
        quant_out = quant_in + (quant_out - quant_in).detach()
        
        # Decoder
        x = self.trans_decoder(quant_out)
        x = self.decoder(x)
        return x, indices,quant_out,quant_in
    
    def tokenize(self, x):
        x = self.encoder(x)
        x = self.trans_encoder(x)
        
        quant_in = l2norm(x)
        
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        
        return quant_out, indices
    
    
    def vec_quantizer_loss(self, quant_in, quant_out):
        # compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_in) ** 2)
        code_book_loss = torch.mean((quant_out - quant_in.detach()) ** 2)
        
        loss = code_book_loss + self.beta * commitment_loss
        
        return loss, code_book_loss, commitment_loss
    
    

    
    
    
### Baseline reproductions

class LaBRAM_VQVAE(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 kernel_size = 1024,
                 stride = 64,
                 smallest_kernel_divider = 4,
                 emb_size = 64,
                 code_book_size = 4096,
                 beta = 0.2):
        super().__init__()
        
        
        # Encoder
        self.encoder = TemporalConv1D(in_channels = in_channels,
                                        emb_size = emb_size,
                                        kernel_size = kernel_size,
                                        smallest_kernel_divider = smallest_kernel_divider,
                                        stride = stride)
        
        self.trans_encoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = 12)#4)
        
        
        # Vector quantization bottleneck
        self.quantizer = VectorQuantizer(emb_size, code_book_size)
        self.beta = beta

        # Decoder
        self.trans_decoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = 3)#4)
        self.fourier_decoder = nn.Linear(emb_size, 99)
        
        self.angle_decoder = nn.Linear(emb_size, 99)
            
    def forward(self, x):
        x = self.encoder(x)
        # print('Shape after encoder:', x.shape)
        x = self.trans_encoder(x)
        # print('Shape after transformer:', x.shape)
        
        quant_in = l2norm(x)
        
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        # print('Shape after quantization:', quant_out.shape)
        
        # Straight through estimator
        quant_out = quant_in + (quant_out - quant_in).detach()
        
        # Decoder
        x = self.trans_decoder(quant_out)
        fft_mag = self.fourier_decoder(x)
        fft_angle = self.angle_decoder(x)
        return fft_mag,fft_angle, indices,quant_out,quant_in
    
    def tokenize(self, x):
        x = self.encoder(x)
        x = self.trans_encoder(x)
        
        quant_in = l2norm(x)
        
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        
        return quant_out, indices
    
    
    def vec_quantizer_loss(self, quant_in, quant_out):
        # compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_in) ** 2)
        code_book_loss = torch.mean((quant_out - quant_in.detach()) ** 2)
        
        loss = code_book_loss + self.beta * commitment_loss
        
        return loss, code_book_loss, commitment_loss
    


