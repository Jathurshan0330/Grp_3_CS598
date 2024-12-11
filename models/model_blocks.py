
# from ast import Module
import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        # print(x.shape,self.pe[:, : x.size(1)].shape)
        # print(x.device,self.pe.device)
        x = x + self.pe[:, : x.size(1)]#.to(x.device)

        return self.dropout(x)
    
    
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_size, emb_size*4),
            nn.GELU(),
            nn.Linear(emb_size*4, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out