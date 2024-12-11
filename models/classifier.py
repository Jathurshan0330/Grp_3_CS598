
import torch
import torch.nn as nn

import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
from models.model_blocks import  ClassificationHead

from models.model_blocks import PositionalEncoding


class LAT_Classifier(nn.Module):
    def __init__(
        self, 
        emb_size = 256,
        num_heads = 8,
        depth = 4,
        max_seq_len = 1024,
        n_classes = 5):            
        super().__init__()
    
    

        # self.positional_encoding = PositionalEncoding(emb_size, max_len=max_seq_len)
        
        self.LAT = LinearAttentionTransformer(
            dim = emb_size,
            heads = num_heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
            )    

        self.classification_head = ClassificationHead(emb_size, n_classes = n_classes)
        
    def forward(self, x):
        # x = self.positional_encoding(x)
        x = self.LAT(x)
        x = x.mean(dim=1)
        x = self.classification_head(x)
        return x
    
    def masked_prediction(self, x):
        # x = self.positional_encoding(x)
        x = self.LAT(x)
        x = self.classification_head(x)
        return x
    
