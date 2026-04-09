import torch
import torch.nn as nn
import torch.nn.functional as F 


class SingleHeadAttention(nn.Module):
    def __init__(self, dim: int , bias: bool = False, dropout: float = 0.0):
        super().__init__()
        
        

        self.dim = dim
        self.scale = dim ** -0.5


        
        self.qkv = nn.Linear(dim, dim*3, bias = bias)
        self.out = nn.Linear(dim, dim, bias = bias)

        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x, mask = None, kv_cache = None):


        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)

        if kv_cache is not None:
            if kv_cache.k is not None:
                k = torch.cat([kv_cache.k, k], dim=1)
                v = torch.cat([kv_cache.v, v], dim=1)
            kv_cache.k, kv_cache.v = k, v   

        attn = ( q @ k.transpose(-2, -1)) * self.scale
        

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)

        out = attn @ v

        return self.out(out)


