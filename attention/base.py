import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    @abstractmethod
    def forward(self, q, k, v, mask = None, kv_cache = None):
        pass
