from __future__ import annotations
 
import torch
from torch import Tensor
 
__all__ = [
    "causal_mask",
    "padding_mask_to_attn_bias",
    "combine_masks",
    "make_local_window_mask",
    "make_block_diagonal_mask",
    "NEG_INF",
]

NEG_INF: float = -1e9

def causal_mask( seq_len : int,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32
                 ) -> Tensor:
    
    if seq_len <= 0:
        raise ValueError(
            f"`seq_len` must be a positive integer, got {seq_len!r}."
        )

    upper_bool: Tensor = torch.ones(
        seq_len, seq_len, device=device, dtype=torch.bool
    ).triu(diagonal = 1)

    bias: Tensor = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    bias.masked_fill_(upper_bool, NEG_INF)

    return bias.unsqueeze(0).unsqueeze(0)
    

def padding_mask_to_attn_bias():
    pass

def combine_masks():
    pass

def make_local_window_mask():
    pass

def make_block_diagonal_mask():
    pass




