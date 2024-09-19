import torch

from torch import nn
from typing import Tuple

class KVCache(nn.Module):

    def __init__(self, batch_size: int, max_seq_len: int, kv_head_dim: int) -> None:
        super().__init__()
        self.shape = (batch_size, max_seq_len, kv_head_dim)
        self.register_buffer("k_cache", torch.zeros(self.shape))
        self.register_buffer("v_cache", torch.zeros(self.shape))
        self.current_length = 0
        
        
    def __len__(self) -> int:
        return self.current_length
    
    
    def size(self, dim: int) -> int:
        return self.shape[dim]


    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[:, :self.current_length, :], self.v_cache[:, :self.current_length, :]      


    def update(self, kx: torch.Tensor, vx: torch.Tensor) -> None:
        assert kx.shape == vx.shape, "Key and value projections differ in size!"
        assert all(kx.size(dim) == self.size(dim) for dim in (0, 2)), f"Wrong input format! First and last dimension MUST match! Shape: {kx.shape}. KV-Cache Shape: {self.shape}."
        assert kx.size(1) == 1, f"Cache is autoregressive but received input of sequence length {kx.size(1)} != 1 !"
        assert self.current_length < self.size(1), "Cache overflow!"
        self.k_cache[:, self.current_length, :] = kx
        self.v_cache[:, self.current_length, :] = vx
        self.current_length += 1
        return self.get_kv()
    