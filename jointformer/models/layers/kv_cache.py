import torch

from torch import nn
from typing import Tuple

class KVCache(nn.Module):

    def __init__(self, batch_size: int, max_seq_len: int, kv_head_dim: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_head_dim = kv_head_dim
        self.register_buffer("k_cache", torch.zeros((batch_size, max_seq_len, kv_head_dim)))
        self.register_buffer("v_cache", torch.zeros((batch_size, max_seq_len, kv_head_dim)))
        self.current_length = 0
        
        
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[:, :self.current_length, :], self.v_cache[:, :self.current_length, :]      
    

    def update_kv(self, kx: torch.Tensor, vx: torch.Tensor) -> None:
        assert kx.shape == vx.shape, "Key and value projections differ in size!"
        assert kx.shape[0] == self.batch_size and kx.shape[2] == self.kv_head_dim, f"Input Tensors have wrong format, first and last dimensions must match! \nShape: {(kx.shape[0], '?', kx.shape[2])}. \nKV-Cache Shape: {(self.batch_size, '?', self.kv_head_dim)}"
        in_seq_len = kx.shape[1]
        new_len = self.current_length + in_seq_len
        assert new_len <= self.max_seq_len, "KV-Cache overflow!"
        self.k_cache[:, self.current_length:new_len, :] = kx
        self.v_cache[:, self.current_length:new_len, :] = vx
        return
    