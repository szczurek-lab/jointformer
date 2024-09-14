import torch

from torch import nn
from typing import Tuple
from enum import Enum, auto

class Modes(Enum):
    AUTOREGRESSIVE = auto()
    PREFILL = auto()
    

class KVCache(nn.Module):

    def __init__(self, batch_size: int, max_seq_len: int, kv_head_dim: int) -> None:
        super().__init__()
        self.shape = (batch_size, max_seq_len, kv_head_dim)
        self.register_buffer("k_cache", torch.zeros(self.shape))
        self.register_buffer("v_cache", torch.zeros(self.shape))
        self.current_length = 0
        self.mode = Modes.PREFILL
        
        
    def __len__(self) -> int:
        return self.current_length
    
    
    def size(self, dim: int) -> int:
        return self.shape[dim]
    
         
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[:, :self.current_length, :], self.v_cache[:, :self.current_length, :]      
    
    
    def set_mode(self, mode: Modes) -> None:
        self.mode = True
    
    
    def check_format(self, kx: torch.Tensor, vx: torch.Tensor) -> None:
        if self.mode == Modes.PREFILL:
            assert self.current_length == 0, ""
            # Add stuff
            return
        assert kx.shape == vx.shape, "Key and value projections differ in size!"
        assert all(kx.size(dim) == self.size(dim) for dim in (0, 2)), f"Wrong input format! First and last dimension MUST match! Shape: {kx.shape}. KV-Cache Shape: {self.shape}."
        assert kx.size(1) == 1, f"Cache is in autoregressive mode but received more than one input! Input sequence length: {kx.size(1)}"

    
    def prefill_kv(self, kx: torch.Tensor, vx: torch.Tensor) -> None:
        self.check_format(kx, vx)

    def update_kv(self, kx: torch.Tensor, vx: torch.Tensor) -> None:
        self.check_format(kx, vx)
        return 
    
        new_len = self.current_length + in_seq_len
        assert new_len <= self.max_seq_len, "KV-Cache overflow!"
        self.k_cache[:, self.current_length:new_len, :] = kx
        self.v_cache[:, self.current_length:new_len, :] = vx
        return
    
    