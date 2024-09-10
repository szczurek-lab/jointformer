import torch

from torch import nn

class KVCache(nn.Module):
    """
    Are Batch size lengths changing ?
    How to get max seq_len ?
    kv_dim = (embedding_dim * num_q_heads) // group_size 
    dtype expects k. / v.dtype obviously
    """
    def __init__(self, batch_size: int, seq_len: int, kv_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("k_cache", torch.zeros((batch_size, seq_len, kv_dim), dtype=dtype))
        self.register_buffer("v_cache", torch.zeros((batch_size, seq_len, kv_dim), dtype=dtype))

    def update_cache(self, x: torch.Tensor):
        pass
    
    def get_kv(self):
        pass
    