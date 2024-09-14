import math
import torch
import torch.nn as nn

from einops import einsum, rearrange
from typing import Tuple

from jointformer.models.layers.rotary import RotaryPositionalEmbedding
from jointformer.models.layers.kv_cache import KVCache

class GroupedQueryAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_q_heads: int, group_size:int, bias: bool, dropout: float, max_seq_len: int, batch_size: int):
        super().__init__()
        self.training_running = False
        self.embedding_dim = embedding_dim
        self.num_q_heads = num_q_heads
        self.group_size = group_size
        self.bias = bias
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.q_head_dim = embedding_dim // num_q_heads
        self.kv_head_dim = embedding_dim // group_size
        self.num_kv_heads = num_q_heads // group_size
        assert num_q_heads % group_size == 0, f"num_heads % group_size == 0 must hold! num_heads: {num_q_heads}, group_size: {group_size}" 
        self.relative_embedding = RotaryPositionalEmbedding(self.q_head_dim)
        self.kv_cache = KVCache(max_seq_len=self.max_seq_len, batch_size=self.batch_size, kv_head_dim=self.kv_head_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.k_proj = nn.Linear(self.embedding_dim, self.kv_head_dim, bias=False)
        self.v_proj = nn.Linear(self.embedding_dim, self.kv_head_dim, bias=False)
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)


    def update_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.kv_cache = KVCache(max_seq_len=self.max_seq_len, batch_size=self.batch_size, kv_head_dim=self.kv_head_dim)
        
    
    def update_training_mode(self, mode: bool) -> None:
        # self.train() is set to False when evalutaing mid-training, which is why self.training_running is used here
        self.training_running = mode
        
        
    def forward_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        return q, k, v
        
    def handle_caching(self, x: torch.Tensor, in_seq_len: int):
        if self.training_running:
            return self.forward_qkv(x)
        

        
        # Else, in_seq_len == self.kv_cache.current_length + 1
        # Chop off topmost row of x
        # Project it with to k & v
        # Update cache
        
        new_seq_entries = x[:, self.kv_cache.current_length:, :]
        kx = self.k_proj.forward(new_seq_entries)
        vx = self.v_proj.forward(new_seq_entries)
        self.kv_cache.update_kv(kx=kx, vx=vx)
        k, v = self.kv_cache.get_kv()
        return q, k, v
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs grouped query attention.
        Code inspired by [https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py]

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            attn_mask (torch.Tensor): Placeholder for future masking implementation
            is_causal (bool): Placeholder for future masking implementation
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        
        in_batch_size, seq_len, _ = x.shape

        # Saving computations by caching, not caching when in training mode
        q, k, v = self.handle_caching(x, seq_len)

        # Rearranging linear projection to fit attention calculation with multiple heads & swapping seq_len with num_heads for more efficient computation
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_q_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        
        # Introducing group logic to query matrix
        q = rearrange(q, "b (h g) n d -> b g h n d", g=self.group_size)

        # Actual attention score calculation
        attn_tmp = einsum(q, k, "b g h n d, b h s d -> b g h n s")
        scaled_attn_tmp = attn_tmp / math.sqrt(self.q_head_dim)
        scores = torch.softmax(scaled_attn_tmp, dim=-1)
        scores = torch.dropout(scores, self.dropout, train=True)
              
        # Weighing value matrix with calculated attention scores & converting dimensions back to original format
        val_scores = einsum(scores, v, "b g h n s, b h s d -> b g h n d")
        val_scores = rearrange(val_scores, "b g h n d -> b n (h g) d")
        
        # Concatenating heads for multiplication with projection matrix
        concat_heads = rearrange(val_scores, 'b n h d -> b n (h d)')
        assert concat_heads.shape == (in_batch_size, seq_len, self.embedding_dim), f"Expected concat_head's shape to be {(in_batch_size, seq_len, self.embedding_dim)}, but was {concat_heads.shape}"
        
        # Projecting back into original embedding dimension for the next layer
        y = self.out(concat_heads)
        return y
    
    
if __name__ == "__main__":
    emb_dim = 512
    num_q_heads = 8
    group_size = 4
    in_batch_size = 1
    seq_len = 256
    
    gqa = GroupedQueryAttention(emb_dim, num_q_heads, group_size, False, 0, 0)
    ex1 = torch.ones((in_batch_size, seq_len, emb_dim))
    att = gqa.forward(ex1, None, None)
    