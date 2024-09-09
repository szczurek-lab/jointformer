import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from jointformer.models.layers.rotary import RotaryPositionalEmbedding


class GroupedQueryAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, group_size:int, bias: bool, dropout: float, block_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = embedding_dim // num_heads
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size
        self.sdpa = F.scaled_dot_product_attention if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else self.scaled_dot_product_attention
        
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        self.relative_embedding = RotaryPositionalEmbedding(self.head_dim)
        assert num_heads % group_size == 0, f"num_heads % group_size == 0 must hold! num_heads: {num_heads}, group_size: {group_size}" 
        self.num_kv_heads = num_heads // group_size
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.k_proj = nn.Linear(self.embedding_dim, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embedding_dim, self.head_dim * self.num_kv_heads, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, is_causal: bool) -> torch.Tensor:
        """ Forward pass of the attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            attn_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len) and type torch.bool
            is_causal (bool): If True, the model is autoregressive and variable `mask` is ignored
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        
        """
        batch_size, seq_len, _ = x.shape 

        q = self.q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj.view(batch_size, seq_len, self.num_kv_heads, self.group_size)
        v = self.v_proj.view(batch_size, seq_len, self.num_kv_heads, self.group_size)

        q = RotaryPositionalEmbedding(self.head_dim).forward(q)
        k = RotaryPositionalEmbedding(self.group_size).forward(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) 
        
        attn_tmp = q @ k.transpose(1, 2) / math.sqrt(self.head_dim)
        attn_tmp = torch.softmax(attn_tmp, dim=-1)
        attn_tmp = self.apply_utm_mask(attn_tmp)
        attn_tmp = torch.dropout(attn_tmp, self.dropout, train=True)
        attn_score = attn_tmp @ v
        
        y = attn_score.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)        
        y = self.out(y)
        return y
    
    def apply_utm_mask(x: torch.Tensor) -> torch.Tensor:
        # TODO
        return x
        