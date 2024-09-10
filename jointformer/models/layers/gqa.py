import math
import torch
import torch.nn as nn

from rotary import RotaryPositionalEmbedding
from einops import einsum, rearrange

class GroupedQueryAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_q_heads: int, group_size:int, bias: bool, dropout: float, block_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_q_heads = num_q_heads
        self.group_size = group_size
        self.head_dim = embedding_dim // num_q_heads
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size
        self.num_kv_heads = num_q_heads // group_size
        assert num_q_heads % group_size == 0, f"num_heads % group_size == 0 must hold! num_heads: {num_q_heads}, group_size: {group_size}" 
        self.relative_embedding = RotaryPositionalEmbedding(self.head_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim // self.group_size, bias=False)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim // self.group_size, bias=False)
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, is_causal: bool) -> torch.Tensor:
        """ Performs grouped query attention.
        Code inspired by [https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py]

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            attn_mask (torch.Tensor): Placeholder for future masking implementation
            is_causal (bool): Placeholder for future masking implementation
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        
        """
        batch_size, seq_len, _ = x.shape 

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        assert q.shape == (batch_size, seq_len, self.embedding_dim), f"Expected q's shape to be {(batch_size, seq_len, self.embedding_dim)}, but was {q.shape}"
        assert k.shape == (batch_size, seq_len, self.embedding_dim // self.group_size), f"Expected k's shape to be {(batch_size, seq_len, self.embedding_dim // self.group_size)}, but was {k.shape}"

        # Rearranging linear projection to fit attention calculation with multiple heads & swapping seq_len with num_heads for more efficient computation
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_q_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        
        # Introduce group logic to query matrix
        q = rearrange(q, "b (h g) n d -> b g h n d", g=self.group_size)
        
        # Actual attention score calculation
        attn_tmp = einsum(q, k, "b g h n d, b h s d -> b g h n s")

        scaled_attn_tmp = attn_tmp / math.sqrt(self.head_dim)
        scores = torch.softmax(scaled_attn_tmp, dim=-1)
        scores = torch.dropout(scores, self.dropout, train=True)
        
        # Weigh value matrix with calculated attention scores and convert dimensions back to original format
        val_scores = einsum(scores, v, "b g h n s, b h s d -> b g h n d")
        val_scores = rearrange(val_scores, "b g h n d -> b n (h g) d")
        
        # Concatenate heads for multiplication with projection matrix
        concat_heads = rearrange(val_scores, 'b n h d -> b n (h d)')
        assert concat_heads.shape == (batch_size, seq_len, self.embedding_dim), f"Expected concat_head's shape to be {(batch_size, seq_len, self.embedding_dim)}, but was {concat_heads.shape}"
        y = self.out(concat_heads)
        return y
    
    
if __name__ == "__main__":
    emb_dim = 512
    num_q_heads = 8
    group_size = 4
    batch_size = 1
    seq_len = 256
    
    gqa = GroupedQueryAttention(emb_dim, num_q_heads, group_size, False, 0, 0)
    ex1 = torch.ones((batch_size, seq_len, emb_dim))
    att = gqa.forward(ex1, None, None)
    