import torch
import torch.nn as nn

from jointformer.models.layers.layer_norm import RMSNorm
from jointformer.models.layers.gqa import GroupedQueryAttention
from jointformer.models.layers.mlp import FeedForward


class TransformerLayer(nn.Module):

    def __init__(self, embedding_dim, embedding_hidden_dim, bias, attention_dropout, feed_forward_dropout, num_heads, group_size, max_seq_len, layer_norm_eps, batch_size):
        super().__init__()
        self.attention_layer = GroupedQueryAttention(embedding_dim, num_heads, group_size, bias, attention_dropout, max_seq_len, batch_size)
        self.feed_forward = FeedForward(embedding_dim, embedding_hidden_dim, bias, feed_forward_dropout)
        self.attention_layer_normalization = RMSNorm(embedding_dim, layer_norm_eps)
        self.feed_forward_normalization = RMSNorm(embedding_dim, layer_norm_eps)
        
    def forward(self, x: torch.Tensor, is_causal: bool, mask: torch.Tensor = None, next_token_only: bool = False) -> torch.Tensor:
        x = x + self.attention_layer(x=self.attention_layer_normalization(x), next_token_only=next_token_only)
        x = x + self.feed_forward(self.feed_forward_normalization(x))
        return x

    def update_batch_size(self, batch_size: int) -> None:
        self.attention_layer.update_batch_size(batch_size)
        
        