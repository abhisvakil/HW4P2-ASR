import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

class SelfAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, self_attn_weights = self.self_attn(x, key_padding_mask, attn_mask)
        x = self.ffn(x)
        return x, self_attn_weights

class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = SelfAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)
        self.ffn        = FeedForwardLayer(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_output: torch.Tensor, 
        dec_key_padding_mask: Optional[torch.Tensor] = None, 
        enc_key_padding_mask: Optional[torch.Tensor] = None, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, self_attn_weights = self.self_attn(x, dec_key_padding_mask, attn_mask)
        x, cross_attn_weights = self.cross_attn(x, enc_output, enc_key_padding_mask)
        x = self.ffn(x)
        return x, self_attn_weights, cross_attn_weights
