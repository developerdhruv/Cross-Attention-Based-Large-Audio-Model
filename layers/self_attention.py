import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output