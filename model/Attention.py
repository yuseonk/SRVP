import os
import numpy as np
import torch
from torch import nn
from einops import rearrange

class TemporalAttention(nn.Module):
    """scaled dot-product attention with no linear projection."""
    def __init__(self):
        super(TemporalAttention, self).__init__()

    def forward(self, q, k, v):
        b, q_len, c, h, w = q.shape

        # rearrange
        wq = rearrange(q, 'b q_len c h w -> b q_len (c h w)')
        wk = rearrange(k, 'b kv_len c h w -> b (c h w) kv_len')
        wv = rearrange(v, 'b kv_len c h w -> b kv_len (c h w)')
        unit = (c*h*w)
        self.dk = np.sqrt(unit)

        # scaled-dot product attention
        similarity = torch.matmul(wq, wk) # (b, q_len, kv_len)
        scaled = similarity / self.dk
        if q_len != 1:
            scaled = torch.linalg.norm(scaled, dim=1, keepdim=True)
        score = torch.softmax(scaled, dim=-1)
        weighted_score = torch.matmul(score, wv) # (b, q_len, unit)
        weighted_score = rearrange(weighted_score, 'b q_len (c h w) -> b q_len c h w',
                                   b=b, q_len=1, c=c, h=h, w=w)
        return weighted_score, score


class SpatialAttention(nn.Module):
    """linear projection -> scaled dot-product attention"""
    def __init__(self, num_units):
        super(SpatialAttention, self).__init__()
        self.Wq = nn.Conv2d(num_units, num_units, kernel_size=1)
        self.Wk = nn.Conv2d(num_units, num_units, kernel_size=1)
        self.Wv = nn.Conv2d(num_units, num_units, kernel_size=1)

    def forward(self, q, k, v):
        b, c, h, w = q.shape

        # linear projection
        wq, wk, wv = self.Wq(q), self.Wk(k), self.Wv(v)

        # rearrange
        wq = rearrange(wq, 'b c h w -> b c (h w)')
        wk = rearrange(wk, 'b c h w -> b (h w) c')
        wv = rearrange(wv, 'b c h w -> b c (h w)')
        unit = (h*w)
        self.dk = np.sqrt(unit)

        # scaled-dot product attention
        similarity = torch.matmul(wq, wk) # (b, c, c)
        scaled = similarity / self.dk
        score = torch.softmax(scaled, dim=-1)
        weighted_score = torch.matmul(score, wv) # (b, c, hw)
        weighted_score = rearrange(weighted_score, 'b c (h w) -> b c h w',
                                   b=b, c=c, h=h, w=w)
        return weighted_score, score