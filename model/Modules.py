import os
import numpy as np
import torch
from torch import nn

class StandardAttentionModule(nn.Module):
    """temporal attention -> spatial self-attention -> cross attention."""
    def __init__(self, ta, sa, ta_to_sa, sa_to_ta):
        super(StandardAttentionModule, self).__init__()
        self.ta = ta
        self.sa = sa
        self.ta_to_sa = ta_to_sa
        self.sa_to_ta = sa_to_ta

    def forward(self, ref, target):
        # do temporal attention
        context, ta_weights = self.ta(target, ref, ref)  # (b, 1, c, h, w)

        # do self-attention
        target = torch.linalg.norm(target, dim=1)
        refined, sa_weights = self.sa(target, target, target) # (b, c, h, w)

        # do cross-attention
        context = context.squeeze(1)
        spatialfused, sf_weights = self.ta_to_sa(context, refined, refined) # (b, c, h, w)
        temporalfused, tf_weights = self.sa_to_ta(refined, context, context) # (b, c, h, w)

        return [context, refined, spatialfused, temporalfused], [ta_weights, sa_weights, sf_weights, tf_weights]


class SelfCorrAttentionModule(nn.Module):
    """get the self-correlation map of the target frame -> temporal attention -> spatial self-attention -> cross attention."""
    def __init__(self, dec_sc, ta, sa, c_conv, r_conv, ta_to_sa, sa_to_ta, sf_conv, tf_conv):
        super(SelfCorrAttentionModule, self).__init__()
        self.dec_sc = dec_sc
        self.ta = ta
        self.sa = sa
        self.c_conv = c_conv
        self.r_conv = r_conv
        self.ta_to_sa = ta_to_sa
        self.sa_to_ta = sa_to_ta
        self.sf_conv = sf_conv
        self.tf_conv = tf_conv

    def forward(self, enc_corr, target):
        # self-correlation
        dec_corr, dec_feature_st = self.dec_sc(target)

        # do temporal attention
        context, ta_weights = self.ta(dec_corr.unsqueeze(1), enc_corr, enc_corr)  # (b, 1, hidden, h, w)
        context = context.squeeze(1)
        context = self.c_conv(context) # (b, emb_dim, h, w)
        
        # do self-attention
        refined, sa_weights = self.sa(dec_corr, dec_corr, dec_corr) # (b, hidden, h, w)
        refined = self.r_conv(refined) # (b, emb_dim, h, w)

        # do cross-attention
        spatialfused, sf_weights = self.ta_to_sa(context, refined, refined) # (b, emb_dim, h, w)
        spatialfused = self.sf_conv(spatialfused) # (b, emb_dim//2, h, w)
        temporalfused, tf_weights = self.sa_to_ta(refined, context, context) # (b, emb_dim, h, w)
        temporalfused = self.tf_conv(temporalfused) # (b, emb_dim//2, h, w)

        return [dec_corr, dec_feature_st], [context, refined, spatialfused, temporalfused], [ta_weights, sa_weights, sf_weights, tf_weights]