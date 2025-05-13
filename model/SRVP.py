import os
import numpy as np
import torch
from torch import nn
from typing import Optional
from .ConvGRU import *
from .Attention import *
from .Modules import *


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, dropout, channel, group=1, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(group, channel)

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)


class LinearProjection(nn.Module):
    """linear projection to each frame."""
    def __init__(self, num, layer, **kwargs):
        super(LinearProjection, self).__init__(**kwargs)
        self.layers = nn.ModuleList([])
        for _ in range(num):
            self.layers.append(layer)

    def forward(self, x):
        step = x.shape[1]
        features = []
        for t in range(step):
            frame = x[:, t,...] # (b, c, h, w)
            features.append(self.layers[t](frame))
        return torch.stack(features, dim=1) # (b, t, c, h, w)


class SpatialSelfCorrelation(nn.Module):
    """generate spatial self-correlation map for a target frame."""
    def __init__(self, in_channels, out_channels):
        super(SpatialSelfCorrelation, self).__init__()
        self.norm = AddNorm(0.2, in_channels, in_channels)
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.Tanh())

    def get_corr(self, X):
        b, c, h, w = X.shape
        # channel-wise softmax
        S = torch.softmax(X, dim=1)
        # matmul
        X = rearrange(X, 'b c h w -> b c (h w)')
        S = rearrange(S, 'b c h w -> b (h w) c')
        P = torch.matmul(X, S) # (b,c,soft region)
        P = P/(h*w)
        # channel-wise l2 norm
        NP = torch.linalg.norm(P, dim=1, keepdim=True) # (b,1,soft region)
        NX = torch.linalg.norm(X, dim=1, keepdim=True) # (b,1,hw)
        # gen correlation
        corr = torch.matmul(NP.transpose(1,2), NX) # (b,soft region,hw)
        return rearrange(corr, 'b c (h w) -> b c h w', b=b, c=c, h=h, w=w)

    def forward(self, X):
        selfcorr = self.get_corr(X)
        selfcorr = self.norm(selfcorr, X)
        return self.conv2d(selfcorr), selfcorr # (b,c,h,w)


class TemporalSelfCorrelation(nn.Module):
    """generate temporal self-correlation map for reference frames."""
    def __init__(self, input_time, raw_channels, in_channels, device):
        super(TemporalSelfCorrelation, self).__init__()
        self.lpconv = LinearProjection(input_time, 
                                       nn.Sequential(nn.Conv2d(raw_channels, in_channels//2, kernel_size=3, padding=1, device=device),
                                                     nn.BatchNorm2d(in_channels//2),
                                                     nn.ReLU(),
                                                     nn.Conv2d(in_channels//2, in_channels, kernel_size=3, padding=1, device=device),
                                                     nn.BatchNorm2d(in_channels),
                                                     nn.ReLU()))
        self.norm = AddNorm(0.2, input_time, input_time)

    def get_corr(self, X_dot):
        b, t, c, h, w = X_dot.shape
        # time-wise softmax
        S = torch.softmax(X_dot, dim=1)
        # matmul
        X_dot = rearrange(X_dot, 'b t c h w -> b t (c h w)')
        S = rearrange(S, 'b t c h w -> b (c h w) t')
        P = torch.matmul(X_dot, S) # (b,t, soft time)
        P = P/(c*h*w)
        # time-wise l2 norm
        NP = torch.linalg.norm(P, dim=1, keepdim=True) # (b,1,soft time)
        NX = torch.linalg.norm(X_dot, dim=1, keepdim=True) # (b,1,chw)
        # gen correlation
        corr = torch.matmul(NP.transpose(1,2), NX) # (b,soft time,chw)
        return rearrange(corr, 'b t (c h w) -> b t c h w', b=b, t=t, c=c, h=h, w=w)

    def forward(self, X, X_f):
        X_dot = self.lpconv(X) # feature extraction for each frame
        selfcorr = self.get_corr(X_dot) # get global texture
        return self.norm(selfcorr, X_f), selfcorr # temporal info fusion


class EncoderDecoder(nn.Module):
    """the proposed model based on GRU Encoder-Forecaster."""
    def __init__(self, emb_dim, input_time, input_channels, hidden_channels, kernel_size, img_size,
                 dropout=0.2, horizon=1, output_channels=1, device=None):
        super(EncoderDecoder, self).__init__()
        self.horizon = horizon
        self.num_rnn = len(hidden_channels)
        self.encoder = RecurrentBlock(horizon, img_size, input_channels, hidden_channels, kernel_size, dropout)
        self.forecast = RecurrentBlock(horizon, img_size, input_channels, hidden_channels, kernel_size, dropout)
        
        self.step1 = StandardAttentionModule(TemporalAttention(), 
                                             SpatialAttention(hidden_channels[-1]),
                                             SpatialAttention(hidden_channels[-1]),
                                             SpatialAttention(hidden_channels[-1]))

        self.enc_sc = TemporalSelfCorrelation(input_time, input_channels, hidden_channels[-1], device)
        self.step2 = SelfCorrAttentionModule(SpatialSelfCorrelation(hidden_channels[-1]*self.num_rnn, hidden_channels[-1]),
                                             TemporalAttention(), 
                                             SpatialAttention(hidden_channels[-1]),
                                             nn.Sequential(nn.Conv2d(hidden_channels[-1], emb_dim, kernel_size=3, padding=1),
                                                           nn.Tanh()),
                                             nn.Sequential(nn.Conv2d(hidden_channels[-1], emb_dim, kernel_size=3, padding=1),
                                                           nn.Tanh()),
                                             SpatialAttention(emb_dim),
                                             SpatialAttention(emb_dim),
                                             nn.Sequential(nn.Conv2d(emb_dim, emb_dim//2, kernel_size=3, padding=1),
                                                           nn.Tanh()),
                                             nn.Sequential(nn.Conv2d(emb_dim, emb_dim//2, kernel_size=3, padding=1),
                                                           nn.Tanh()))

        in_channels = hidden_channels[-1]*self.num_rnn+(hidden_channels[-1])*2+(emb_dim//2)*2 # curr hidden, standard, selfcorr
        self.output_layer = nn.Sequential(nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1),
                                          nn.Sigmoid())
        
        self._enc_state: Optional[list] = None
        self._dec_state: Optional[list] = None

    @property
    def enc_state(self) -> Optional[list]:
        return self._enc_state

    @property
    def dec_state(self) -> Optional[list]:
        return self._dec_state

    def forward(self, inputs):
        '''
        encode
        enc_output: (b,t,c,h,w)
        '''
        enc_output, enc_state = self.encoder(inputs)
        if not self.training:
            self._enc_state = enc_state.copy()
        
        '''
        encoder output self-correlation
        do this process only at the first time step.
        '''
        enc_corr, enc_feature_st = self.enc_sc(inputs, enc_output)
        if not self.training:
            self.enc_corr = enc_corr
            self.enc_feature_st = enc_feature_st

        '''
        forecast
        dec_state: [(b,c,h,w),...]
        '''
        z = inputs[:, -1:, ...]  # X_t
        state = enc_state
        outputs = []
        self._dec_state = [None]*self.horizon
        if not self.training:
            self.corr, self.s_features, self.s_weights, self.sc_features, self.sc_weights = [], [], [], [], []
        # generate future sequences reculsively
        for t in range(self.horizon):
            _, state = self.forecast(z, state)
            if not self.training:
                self._dec_state[t] = state.copy()

            hidden_states = state.copy()
            all_hidden = torch.concat(hidden_states, dim=1) # (b, layer*hidden, h, w)
            curr_hidden = torch.stack(hidden_states, dim=1) # (b, layer, hidden, h, w)

            '''standard attention'''
            s_features, s_weights = self.step1(enc_output, curr_hidden)
            if not self.training:
                self.s_features.append(s_features)
                self.s_weights.append(s_weights)

            '''self-correlation attention'''
            corr, sc_features, sc_weights = self.step2(enc_corr, all_hidden)
            if not self.training:
                self.corr.append(corr)
                self.sc_features.append(sc_features)
                self.sc_weights.append(sc_weights)
            
            '''output layer'''
            # spatialfused, temporalfused
            concated = torch.cat([all_hidden,s_features[2],s_features[3],sc_features[2],sc_features[3]], dim=1)
            z = self.output_layer(concated).unsqueeze(1).contiguous() # (batch, 1, channel, height, width)
            outputs.append(z)
        return torch.cat(outputs, dim=1).contiguous()  # (batch, horizon, channel, height, width)