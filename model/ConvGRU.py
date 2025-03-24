import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange

class Conv2DGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, dropout=0.2):
        super(Conv2DGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = 1
        self.dropout = nn.Dropout(p=dropout)

        self.i2h = nn.Conv2d(self.input_channels, 2 * self.hidden_channels, self.kernel_size, 
                             self.stride, self.padding, bias=True)
        self.h2h = nn.Conv2d(self.hidden_channels, 2 * self.hidden_channels, self.kernel_size, 
                             self.stride, self.padding, bias=True)
        self.xg = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 
                             self.stride, self.padding, bias=True)
        self.hg = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 
                             self.stride, self.padding, bias=True)

    def forward(self, x, state):
        i2h = self.i2h(self.dropout(x))
        h2h = self.h2h(state)
        
        gates = i2h + h2h
        cc_z, cc_r = torch.split(gates, self.hidden_channels, dim=1)

        rt = torch.sigmoid(cc_r)  # reset gate
        zt = torch.sigmoid(cc_z)  # update gate
        wx = self.xg(x)
        wh = self.hg(rt*state)
        gt = torch.tanh(wx + wh)  # output gate

        state_h = (1 - zt) * gt + zt * state  # hidden state

        return state_h

    def init_state(self, batch_size, hidden, shape, device=0):
        return Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).float().to(device)


class RecurrentBlock(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, horizon, img_size, input_channels, hidden_channels, kernel_size, dropout=0.2):
        super(RecurrentBlock, self).__init__()
        self.horizon = horizon
        self.img_size = img_size
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = Conv2DGRUCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i], dropout)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs, state=None):
        batch_size, step, channel, height, width = inputs.shape

        if state == None:
            internal_state = []
        else:
            internal_state = state

        outputs = []
        for t in range(step):
            x = inputs[:, t, :, :, :]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)

                if len(internal_state) != self.num_layers:
                    state = getattr(self, name).init_state(batch_size=batch_size,
                                                           hidden=self.hidden_channels[i],
                                                           shape=(height, width),
                                                           device=inputs.device)
                    internal_state.append(state)

                # do forward
                state = internal_state[i]
                x = getattr(self, name)(x, state)
                internal_state[i] = x

            outputs.append(x)  # (batch, hidden, height, width)
        outputs = torch.stack(outputs, dim=1)  # (batch, step, hidden, height, width)

        return outputs, internal_state