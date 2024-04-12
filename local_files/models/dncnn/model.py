"""
Model implementation of a DnCNN model.
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, hidden_nc=17, bias=True):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        hidden_nc: number of body conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(DnCNN, self).__init__()
        head = [nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, padding=1, bias=bias),
                nn.ReLU(inplace=True)]
        body = [nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, padding=1, bias=bias),
                nn.BatchNorm2d(nc),
                nn.ReLU(inplace=True)]*hidden_nc
        tail = nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, padding=1, bias=bias)

        self.model = nn.Sequential(*head, *body, tail)

    def forward(self, in_img):
        noise = self.model(in_img)
        return in_img-noise
