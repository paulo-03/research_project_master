"""
Model implementation of a DnCNN model.
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, hidden_nc=15, bias=True):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        hidden_nc: number of body conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(DnCNN, self).__init__()

        # Initialize the head layer
        layers = [nn.Sequential(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, padding=1, bias=bias),
                                nn.ReLU(inplace=True))]

        # Add all the hidden/middle layers
        for _ in range(hidden_nc):
            layers.append(nn.Sequential(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, padding=1, bias=bias),
                                        nn.BatchNorm2d(nc),
                                        nn.ReLU(inplace=True)))
        # Finally add the last layer
        layers.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, padding=1, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, in_img):
        noised_img = in_img
        noise = self.layers(noised_img)
        return noised_img - noise
