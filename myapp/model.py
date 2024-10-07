import os

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=784, output_dim=47):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_2(x)
        return out




