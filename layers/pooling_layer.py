import torch
import torch.nn as nn

class PoolingLayer(nn.Module):
    def __init__(self):
        super(PoolingLayer, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=0)
