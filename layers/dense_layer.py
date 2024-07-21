import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self):
        super(DenseLayer, self).__init__()
        self.dense = nn.Linear(1024, 512)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.dense(x))
