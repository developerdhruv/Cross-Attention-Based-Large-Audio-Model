import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        self.output = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.softmax(self.output(x))
