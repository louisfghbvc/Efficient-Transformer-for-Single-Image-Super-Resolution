import torch
import torch.nn as nn

# Use for debug
class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        print(x)
        return x