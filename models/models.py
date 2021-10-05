import torch
from torch import nn 
## (TODO) This is TEST CODE PLEASE REMOVE after implement model
class ESRT (nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
    def forward(self,x):
        return self.layers(x)
## (TODO) This is TEST CODE PLEASE REMOVE after implement model