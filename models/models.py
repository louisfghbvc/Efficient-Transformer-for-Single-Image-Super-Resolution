import torch
from torch import nn 
## (TODO) This is TEST CODE PLEASE REMOVE after implement model
class ESRT (nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self,x):
        return self.layers(x)
## (TODO) This is TEST CODE PLEASE REMOVE after implement model