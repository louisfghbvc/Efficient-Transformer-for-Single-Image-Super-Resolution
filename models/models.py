import torch
import torch.nn as nn
from hpb import HPB
from et import EfficientTransformer

class BackBoneBlock(nn.Module):
    def __init__(self, fm, num):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm)
    
    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x

class ESRT(nn.Module):
    def __init__(self, inChannels, scaleFactor):
        super().__init__()
        self.conv3 = nn.Conv2d(inChannels, 32, kernel_size=3)

        self.path1 = nn.Sequential(
            BackBoneBlock(HPB(), 3),
            BackBoneBlock(EfficientTransformer(), 1),
            self.conv3,
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(inChannels, 3, kernel_size=3),
        )

        self.path2 = nn.Sequential(
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(inChannels, 3, kernel_size=3),
        )
    
    def forward(self, x):
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1 + x2


if __name__ == '__main__':
    x = torch.tensor([float(i+1) for i in range(3*48*48)]).reshape((1, 3, 48, 48))
    
    model = ESRT(inChannels=2, scaleFactor=2)

