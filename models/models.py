import torch.nn as nn
from hpb import HPB
from et import EfficientTransformer

class LCB(nn.Module):
    def __init__(self, num=3):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(HPB())
    
    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x

class LTB(nn.Module):
    def __init__(self, num=1):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(EfficientTransformer())
    
    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x

class ESRT(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv3 = nn.Conv2d(inChannels, outChannels)

        self.pixelShuffle = None

        self.path1 = nn.Sequential(
            LCB(),
            LTB(),
            self.conv3,
            self.pixelShuffle,
            self.conv3,
        )

        self.path2 = nn.Sequential(
            self.pixelShuffle,
            self.conv3,
        )
    
    def forward(self, x):
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1 + x2
        

