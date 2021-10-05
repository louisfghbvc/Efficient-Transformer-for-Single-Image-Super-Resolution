import torch
import torch.nn as nn
from torch.nn.modules import module

# class AFRB(nn.Module):
#     def __init__(self):
#         super(AFRB,self).__init__()
#             pass
    
#     def forward(self,x):
#         pass


def defaultConv(inChannels, outChannels, kernelSize, bias=True):
    return nn.Conv2d(
        inChannels, outChannels, kernelSize,
        padding=(kernelSize//2), bias=bias)



class ResidualUnit(nn.Module):
    def __init__(self,nFeats):
        super(ResidualUnit,self).__init__()

        self.reduction = defaultConv(nFeats,nFeats//2,3)
        self.expansion = defaultConv(nFeats//2,nFeats,3)
        self.lamRes = 0.5
        self.lamX= 0.5
    

    def forward(self,x):
        res = self.reduction(x) 
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res
        
        return  x

# class ARHM(nn.Module):
#     def __init__(self,nFeats):
#         super(ARHM,self).__init__()
    
#         RU = [ResidualUnit(nFeats) for _ in range(2)]
#         self.RU = nn.Sequential(*RU)

if __name__=="__main__":
    # RU = ResidualUnit(nFeats=4)
    x = torch.tensor([float(i+1) for i in range(64)]).reshape((1, 4, 4, 4))

    RU = ResidualUnit(x.shape[1])
    res = RU(x)
    

_
