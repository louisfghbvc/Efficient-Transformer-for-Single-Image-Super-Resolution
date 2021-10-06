import torch
import torch.nn as nn
from torch.nn.modules import module



def defaultConv(inChannels, outChannels, kernelSize, bias=True):
    return nn.Conv2d(
        inChannels, outChannels, kernelSize,
        padding=(kernelSize//2), bias=bias)



class ResidualUnit(nn.Module):
    def __init__(self, inChannel, outChannel, reScale,kernelSize=1, bias=True):
        super(ResidualUnit,self).__init__()

        self.reduction = defaultConv(inChannel, outChannel//2, kernelSize, bias)
        self.expansion = defaultConv(outChannel//2, inChannel, kernelSize, bias)
        self.lamRes = reScale.lamRes
        self.lamX= reScale.lamX
    

    def forward(self,x):
        res = self.reduction(x) 
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res
        
        return  x

class ARFB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super(ARFB,self).__init__()
        self.RU = ResidualUnit(inChannel, outChannel, reScale) 
        self.conv1 = defaultConv(2*inChannel, 2*outChannel, kernelSize=1)
        self.conv3 = defaultConv(2*inChannel, outChannel, kernelSize=3)
        self.reScale = reScale
    def forward(self,x):
        
        x_ru1 = self.RU(x)
        x_ru2 = self.RU(x_ru1)
        x_ru  = torch.cat((x_ru1,x_ru2),1)
        x_ru  = self.conv1(x_ru)
        x_ru  = self.conv3(x_ru)
        x_ru  = self.reScale.lamRes*x_ru
        x     = x*self.reScale.lamX + x_ru
        return  x
        
        
class Config():
    lamRes=0.5
    lamX = 0.5
if __name__== "__main__":
    # RU = ResidualUnit(nFeats=4)
    x = torch.tensor([float(i+1) for i in range(128)]).reshape((1, -1,4, 4))
    reScale = Config()
    arfb = ARFB(x.shape[1], x.shape[1],reScale)
    res = arfb(x)
    # RU = ResidualUnit(x.shape[1])
    # res = RU(x)
    


