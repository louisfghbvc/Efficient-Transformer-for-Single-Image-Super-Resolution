import torch
import torch.nn as nn

class HFM(nn.Module):
    def __init__(self, k=3):
        super(HFM, self).__init__()
        self.k = k
        self.avgPool2D = nn.AvgPool2d(kernel_size=self.k, stride=self.k)
        self.upSample = nn.Upsample(scale_factor=self.k, mode='nearest')

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        tA = self.avgPool2D(tL)
        tU = self.upSample(tA)
        return tL - tU


if __name__ == '__main__':
    m = HFM(2)
    x = torch.tensor([float(i+1) for i in range(16)]).reshape((1, 1, 4, 4))
    y = m(x)
