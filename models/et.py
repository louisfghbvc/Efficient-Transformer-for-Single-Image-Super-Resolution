import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from debug import PrintLayer

# LayerNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# TODO: not sure numbers of layer in mlp
class FeedForward(nn.Module):
    def __init__(self, dim, hiddenDim, dropOut = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hiddenDim),
            nn.GELU(),
            nn.Dropout(dropOut),
            nn.Linear(hiddenDim, dim),
            nn.Dropout(dropOut)
        )
    def forward(self, x):
        return self.net(x)

# Efficient Multi-Head Attention
class EMHA(nn.Module):
    def __init__(self, inChannels, splitFactors=4, heads=8, dimHead=64, dropOut=0.):
        super().__init__()
        dim = inChannels // 2
        innerDim = dimHead * heads

        self.heads = heads
        self.splitFactors = splitFactors
        self.scale = dimHead ** -0.5

        self.reduct = nn.Conv1d(in_channels=inChannels, out_channels=inChannels//2, kernel_size=1)
        self.attend = nn.Softmax(dim = -1)
        self.toQKV = nn.Linear(dim, innerDim * 3, bias = False)
        self.toOut = nn.Sequential(
            nn.Linear(innerDim, dim),
            nn.Dropout(dropOut),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(in_channels=inChannels//2, out_channels=inChannels, kernel_size=1),
        )

    def forward(self, x):
        x = self.reduct(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(self.splitFactors, dim = 1), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)
        
        out = torch.cat(tuple(pool), dim = -1)

        return self.toOut(out)

class EfficientTransformer(nn.Module):
    def __init__(self, normDim, mlpDim, inChannels, k=3, splitFactors=4, heads=8, dimHead=64, dropOut=0.):
        super().__init__()

        self.k = k
        self.unFold = nn.Unfold(kernel_size=(k, k), padding=1)

        self.emha = PreNorm(normDim, EMHA(inChannels=inChannels*k*k, splitFactors=splitFactors, heads=heads, dimHead=dimHead, dropOut=dropOut))
        self.mlp = PreNorm(normDim, FeedForward(normDim, mlpDim, dropOut=dropOut))

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.unFold(x)
        x = self.emha(x) + x
        x = self.mlp(x) + x
        return F.fold(x, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))

if __name__ == '__main__':
    ### unit test
    # m = EMHA(inChannels = 32)
    ### B C N
    # x = torch.tensor([float((i+1)%16) for i in range(18*32)]).reshape((1, 32, 18))
    # y = m(x)

    ### dim = channel // 2, normDim = h*w
    et = EfficientTransformer(normDim=4*8, inChannels=18*8, mlpDim=2048)
    ### B C H W
    x = torch.tensor([float((i+1)%16) for i in range(18*8*32)]).reshape((1, 18*8, 4, 8))
    y = et(x)
    print(y.shape)