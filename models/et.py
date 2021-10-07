import torch
import torch.nn as nn
from einops import rearrange
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
    def __init__(self, dim, inChannels, splitFactors=4, heads=8, dimHead=64, dropOut=0.):
        super().__init__()
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
            nn.Conv1d(in_channels=inChannels//2, out_channels=inChannels, kernel_size=1),
        )

    def forward(self, x):
        x = self.reduct(x)
        qkv = self.toQKV(x).chunk(3, dim = -1)
        # TODO: evaluate rearrange
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(self.splitFactors, dim = 2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b (h d) n')
            pool.append(out)
        
        out = torch.cat(tuple(pool), dim = -1)

        return self.toOut(out)

class EfficientTransformer(nn.Module):
    def __init__(self, dim, mlpDim, inChanngels, k=3, splitFactors=4, heads=8, dimHead=64, dropOut=0.):
        super().__init__()

        self.k = k
        self.unFold = nn.Unfold(kernel_size=(k, k))

        self.emha = PreNorm(dim, EMHA(dim, inChannels=inChanngels*k*k, splitFactors=splitFactors, heads=heads, dimHead=dimHead, dropOut=dropOut))
        self.mlp = PreNorm(dim, FeedForward(dim, mlpDim, dropOut=dropOut))

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.unFold(x)
        x = self.emha(x) + x
        x = self.mlp(x) + x
        fold = nn.Fold(output_size=(h, w), kernel_size=(self.k, self.k))
        return fold(x)

if __name__ == '__main__':
    # unit test
    # m = EMHA(dim = 16, inChannels = 32)
    # # B C N
    # x = torch.tensor([float((i+1)%16) for i in range(16*32)]).reshape((1, 32, 16))
    # y = m(x)

    et = EfficientTransformer(dim=4, inChanngels=18*8, mlpDim=2048)
    # B C H W
    x = torch.tensor([float((i+1)%16) for i in range(18*8*18)]).reshape((1, 18*8, 3, 6))
    y = et(x)