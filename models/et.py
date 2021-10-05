import torch
import torch.nn as nn
from einops import rearrange
from debug import PrintLayer


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
    def __init__(self):
        pass


if __name__ == '__main__':
    # unit test
    m = EMHA(dim = 16, inChannels = 32)
    # B C N
    x = torch.tensor([float((i+1)%16) for i in range(16*32)]).reshape((1, 32, 16))
    y = m(x)