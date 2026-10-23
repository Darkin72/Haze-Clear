"""SwiftWave Dehaze Net: a small Haar encoder/decoder trained from random weights.

This is a new implementation combining established Haar and gated-convolution ideas;
no claim of research novelty. It uses ordinary PyTorch operators only.
"""
import torch
from torch import nn
import torch.nn.functional as F

ARCH_VERSION = 'swdnet-v1'


class HaarDown(nn.Module):
    def forward(self, x):
        a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
        c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
        return torch.cat(((a+b+c+d)*.5, (-a-b+c+d)*.5,
                          (-a+b-c+d)*.5, (a-b-c+d)*.5), 1)


class HaarUp(nn.Module):
    def forward(self, x):
        ll, lh, hl, hh = x.chunk(4, 1)
        # Stack in pixel-shuffle order, preserving each channel's four spatial samples.
        a = (ll-lh-hl+hh)*.5
        b = (ll-lh+hl-hh)*.5
        c = (ll+lh-hl-hh)*.5
        d = (ll+lh+hl+hh)*.5
        return F.pixel_shuffle(torch.stack((a, b, c, d), 2).flatten(1, 2), 2)


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Accumulate normalization in FP32, including during BF16 training.
        z = x.float()
        z = (z-z.mean(1, keepdim=True))*torch.rsqrt(z.var(1, keepdim=True, unbiased=False)+1e-6)
        return (z*self.weight+self.bias).to(x.dtype)


class GatedWaveBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, 2*channels, 1)
        self.spatial = nn.Conv2d(2*channels, 2*channels, 5, padding=2*dilation,
                                 dilation=dilation, groups=2*channels)
        self.context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), .1))

    def forward(self, x):
        a, b = self.spatial(self.expand(self.norm(x))).chunk(2, 1)
        z = a*F.silu(b)
        return x+self.scale*self.project(z*self.context(z))


def blocks(channels, count, dilated=False):
    return nn.Sequential(*(GatedWaveBlock(channels, 1+(i % 2) if dilated else 1) for i in range(count)))


class SWDNet(nn.Module):
    def __init__(self, width=32, depths=(2, 2, 4, 2, 2)):
        super().__init__()
        if width < 8 or len(depths) != 5 or any(d < 1 for d in depths):
            raise ValueError('width >= 8; depths chứa năm số nguyên dương.')
        self.spec = {'arch_version': ARCH_VERSION, 'width': int(width), 'depths': list(depths)}
        self.haar, self.inverse = HaarDown(), HaarUp()
        self.stem = nn.Conv2d(12, width, 1)
        self.enc1 = blocks(width, depths[0])  # All deep processing starts at half resolution.
        self.down1 = nn.Conv2d(width*4, width*2, 1)
        self.enc2 = blocks(width*2, depths[1])
        self.down2 = nn.Conv2d(width*8, width*3, 1)
        self.middle = blocks(width*3, depths[2], dilated=True)
        self.up2 = nn.Conv2d(width*3, width*8, 1)
        self.dec2 = blocks(width*2, depths[3])
        self.up1 = nn.Conv2d(width*2, width*4, 1)
        self.dec1 = blocks(width, depths[4])
        self.wave_residual = nn.Conv2d(width, 12, 3, padding=1)
        detail_width = max(8, width//2)
        self.detail = nn.Sequential(nn.Conv2d(3, detail_width, 1), nn.SiLU(),
                                    nn.Conv2d(detail_width, detail_width, 3, padding=1, groups=detail_width),
                                    nn.SiLU(), nn.Conv2d(detail_width, 3, 1))
        # Exact identity at initialization; only the two output heads start at zero.
        nn.init.zeros_(self.wave_residual.weight); nn.init.zeros_(self.wave_residual.bias)
        nn.init.zeros_(self.detail[-1].weight); nn.init.zeros_(self.detail[-1].bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        padded = F.pad(x, (0, pw, 0, ph), mode='replicate') if ph or pw else x
        e1 = self.enc1(self.stem(self.haar(padded)))
        e2 = self.enc2(self.down1(self.haar(e1)))
        z = self.middle(self.down2(self.haar(e2)))
        z = self.dec2(self.inverse(self.up2(z))+e2)
        z = self.dec1(self.inverse(self.up1(z))+e1)
        restored = padded+self.inverse(self.wave_residual(z))+self.detail(padded)
        # Never clamp inside training; the evaluator clamps to [0, 1].
        return restored[..., :h, :w]
