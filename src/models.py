from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gen_block(
    in_chan: int, out_chan: int, kernel: int, stride: int, padding: int
) -> List[nn.Module]:

    block = []
    block.append(
        nn.ConvTranspose2d(in_chan, out_chan, kernel, stride, padding, bias=False)
    )
    block.append(nn.BatchNorm2d(out_chan))
    block.append(nn.LeakyReLU(0.2, inplace=True))

    return block


def disc_block(
    in_chan: int, out_chan: int, kernel: int, stride: int, padding: int, bn: bool
) -> List[nn.Module]:

    block = []
    if bn:
        block.append(nn.Conv2d(in_chan, out_chan, kernel, stride, padding, bias=False))
        block.append(nn.BatchNorm2d(out_chan))
    else:
        block.append(nn.Conv2d(in_chan, out_chan, kernel, stride, padding))

    block.append(nn.LeakyReLU(0.2, inplace=True))
    block.append(nn.Dropout(0.1))

    return block


class EnforceSym(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:

        # relu before clamps >0
        # g(z) = (g(z) + g(z).T)/2, from sup
        # transpose on hw dims, leave bc alone
        return (x + x.permute(0, 1, 3, 2)) / 2


class Generator16(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        # H_out = (H_in - 1)*stride + H_filter - 2*padding
        # 4 = (1-1)*1 + 4 - 2*0

        # input = (b, 100, 1, 1)
        self.net = nn.Sequential(
            *gen_block(z_dim, 512, 4, 1, 0),     # (b, 512, 4, 4)
            *gen_block(512, 256, 3, 1, 1),       # (b, 256, 4, 4)
            *gen_block(256, 128, 4, 2, 1),       # (b, 128, 8, 8)
            *gen_block(128, 64, 4, 2, 1),        # (b, 64, 16, 16)
            nn.ConvTranspose2d(64, 1, 3, 1, 1),  # (b, 1, 16, 16)
            nn.ReLU(inplace=True),
            EnforceSym()
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)


class Discriminator16(nn.Module):
    def __init__(self):
        super().__init__()

        # H_out = (H_in - H_filter + 2*padding)/stride + 1
        # 16 = (16-3+ 2*1)/1 + 1

        # input = (b, 1, 16, 16)
        self.net = nn.Sequential(
            *disc_block(1, 64, 3, 1, 1, bn=False),    # (b, 64, 16, 16)
            *disc_block(64, 128, 4, 2, 1, bn=True),   # (b, 128, 8, 8)
            *disc_block(128, 256, 4, 2, 1, bn=True),  # (b, 256, 4, 4)
            *disc_block(256, 512, 3, 1, 1, bn=True),  # (b, 512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0),               # (b, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)


class Generator64(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        # input = (b, 100, 1, 1)
        self.net = nn.Sequential(
            *gen_block(z_dim, 512, 4, 1, 0),     # (b, 512, 4, 4)
            *gen_block(512, 256, 4, 2, 1),       # (b, 256, 8, 8)
            *gen_block(256, 128, 4, 2, 1),       # (b, 128, 16, 16)
            *gen_block(128, 64, 4, 2, 1),        # (b, 64, 32, 32)
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # (b, 1, 64, 64)
            nn.ReLU(inplace=True),
            EnforceSym()
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)


class Discriminator64(nn.Module):
    def __init__(self):
        super().__init__()

        # input = (b, 1, 64, 64)
        self.net = nn.Sequential(
            *disc_block(1, 64, 4, 2, 1, bn=False),    # (b, 64, 32, 32)
            *disc_block(64, 128, 4, 2, 1, bn=True),   # (b, 128, 16, 16)
            *disc_block(128, 256, 4, 2, 1, bn=True),  # (b, 256, 8, 8)
            *disc_block(256, 512, 4, 2, 1, bn=True),  # (b, 512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0),               # (b, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)


class Generator128(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()

        # input = (b, 100, 1, 1)
        self.net = nn.Sequential(
            *gen_block(z_dim, 512, 4, 4, 0),     # (b, 512, 4, 4)
            *gen_block(512, 256, 4, 2, 1),       # (b, 256, 8, 8)
            *gen_block(256, 128, 4, 4, 0),       # (b, 128, 32, 32)
            *gen_block(128, 64, 4, 2, 1),        # (b, 64, 64, 64)
            nn.ConvTranspose2d(64, 1, 4, 2, 1),  # (b, 1, 128, 128)
            nn.ReLU(inplace=True),
            EnforceSym()
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)


class Discriminator128(nn.Module):
    def __init__(self):
        super().__init__()

        # input = (b, 1, 128, 128)
        self.net = nn.Sequential(
            *disc_block(1, 64, 4, 2, 1, bn=False),    # (b, 64, 64, 64)
            *disc_block(64, 128, 4, 2, 1, bn=True),   # (b, 128, 32, 32)
            *disc_block(128, 256, 4, 4, 0, bn=True),  # (b, 256, 8, 8)
            *disc_block(256, 512, 4, 2, 1, bn=True),  # (b, 512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0),               # (b, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:

        return self.net(x)


def test():

    z_dim = 100
    gen_in = torch.randn(10, 100, 1, 1)
    print("gen input:", gen_in.shape)
    print()

    disc16_in = torch.randn(10, 1, 16, 16)
    disc64_in = torch.randn(10, 1, 64, 64)
    disc128_in = torch.randn(10, 1, 128, 128)

    model = Generator16(z_dim)
    print("gen16 output:", model(gen_in).shape)
    model = Discriminator16()
    print("disc16 output:", model(disc16_in).shape)
    print()

    model = Generator64(z_dim)
    print("gen64 output:", model(gen_in).shape)
    model = Discriminator64()
    print("disc64 output:", model(disc64_in).shape)
    print()

    model = Generator128(z_dim)
    print("gen128 output:", model(gen_in).shape)
    model = Discriminator128()
    print("disc128 output:", model(disc128_in).shape)
    print()


if __name__ == "__main__":
    test()
