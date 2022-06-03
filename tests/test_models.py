from src.models import *
import torch
import torch.nn as nn

Z_DIM = 100
BATCH_SIZE = 10

def test_gen_block():
    block = gen_block(in_chan=100, out_chan=512, kernel=4, stride=1, padding=0)
    assert isinstance(block[0], nn.ConvTranspose2d)
    assert isinstance(block[1], nn.BatchNorm2d)
    assert isinstance(block[2], nn.LeakyReLU)

def test_disc_block():
    block = disc_block(in_chan=1, out_chan=64, kernel=4, stride=2, padding=1, bn=False)
    assert isinstance(block[0], nn.Conv2d)
    assert isinstance(block[1], nn.LeakyReLU)
    assert isinstance(block[2], nn.Dropout)

    block = disc_block(in_chan=64, out_chan=128, kernel=4, stride=2, padding=1, bn=True)
    assert isinstance(block[0], nn.Conv2d)
    assert isinstance(block[1], nn.BatchNorm2d)
    assert isinstance(block[2], nn.LeakyReLU)
    assert isinstance(block[3], nn.Dropout)

def test_enforce_sym():
    x = torch.randn(1, 1, 64, 64)
    x_sym = EnforceSym()(x)
    assert torch.allclose(x_sym, x_sym.permute(0, 1, 3, 2))

def test_gen64_out_shape():
    z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
    model = Generator64(Z_DIM)
    out = model(z)
    assert out.shape == (BATCH_SIZE, 1, 64, 64)

def test_disc64_out_shapes():
    img = torch.randn(BATCH_SIZE, 1, 64, 64)
    model = Discriminator64()
    out = model(img)
    assert out.shape == (BATCH_SIZE, 1, 1, 1)
