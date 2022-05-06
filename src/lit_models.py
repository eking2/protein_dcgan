import torch
from torch import Tensor
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from .models import (Generator16, Generator64, Generator128,
                     Discriminator16, Discriminator64, Discriminator128)

class DCGAN(pl.LightningModule):
    def __init__(self, frag_size: int, z_dim: int, lr: float, beta1: float, beta2: float, downsample: int):
        super().__init__()

        self.save_hyperparameters()
        self.fixed_z = torch.randn(8, self.hparams.z_dim, 1, 1)

        self.setup_models()

    def forward(self, z: Tensor) -> Tensor:

        '''generate a distogram'''

        return self.generator(z)

    @staticmethod
    def init_weights(module: nn.Module) -> None:

        '''normal weight init'''

        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def setup_models(self) -> None:

        '''setup models based on fragment size'''

        if self.hparams.frag_size == 16:
            self.generator = Generator16(self.hparams.z_dim)
            self.discriminator = Discriminator16()

        elif self.hparams.frag_size == 64:
            self.generator = Generator64(self.hparams.z_dim)
            self.discriminator = Discriminator64()

        else:
            self.generator = Generator128(self.hparams.z_dim)
            self.discriminator = Discriminator128()

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

    def configure_optimizers(self):

        lr = self.hparams.lr
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        betas = (beta1, beta2)

        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        return [opt_d, opt_g], []

    def gen_fake(self) -> Tensor:

        '''use same generated fake for both discriminator and generator training steps'''

        random_z = torch.randn_like(self.fixed_z, device=self.device)
        fake = self(random_z)

        return fake

    def generator_step(self, fake: Tensor) -> Tensor:

        '''how well does the fake fool the discriminator'''

        # fake through disc
        fake_pred = self.discriminator(fake).view(-1)
        real_labels = torch.ones_like(fake_pred, device=self.device)
        gen_loss = F.binary_cross_entropy(fake_pred, real_labels)

        return gen_loss

    def discriminator_step(self, real: Tensor, fake: Tensor) -> Tensor:

        '''can the discriminator distinguish fakes and reals'''

        # real
        real_pred = self.discriminator(real).view(-1)
        real_labels = torch.ones_like(real_pred, device=self.device)
        real_loss = F.binary_cross_entropy(real_pred, real_labels)

        # fake
        # do not backward through gen
        fake_pred = self.discriminator(fake.detach()).view(-1)
        fake_labels = torch.zeros_like(fake_pred, device=self.device)
        fake_loss = F.binary_cross_entropy(fake_pred, fake_labels)

        disc_loss = (real_loss + fake_loss) / 2

        return disc_loss


    def training_step(self, batch: Tensor, batch_idx: int, optimizer_idx: int) -> Tensor:

        # add channels dim -> (bchw)
        # downsample instead of normalize
        real = batch.unsqueeze(1) / self.hparams.downsample
        fake = self.gen_fake()

        # train disc
        if optimizer_idx == 0:
            disc_loss = self.discriminator_step(real, fake)
            self.log('disc_loss', disc_loss, on_epoch=True, prog_bar=True)

            return disc_loss

        # train gen
        if optimizer_idx == 1:
            gen_loss = self.generator_step(fake)
            self.log('gen_loss', gen_loss, on_epoch=True, prog_bar=True)

            return gen_loss

    def on_epoch_end(self) -> None:

        '''log generator distograms from fixed_z and randomly sampled'''

        self.eval()
        random_z = torch.randn(8, self.hparams.z_dim, 1, 1, device=self.device)

        with torch.no_grad():
            from_fixed = self(self.fixed_z.type_as(random_z))
            from_random = self(random_z)

        fixed_grid = make_grid(from_fixed)
        random_grid = make_grid(from_random)

        self.logger.experiment.add_image('fixed', fixed_grid, self.current_epoch)
        self.logger.experiment.add_image('random', random_grid, self.current_epoch)