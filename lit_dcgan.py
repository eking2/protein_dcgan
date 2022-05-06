import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from models import (Generator16, Generator64, Generator128,
                    Discriminator16, Discriminator64, Discriminator128)
import pytorch_lightning as pl
import argparse
import h5py


class DCGAN(pl.LightningModule):
    def __init__(self, size, z_dim, learning_rate, beta1, beta2, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.fixed_z = torch.randn(8, self.hparams.z_dim, 1, 1)

        self.setup_models()

    def forward(self, z):

        '''generate a distogram'''

        return self.generator(z)

    @staticmethod
    def init_weights(module):

        '''normal weight init'''

        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def setup_models(self):

        '''setup models based on fragment size'''

        if self.hparams.size == 16:
            self.generator = Generator16(self.hparams.z_dim)
            self.discriminator = Discriminator16()

        elif self.hparams.size == 64:
            self.generator = Generator64(self.hparams.z_dim)
            self.discriminator = Discriminator64()

        else:
            self.generator = Generator128(self.hparams.z_dim)
            self.discriminator = Discriminator128()

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

    def configure_optimizers(self):

        lr = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        betas = (beta1, beta2)

        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        return [opt_d, opt_g], []

    def gen_fake(self):

        '''use same generated fake for both discriminator and generator training steps'''

        random_z = torch.randn_like(self.fixed_z, device=self.device)
        fake = self(random_z)

        return fake

    def generator_step(self, fake):

        '''how well does the fake fool the discriminator'''

        # fake through disc
        fake_pred = self.discriminator(fake).view(-1)
        real_labels = torch.ones_like(fake_pred, device=self.device)
        gen_loss = F.binary_cross_entropy(fake_pred, real_labels)

        return gen_loss

    def discriminator_step(self, real, fake):

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


    def training_step(self, batch, batch_idx, optimizer_idx):

        # add channels dim
        real = batch.unsqueeze(1)
        fake = self.gen_fake()

        # train disc
        if optimizer_idx == 0:
            disc_loss = self.discriminator_step(real, fake)
            self.log('disc_loss', disc_loss, on_epoch=True)

            return disc_loss

        # train gen
        if optimizer_idx == 1:
            gen_loss = self.generator_step(fake)
            self.log('gen_loss', gen_loss, on_epoch=True)

            return gen_loss

    def on_epoch_end(self):

        '''log generator distograms from fixed_z and randomly sampled'''

        random_z = torch.randn(8, self.hparams.z_dim, 1, 1, device=self.device)

        with torch.no_grad():
            from_fixed = self(self.fixed_z.type_as(random_z))
            from_random = self(random_z)

        fixed_grid = make_grid(from_fixed)
        random_grid = make_grid(from_random)

        self.logger.experiment.add_image('fixed', fixed_grid, self.current_epoch)
        self.logger.experiment.add_image('random', random_grid, self.current_epoch)

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                            help='learning rate (default: 1e-4)')
        parser.add_argument('-b1', '--beta1', type=float, default=0.5,
                            help='adam beta1 (default: 0.5)')
        parser.add_argument('-b2', '--beta2', type=float, default=0.999,
                            help='adam beta2 (default: 0.999)')

        parser.add_argument('-z', '--z_dim', type=int, default=100,
                            help='latent dim')
        parser.add_argument('-d', '--downsample', type=int, default=10,
                            help='downsampling constant (default: 10)')
        parser.add_argument('-s', '--size', type=int, choices=[16, 64, 128], default=64,
                            help='fragment length')

        return parser


def cli_main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('-p', '--path', type=str, default='./data/training_30_64.hdf5',
                        help='dataset path fragments hdf5')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='epochs (default: 10)')

    script_args, _ = parser.parse_known_args()


    # parse args
    parser = DCGAN.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dataset = Distograms(script_args.path)
    loader = DataLoader(dataset, batch_size = script_args.batch_size, shuffle=True, pin_memory=True)

    # model
    model = DCGAN(**vars(args))

    # train
    ngpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(gpus = ngpus,
                         max_epochs = script_args.epochs,
                         fast_dev_run = True)
    trainer.fit(model, train_dataloader = loader)


if __name__ == '__main__':

    cli_main()

