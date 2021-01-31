import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import (Generator16, Generator64,
                    Discriminator16, Discriminator64)
import pytorch_lightning as pl
import argparse
import h5py


class Distrograms(Dataset):
    def __init__(self, path):
        super().__init__()

        self.maps = h5py.File(path, 'r')['arr']

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
        return torch.as_tensor(self.maps[idx, ...], dtype=torch.float)
        

class DCGAN(pl.LightningModule):
    def __init__(self, batch_size, z_dim, learning_rate, beta1, beta2, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.fixed_z = torch.randn(8, self.hparams.z_dim, 1, 1)

    def forward(self, z):
        return self.gen(z)

    def configure_optimizers(self):
        pass

    def generator_step(self, x):
        pass

    def discriminator_step(self, x):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

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
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='dataset path fragments hdf5')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='epochs (default: 50)')

    script_args, _ = parser.parse_known_args()

    # parse args
    parser = DCGAN.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dataset = Distrograms(script_args.path)
    loader = DataLoader(dataset, batch_size = script_args.batch_size, shuffle=True, pin_memory=True)

    # model


    # train




if __name__ == '__main__':

    cli_main()
    
    #ds = Distrograms('./data/training_30_16.hdf5')
    
