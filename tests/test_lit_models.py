from src.lit_models import DCGAN
from src.datasets import Distograms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def test_dcgan():

    path = 'tests/data/demo_disto.hdf5'
    loader = DataLoader(Distograms(path, 100), batch_size=1)

    model = DCGAN(frag_size=64,
                  z_dim=100,
                  lr=1e-4,
                  beta1=0.5,
                  beta2=0.999)

    model.on_epoch_end = None

    trainer = pl.Trainer(max_epochs=1, 
                         checkpoint_callback=False,
                         logger=False)

    trainer.fit(model, train_dataloaders=loader)

    assert model
    


