import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

def train(cfg: DictConfig) -> float:
    
    # no datamodule, only train loader
    train_loader = instantiate(cfg.data)

    # dcgan lit module
    model = instantiate(cfg.model)

    # trainer
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=cfg.trainer.epochs)
                         #fast_dev_run=True)

    trainer.fit(model, train_dataloaders=train_loader)

    gen_loss = trainer.callback_metrics.get('gen_loss')
    disc_loss = trainer.callback_metrics.get('disc_loss')

    return gen_loss, disc_loss