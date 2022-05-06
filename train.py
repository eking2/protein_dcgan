import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    
    from src.train_pipe import train

    #print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == '__main__':
    main()