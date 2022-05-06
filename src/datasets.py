import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class Distograms(Dataset):
    def __init__(self, path: str):
        super().__init__()

        with h5py.File(path, "r") as f:
            self.maps = f["arr"][:]

    def __len__(self) -> int:
        return self.maps.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        return torch.as_tensor(self.maps[idx, ...], dtype=torch.float)


if __name__ == "__main__":

    path = "../data/training_30_64.hdf5"
    dataset = Distograms(path)

    print(len(dataset))
    for dist in dataset:
        print(dist.shape)
        break
