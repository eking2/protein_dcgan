import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Distograms(Dataset):
    def __init__(self, path: str, xmax: int):
        super().__init__()

        with h5py.File(path, "r") as f:
            self.maps = f["arr"][:]

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: x / xmax),  # to [0, 1]
            #transforms.Normalize((0.5,), (0.5,))  # to [-1, 1]
            ])

    def __len__(self) -> int:
        return self.maps.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        x = torch.as_tensor(self.maps[idx, ...], dtype=torch.float).unsqueeze(0)
        return self.transforms(x)

if __name__ == "__main__":

    path = "../data/training_30_64.hdf5"
    dataset = Distograms(path, 100)

    print(len(dataset))
    for dist in dataset:
        print(dist.shape)
        print(dist.min())
        print(dist.max())
        break
