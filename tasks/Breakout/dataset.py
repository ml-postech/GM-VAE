import torch
import numpy as np
from pathlib import Path
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def add_dataset_args(parser):
    parser.add_argument('--data_dir', type=str, default='data/Breakout')


class Dataset(data.Dataset):
    def __init__(self, args, split='train') -> None:
        super().__init__()

        self.args = args
        self.transform = ToTensor()
        self.data_dir = Path(args.data_dir)
    
        raw_data = np.load(
            self.data_dir / f'{split}.npy'
        )
        self.data = raw_data[..., None]
        self.features = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.transform(x)
        x[x >= 0.1] = 1.
        x[x < 0.1] = 0.
        return x

    def filter_dataset(self, filter_reward=0):
        self.features = np.log(self.features + 1)
        filter_index  = np.where(self.features >= filter_reward)
        self.data     = self.data[filter_index]
        self.features = self.features[filter_index] - filter_reward
        return
