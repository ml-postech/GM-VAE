import numpy as np
import cv2
import glob

from torch.utils import data
from torchvision.transforms import ToTensor
# from torchvision.datasets import ImageFolder


def add_dataset_args(parser):
    parser.add_argument('--data_dir', type=str, default='data/Oxford102')


class Dataset(data.Dataset):
    def __init__(self, args, split='train') -> None:
        super().__init__()

        self.args = args
        self.data_dir = args.data_dir
        self.transform = ToTensor()
        
        self.data = []
        for img_idx, img_dir in enumerate(glob.glob(f'{self.data_dir}/{split}/*.png')):
            img_data = cv2.imread(img_dir)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            self.data.append(img_data)

        self.features = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.transform(x)
        x = (x - 0.5) * 2
        return x

