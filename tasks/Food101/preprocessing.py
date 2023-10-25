import os
import torch
import itertools
import numpy as np
import torchvision.transforms as T

from tqdm import tqdm
from pathlib import Path
from torch.utils import data
from torchvision.utils import save_image
from torchvision.datasets import Food101

if __name__ == '__main__':

    data_dir = Path('data/Food101')

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((64, 64))
    ])

    print('Loading train dataset...')
    train_data = Food101(
        data_dir,
        split='train',
        download=True
    )

    print('Loading test dataset...')
    test_data = Food101(
        data_dir,
        split='test',
        download=True
    )

    for dir_name in ['train', 'valid', 'test']:
        if not os.path.exists(f'{data_dir}/{dir_name}'):
            os.makedirs(f'{data_dir}/{dir_name}')

    print('Saving images...')
    split = int(len(train_data) * 8 / 10)
    for idx, img in enumerate(tqdm(train_data)):
        if idx < split: dir_name = 'train'
        else: dir_name = 'valid'

        img = transform(img[0])
        save_image(img, data_dir / dir_name / f'{idx:05d}.png')

    for idx, img in enumerate(tqdm(test_data)):
        img = transform(img[0])
        save_image(img, data_dir / 'test' / f'{idx:05d}.png')


