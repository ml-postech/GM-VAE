import os
import torch
import itertools
import numpy as np
import torchvision.transforms as T

from tqdm import tqdm
from pathlib import Path
from torch.utils import data
from torchvision.utils import save_image
from torchvision.datasets import Flowers102

def get_dir_name(cnt, data_num):
    if cnt < int(data_num * 5 / 7):
        dir_name = 'train'
    elif int(data_num * 5 / 7) <= cnt < int(data_num * 6 / 7):
        dir_name = 'valid'
    else:
        dir_name = 'test'

    return dir_name

if __name__ == '__main__':
    data_dir = 'data/Oxford102'
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((64, 64))
    ])

    train_data = Flowers102(
        str(data_dir),
        split='train',
        download=True
    )
    test_data = Flowers102(
        str(data_dir),
        split='test',
        download=True
    )

    for dir_name in ['train', 'valid', 'test']:
        if not os.path.exists(f'{data_dir}/{dir_name}'):
            os.makedirs(f'{data_dir}/{dir_name}')

    data_num = len(train_data) + len(test_data)
    cnt = 0
    for idx, img in enumerate(tqdm(train_data)):
        dir_name = get_dir_name(cnt, data_num)
        img = transform(img[0])
        save_image(img, f'{data_dir}/{dir_name}/{cnt:05d}.png')
        cnt += 1
    for idx, img in enumerate(tqdm(test_data)):
        dir_name = get_dir_name(cnt, data_num)
        img = transform(img[0])
        save_image(img, f'{data_dir}/{dir_name}/{cnt:05d}.png')
        cnt += 1


