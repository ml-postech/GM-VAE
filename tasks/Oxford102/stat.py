import torch
import itertools
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils import data
from torchvision.utils import save_image
from torchvision.datasets import Flowers102

if __name__ == '__main__':

    data_dir = './datasets/Oxford102'

    # transform = T.Compose([
    #                 T.ToTensor(),
    #                 T.Resize((64, 64))
    #             ])

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
    print('loaded Oxford102')

    imgs = [] # PIL images
    split = int((len(train_data) + len(test_data)) * 5 / 7)
    for idx, img in enumerate(tqdm(train_data)):
        if len(imgs) < split: 
            # img = transform(img[0])
            img = img[0].resize((64, 64))
            imgs.append(np.array(img))
            # print(imgs[-1])
        else:
            break
    for idx, img in enumerate(tqdm(test_data)):
        if len(imgs) < split: 
            # img = transform(img[0])
            img = img[0].resize((64, 64))
            imgs.append(np.array(img))
        else:
            break

    print(np.mean(imgs), np.std(imgs))
