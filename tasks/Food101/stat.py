import torch
import itertools
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils import data
from torchvision.utils import save_image
from torchvision.datasets import Food101

if __name__ == '__main__':

    data_dir = './datasets/Food101'

    # transform = T.Compose([
    #                 T.ToTensor(),
    #                 T.Resize((64, 64))
    #             ])

    train_data = Food101(
            str(data_dir),
            split='train',
            download=True
    )
    print('loaded Food101')

    imgs = [] # PIL images
    split = int(len(train_data) * 8 / 10)
    for idx, img in enumerate(tqdm(train_data)):
        if idx < split: 
            # img = transform(img[0])
            img = img[0].resize((64, 64))
            imgs.append(np.array(img))
            # print(imgs[-1])


    print(np.mean(imgs), np.std(imgs))
