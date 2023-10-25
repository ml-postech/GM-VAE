import numpy as np
import cv2
import glob

from torch.utils import data
from torchvision.transforms import ToTensor
# from torchvision.datasets import ImageFolder

if __name__ == "__main__": 
    split = 'train'
    data_dir = './datasets/CUB'

    data = []
    # string = 'train' if is_train else 'test'
    print(f'Start loading CUB {split} dataset...')
    # print(self.data_dir)
    for img_idx, img_dir in enumerate(glob.glob(f'{data_dir}/{split}/*.png')):
        img_data = cv2.imread(img_dir)

        data.append(img_data)
    print(f'Finished loading CUB {split} dataset!\n')

    split = int(len(data) * 8 / 10)
    print(np.mean(data[:split]), np.std(data[:split]))
