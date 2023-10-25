import os
import h5py
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


path = 'breakout_states_v2.h5'
raw_data = h5py.File(path, 'r')
raw_imgs = raw_data['states']

data = []
for i in tqdm(range(len(raw_imgs))):
    raw_img = raw_imgs[i, ..., 0]
    img = Image.fromarray(raw_img, 'L')
    img = img.resize((64, 64))

    img = np.asarray(img)
    img = img / 255.
    img[img <= 0.1] = 0.
    img[img > 0.1] = 1.

    if np.max(img) > 0.5:
        data.append(img)

random.shuffle(data)

train = data[:80000]
valid = data[80000:89503]
test = data[89503:]

if not os.path.exists(f'data/Breakout'):
    os.makedirs(f'data/Breakout')

train = np.stack(train, axis=0)
np.save('data/Breakout/train.npy', train)
valid = np.stack(valid, axis=0)
np.save('data/Breakout/valid.npy', valid)
train = np.stack(test, axis=0)
np.save('data/Breakout/test.npy', test)
