# code adopted from https://github.com/tcapelle/torch_moving_mnist/blob/main/torch_moving_mnist/data.py

from functools import partial
from types import SimpleNamespace
from torchvision.datasets import MNIST

import random

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def padding(img_size=64, mnist_size=28): 
    return (img_size - mnist_size) // 2




import math
import os
import random
import json
from PIL import Image
from fastprogress import progress_bar

import cv2
import json
import numpy as np

from tqdm import tqdm

from pathlib import Path

EXTS = ['jpg', 'jpeg', 'png']
class FloatingMNIST:
    def __init__(self, 
                 path=".",  # path to store the MNIST dataset
                 mode="train",
                 write_path=".",
                 affine_params=None, # affine transform parameters, refer to torchvision.transforms.functional.affine
                 num_digits =2, # how many digits to move, random choice between the value provided
                 img_size=64, # the canvas size, the actual digits are always 28x28
                 concat=True, # if we concat the final results (frames, 1, 28, 28) or a list of frames.
                ):
        self.root_dir = path
        self.mode = mode
        path = os.path.join(self.root_dir, self.mode)
        self.files = [str(p) for ext in EXTS for p in Path(f'{path}').glob(f'**/*.{ext}')]
        
        self.classes = os.listdir(path)

        self.write_dir = os.path.join(write_path, self.mode)
        os.makedirs(self.write_dir, exist_ok=True)

        self.total_digits = len(self.files)


        self.affine_params = affine_params
        self.num_digits = num_digits
        self.img_size = img_size
        self.pad = padding(img_size)
        
        self.tf = T.RandomAffine(degrees=(-4, 4), 
                            scale=(0.8, 1.2), 
                            shear=(-3, 3))
        
        

    def random_place(self, img, translate):
        "Randomly place the digit inside the canvas"
        x = random.uniform(translate[0][0], translate[0][1])
        y = random.uniform(translate[1][0], translate[1][1])
        return TF.affine(img, translate=(x,y), angle=0, scale=1, shear=(0,0))
    

    def random_digit(self):
        "Get a random MNIST digit randomly placed on the canvas"
        idx = np.random.randint(0, self.total_digits)
        
        path = self.files[idx]
        digit = int(path.split('/')[-2])

        img = Image.open(path).convert("RGB")
        img = np.array(img)
        img = torch.from_numpy(img*1.0).permute(2, 0, 1)
        img = TF.pad(self.tf(img), padding=self.pad)
        return img, digit
    
    def _one_moving_digit(self, i):
        digit, label = self.random_digit()
        # place randomizer
        digit = self.random_place(digit, self.affine_params[i])
        return digit, label
    
    def getitem(self):
        moving_digits = []
        digits_set = []

        for i in range(self.num_digits):
            img, digit = self._one_moving_digit(i)
            moving_digits.append(img)
            digits_set.append(digit)

        moving_digits = torch.stack(moving_digits)
        combined_digits = moving_digits.max(dim=0)[0]

        return combined_digits, digits_set


    def create_ds(self, N=10000):
        print ('creating ds N=', N)
        
        for i in tqdm(range(N)):
            image, digits = self.getitem()
            digit_str = '_'.join(str(e) for e in digits)
            path = os.path.join(self.write_dir, f'{digit_str}_MNIST_{i}.png')
            image = image.numpy().transpose(1, 2, 0)
            cv2.imwrite(path, image)



if __name__ == '__main__':
    affine_params = []
    affine_params.append(((-20, -10), (-20, 20)))
    affine_params.append(((10, 20), (-20, -14)))
    affine_params.append(((10, 20), (14, 20)))


    floatingmnist = FloatingMNIST(path="/vol/biomedic2/agk21/PhDLogs/datasets/MNIST", 
                 mode="training",
                 write_path="/vol/biomedic3/agk21/datasets/FloatingMNIST3",
                 num_digits =len(affine_params), 
                 affine_params = affine_params,
                 img_size = 64)
    
    floatingmnist.create_ds(N=60000)   


    floatingmnist = FloatingMNIST(path="/vol/biomedic2/agk21/PhDLogs/datasets/MNIST", 
                 mode="testing",
                 write_path="/vol/biomedic3/agk21/datasets/FloatingMNIST3",
                 num_digits =len(affine_params), 
                 affine_params = affine_params,
                 img_size = 64)
    
    floatingmnist.create_ds(N=10000)       
