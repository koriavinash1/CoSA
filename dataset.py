import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class DataGenerator(Dataset):
    def __init__(self, root, mode='train', resolution=(128, 128)):
        super(DataGenerator, self).__init__()
        
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.root_dir = root     
        self.resolution = resolution
        self.files = os.listdir(os.path.join(self.root_dir, self.mode, 'images'))
        self.img_transform = transforms.Compose([
                                transforms.ToTensor()])


    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, self.mode, 'images', path)).convert("RGB")
        image = image.resize(self.resolution)
        image = self.img_transform(image)
        sample = {'image': image}

        return sample
            
    
    def __len__(self):
        return len(self.files)
