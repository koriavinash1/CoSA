import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import json
from pathlib import Path






def get_paths_with_properties_CLEVR(root_path, mode, max_objects=7):
    size_mapping = {'large': 0, 'small': 1}
    shape_mapping ={'sphere': 0, 'cube': 1, 'cylinder':2}
    color_mapping = {'red':0, 'green': 1, 'blue': 2, 'yellow': 3, 'gray': 4, 'cyan': 5, 'brown': 6, 'purple': 7}
    material_mapping = {'rubber': 0, 'metal': 1}
    data = json.load(open(os.path.join(root_path, mode, f'CLEVR_HANS_scenes_{mode}.json'), 'r'))['scenes']
    paths = []; properties = []
    for data_info in data:
        paths.append(os.path.join(root_path, mode, 'images', data_info['image_filename']))
        
        objects = []
        for object_info in data_info['objects']:
            object_property = np.eye(len(size_mapping))[size_mapping[object_info['size']]]
            object_property = np.concatenate([object_property, 
                            np.eye(len(shape_mapping))[shape_mapping[object_info['shape']]]])
            object_property = np.concatenate([object_property, 
                            np.eye(len(color_mapping))[color_mapping[object_info['color']]]])
            object_property = np.concatenate([object_property, 
                            np.eye(len(material_mapping))[material_mapping[object_info['material']]]])
            object_property = np.concatenate([object_property, [1]])
            # object_property = np.concatenate([object_property, object_info['3d_coords']])
            objects.append(object_property)

        for _ in range(max_objects - len(objects)):
            objects.append(np.zeros_like(object_property))

        properties.append(np.array(objects, dtype='float32')[:max_objects, ...])
    
    return paths, properties




# 3D shapes dataset
# Shape stack
# bitmoji
# flying mnist
EXTS = ['jpg', 'jpeg', 'png']
class DataGenerator(Dataset):
    def __init__(self, root, 
                        mode='train',
                        max_objects=10,
                        properties=False,
                        class_info=False,
                        resolution=(128, 128)):
        super(DataGenerator, self).__init__()
        
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.root_dir = root     
        self.resolution = resolution
        self.properties = properties
        self.class_info = class_info

        if self.properties:
            self.files, self.properties = get_paths_with_properties_CLEVR(root, mode, max_objects)
        else:
            path = os.path.join(self.root_dir, self.mode)
            self.files = [str(p) for ext in EXTS for p in Path(f'{path}').glob(f'**/*.{ext}')]
        

        # self.files = self.files[:100]
        # self.img_transform = transforms.Compose([
        self.img_transform = torch.nn.Sequential(
                                        transforms.Resize(resolution),
                                        # transforms.RandomAffine(15, 
                                        #                         translate=(0.15, 0.15), 
                                        #                         scale=(0.8, 1.2), 
                                        #                         shear=0.0),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    )
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        image = self.to_tensor(self.img_transform(image))
        sample = {'image': image}

        if self.properties:
            property_info = self.properties[index]
            property_info = torch.from_numpy(property_info)
            sample['properties'] = property_info

        if self.class_info:
            target = int(path.split('_')[2])
            sample['target'] = target   

        return sample
            
    
    def __len__(self):
        return len(self.files)
