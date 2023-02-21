import os 
import sys 
sys.path.append('../')

from multi_object_datasets import multi_dsprites, clevr_with_masks, cater_with_masks, objects_room, tetrominoes
import tensorflow as tf
import cv2
import json
import numpy as np

from tqdm import tqdm

CONSIDERED = 'dsprites'

def get_constructor(CONSIDERED):
    if CONSIDERED.lower().__contains__('dsprites'):
        dataset_constructor = multi_dsprites.dataset
    elif CONSIDERED.lower().__contains__('clevr'):
        dataset_constructor = clevr_with_masks.dataset
    elif CONSIDERED.lower().__contains__('cater'):
        dataset_constructor = cater_with_masks.dataset
    elif CONSIDERED.lower().__contains__('objects_room'):
        dataset_constructor = objects_room.dataset
    elif CONSIDERED.lower().__contains__('tetrominoes'):
        dataset_constructor = tetrominoes.dataset
    else:
        raise ValueError()

    return dataset_constructor



save_dir = '/vol/biomedic3/agk21/datasets/multi-objects/RawData'
os.makedirs(save_dir, exist_ok=True)


root_dir = '/vol/biomedic3/agk21/datasets/multi-objects/multi-object-datasets'
datasets = os.listdir(root_dir)


for dataset_ in tqdm(datasets):
    if dataset_.__contains__('cater'): continue

    tfrecords_list = os.listdir(os.path.join(root_dir, dataset_))
    save_root = os.path.join(save_dir, dataset_)

    for tfrecord in tqdm(tfrecords_list):
        tfrecord_considered = os.path.join(root_dir, dataset_, tfrecord)

        variant = tfrecord.split('.')[0][len(dataset_) + 1:]
        dataset_constructor = get_constructor(dataset_)
        try:
            dataset = dataset_constructor(tfrecords_path = tfrecord_considered, 
                                            dataset_variant = variant)
            save_root = os.path.join(save_root, variant)
        except:
            dataset = dataset_constructor(tfrecords_path = tfrecord_considered)
        

        os.makedirs(os.path.join(save_root, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'val', 'images'), exist_ok=True)

        os.makedirs(os.path.join(save_root, 'train', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'test', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'val', 'masks'), exist_ok=True)


        dictionary = []
        for id_, data in tqdm(enumerate(dataset), desc=f'Dataset: {dataset_} - Variant: {variant}'):
            info = {}

            p = np.random.uniform(0,1)
            if p <= 0.7: mode = 'train'
            elif 0.7< p <=0.9: mode = 'val'
            else: mode = 'test'

            image = data.pop('image')
            mask = data.pop('mask')

            img_path = os.path.join(save_root, mode, 'images', f'{id_}.png')
            msk_path = os.path.join(save_root, mode, 'masks', f'{id_}')

            cv2.imwrite(img_path, image.numpy())

            os.makedirs(msk_path, exist_ok=True)
            for i in range(mask.shape[0]):
                cv2.imwrite(os.path.join(msk_path, f'{i}.png'), mask.numpy()[i, ...])

            info['image'] = img_path
            info['mask'] = msk_path
            info['mode'] = mode

            for k, v in data.items():
                info[k] = v.numpy().tolist()

            dictionary.append(info)

        with open(os.path.join(save_root, 'properties.json'), 'w') as f:
            json.dump(dictionary, f)



