import os
import glob
import cv2
from typing import Tuple

import numpy as np 
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLASS_NAMES = [
    'butterfly',
    'lion',
    'cat',
    'dog',
    'forest',
    'monkey',
    'rose',
    'sunflower'
]


class DeepClusteringDataset(Dataset):
    def __init__(self,
                data_dir: str,
                transform = None,
                image_size: Tuple[int, int] = (224, 224),
                is_train: bool = False):
        # Commom setting
        self.is_train = is_train
        self.image_size = image_size
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        if self.is_train:
            self.all_samples = glob.glob(os.path.join(data_dir, '*'))
        else:
            self.all_samples = glob.glob(os.path.join(data_dir, '*/*'))
            self.class_names = CLASS_NAMES
        self.num_samples = len(self.all_samples)
        print('Loaded {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        image = cv2.imread(sample)
        image = Image.fromarray(image) 
        
        X = self.transform(image)
        if not self.is_train:
            image_label = sample.split('/')[-2]
            Y = self.class_names.index(image_label)
            return X, Y
        return X
