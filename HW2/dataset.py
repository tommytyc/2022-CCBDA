import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class MRI_Dataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.GaussianBlur((3, 3), (1, 2)
                )], p=0.3
            ),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        def readout_img(dir_path):
            for root, _, file_name in os.walk(dir_path):
                img_list = [os.path.join(root, file) for file in file_name if file.endswith('.jpg')]
            return img_list

        self.img_list = readout_img(dir_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img1 = self.transform(Image.open(img_path).convert('RGB'))
        img2 = self.transform(Image.open(img_path).convert('RGB'))
        return img1, img2