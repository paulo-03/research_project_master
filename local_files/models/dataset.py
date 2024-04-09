"""
Script to implement our own Dataset for model training.
"""
import torch
from os import path, listdir
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class DeadLeaves(Dataset):
    """Class representing custom Dataset"""

    def __init__(self, images_folder_path: str, add_noise):
        self.images_folder_path = images_folder_path
        self.images_name = list(listdir(images_folder_path))
        self.to_float = v2.ToDtype(torch.float32, scale=False)
        self.add_noise = add_noise

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, item: int):
        image_path = path.join(self.images_folder_path, self.images_name[item])
        # Retrieve the ideal image and add to it some noise
        ideal = read_image(image_path)*255
        noised = self.add_noise(ideal)
        # Transform tensor type to float
        data = self.to_float(noised)
        target = self.to_float(ideal)

        return data, target
