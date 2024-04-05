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

    def __init__(self, images_folder_path: str, target_folder_path: str):
        self.images_folder_path = images_folder_path
        self.target_folder_path = target_folder_path
        self.images_name = list(sorted(listdir(images_folder_path)))
        self.targets_name = list(sorted(listdir(target_folder_path)))
        self.to_float = v2.ToDtype(torch.float32, scale=True)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, item: int):
        image_path = path.join(self.images_folder_path, self.images_name[item])
        target_path = path.join(self.target_folder_path, self.targets_name[item])

        data = read_image(image_path)
        target = read_image(target_path)

        data = self.to_float(data)
        target = self.to_float(target)

        return data, target