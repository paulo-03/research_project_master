"""
Python script to run some trainings
"""

import os
import random
import torch
import numpy as np

from datetime import datetime
from models.cnn_trainer import CnnTrainer
from models.noises import gaussian, pixel_intensity_adaptive


def training_denoiser(model_name: str,
                      num_epochs: int,
                      batch_size: int,
                      add_noise,
                      device: str = 'cuda',
                      seed: int = 42,
                      val_size: float = 0.15,
                      learning_rate: float = 1e-3,
                      weight_decay: float = 1e-2,
                      training_folder: str = 'dataset/training/dl_images',
                      model_saving_root_folder: str = 'models',
                      keep_training_model: str = None):
    """
    params:

    keep_training_model: attribute to keep training a model from its chosen checkpoint (full path must be given)
    """
    # Start by fixing all the seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Prepare the path to store model checkpoints
    model_saving_root_folder = "/".join([model_saving_root_folder, model_name.lower()])
    # Use time stamp to give a unique folder name to the training
    date_str = str(datetime.now()).replace(':', 'h', 1).replace(':', 'm', 1).replace('.', 's', 1)[:-6]
    model_saving_folder = "/".join([model_saving_root_folder, date_str])
    print(model_saving_folder)
    # Make directory to store the model checkpoints
    os.makedirs(model_saving_folder)

    # Fix training parameters
    # Set the data loader and optimizer parameters
    data_kwargs = dict(
        batch_size=batch_size,
        images_folder_path=training_folder,
        add_noise=add_noise
    )

    optimizer_kwargs = dict(
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Initialize the model and its environment
    cnn = CnnTrainer(
        model_name=model_name,
        data_kwargs=data_kwargs,
        num_epochs=num_epochs,
        device=device,
        optimizer_kwargs=optimizer_kwargs,
        model_saving_path=model_saving_folder,
        val_size=val_size
    )

    # If a previous model checkpoint was given, keep training from there
    if keep_training_model is not None:
        cnn.restore_model(model_path=keep_training_model)

    # Start training
    cnn.fit(plot=False)


if __name__ == "__main__":
    training_denoiser(model_name='DnCNN',
                      num_epochs=50,
                      batch_size=15,
                      add_noise=lambda x: gaussian(x, var=20),
                      device='cuda',
                      seed=42,
                      val_size=0.2,
                      learning_rate=2e-3,
                      weight_decay=1e-3,
                      training_folder='data/train/dl_images_texture',
                      model_saving_root_folder='models',
                      keep_training_model=None)
