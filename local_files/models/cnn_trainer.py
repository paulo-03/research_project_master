"""
Class CnnTrainer will inherit from class CNN and complete it for training purpose of our models.
"""

from os import path, listdir

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import DeadLeaves
from torch.utils.data import DataLoader, random_split
from performance_metrics import get_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR

from cnn import CNN


class CnnTrainer(CNN):
    def __init__(self, data_kwargs: dict,
                 num_epochs: int,
                 device: str,
                 optimizer_kwargs: dict,
                 model_name: str = "DnCNN",
                 model_saving_path: str = None,
                 val_size: float = 0.2) -> None:

        super().__init__(model_name, device)

        self.val_size = val_size
        self.batch_size = data_kwargs['batch_size']
        self.num_epochs = num_epochs
        self.model_saving_path = model_saving_path

        # Get loader from disk
        self.training_loader, self.validation_loader = self._get_data_loader_from_disk(
            images_folder_path=data_kwargs['images_folder_path'],
            add_noise=data_kwargs['add_noise']
        )

        # Ensure that the function that got the dataloader correctly set the train and val set size
        assert self.train_set_size != 0, 'The ct_images set size has not been properly set'
        assert self.val_set_size != 0, 'The val set size has not been properly set'

        self.training_batch_number = int(self.train_set_size / self.batch_size)
        self.val_batch_number = int(self.val_set_size / self.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_kwargs)
        self.criterion = nn.MSELoss()
        self.schedular = CosineAnnealingLR(
            self.optimizer,
            T_max=(len(self.training_loader.dataset) * self.num_epochs) // self.training_loader.batch_size
        )

    def fit(self, plot: bool = False) -> None:
        """Compute the training of the model"""

        # Start training
        for epoch in range(self.cur_epoch + 1, self.num_epochs + 1):

            print(f"Start Training Epoch {epoch}/{self.num_epochs}...")
            train_loss, train_mse, train_psnr, train_ssim, lrs = self._train_epoch()
            val_loss, val_mse, val_psnr, val_ssim = self._validate()

            self.cur_epoch += 1  # increment the current epoch counter

            print(
                f"- Average metrics: \n"
                f"\t- train loss={np.mean(train_loss):.2e}, "
                f"train mse={np.mean(train_mse):.2e}, "
                f"learning rate={np.mean(lrs)} \n"
                f"\t- val loss={np.mean(val_loss):.2e}, "
                f"val mse={np.mean(val_mse):.2e} \n"
                f"Finish Training Epoch {epoch} !\n"
            )

            # Store all metrics in array, to plot them at the end of training
            self.train_loss_history.append(train_loss)
            self.train_mse_history.append(train_mse)
            self.train_psnr_history.append(train_psnr)
            self.train_ssim_history.append(train_ssim)
            self.val_loss_history.append(val_loss)
            self.val_mse_history.append(val_mse)
            self.val_psnr_history.append(val_psnr)
            self.val_ssim_history.append(val_ssim)
            self.lr_history.append(lrs)

            # Save the model
            if self.model_saving_path is not None:
                self._save_model()

        # Plot training curves TODO: code function to plot history metric after the all training
        if plot:
            self.print_training_stats()

    def _train_epoch(self):
        """Train one epoch"""

        # Set the model in training mode
        self.model.train()

        # Initiate array to store metrics evolution within batches
        train_loss_history = []
        train_mse_history = []
        train_psnr_history = []
        train_ssim_history = []
        lr_history = []

        for batch_idx, (data, target) in tqdm(enumerate(self.training_loader), total=self.training_batch_number,
                                              desc=f"Progress of training epoch {self.cur_epoch + 1}"):
            # Move the data to the device
            data, target = data.to(self.device), target.to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute model output
            output = self.model(data)
            # Compute loss
            loss = self.criterion(output, target)
            # Backpropagation loss
            loss.backward()
            # Perform an optimizer step
            self.optimizer.step()
            # Perform a learning rate scheduler step
            self.schedular.step()

            # Calculate batch metrics TODO: Find a way to efficiently compute ssim
            mse, pnsr, ssim = get_metrics(output, target, self.device)

            # Store metrics
            train_loss_history.append(loss.item())
            train_mse_history.append(mse)
            train_psnr_history.append(pnsr)
            train_ssim_history.append(ssim)
            lr_history.append(self.schedular.get_last_lr()[0])

        return train_loss_history, train_mse_history, train_psnr_history, train_ssim_history, lr_history

    @torch.no_grad()
    def _validate(self):
        """Compute the accuracy and loss for the validation set"""

        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        # Initiate array to store metrics evolution within batches
        val_loss_history = []
        val_mse_history = []
        val_psnr_history = []
        val_ssim_history = []
        for data, target in tqdm(self.validation_loader, total=self.val_batch_number,
                                 desc=f"Progress of validation metrics epoch {self.cur_epoch + 1}"):
            # Move the data to the device
            data, target = data.to(self.device), target.to(self.device)
            # Compute model output
            output = self.model(data)
            # Compute loss
            loss = self.criterion(output, target)

            # Calculate batch metrics TODO: Find a way to efficiently compute ssim
            mse, psnr, ssim = get_metrics(output, target, self.device)

            # Store metrics
            val_loss_history.append(loss.item())
            val_mse_history.append(mse)
            val_psnr_history.append(psnr)
            val_ssim_history.append(ssim)

        return val_loss_history, val_mse_history, val_psnr_history, val_ssim_history

    def _save_model(self):
        """Save important variable used by the model"""
        state = {
            'model_name': self.model_name,
            'train_set_size': self.train_set_size,
            'val_set_size': self.val_set_size,
            'cur_epoch': self.cur_epoch,
            'batch_size': self.batch_size,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'schedular_state_dict': self.schedular.state_dict(),
            'train_loss_history': self.train_loss_history,
            'train_mse_history': self.train_mse_history,
            'train_psnr_history': self.train_psnr_history,
            'train_ssim_history': self.train_ssim_history,
            'val_loss_history': self.val_loss_history,
            'val_mse_history': self.val_mse_history,
            'val_psnr_history': self.val_psnr_history,
            'val_ssim_history': self.val_ssim_history,
            'lr_history': self.lr_history,
            'training_batch_number': self.training_batch_number,
            'val_batch_number': self.val_batch_number
        }

        torch.save(state, path.join(self.model_saving_path, f'training_save_epoch_{self.cur_epoch}.tar'))

    def _get_data_loader_from_disk(self, images_folder_path: str, add_noise) -> (DataLoader, DataLoader):
        """Helper function to load data into train and validation Dataloader directly from the disk."""

        # Get the dataset
        dataset = DeadLeaves(images_folder_path, add_noise)

        # Calculate the val and train size and split the dataset
        self.val_set_size = int(self.val_size * len(dataset))
        self.train_set_size = len(dataset) - self.val_set_size
        train_dataset, val_dataset = random_split(dataset, lengths=[self.train_set_size, self.val_set_size])

        # Create train and validation Dataloader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader


# Add noise in the CT trainer class.