"""
Class CnnTrainer will inherit from class CNN and complete it for training purpose of our models.
"""

from os import path, listdir

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.cnn import CNN
from models.dataset import DeadLeaves
from models.performance_metrics import get_metrics



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

        print(f"Start Training {self.model_name} model :")

        # Start training
        for epoch in range(self.cur_epoch + 1, self.num_epochs + 1):

            print(f"Epoch {epoch}/{self.num_epochs}...")
            train_loss, train_mse, train_psnr, train_ssim, lrs = self._train_epoch()
            val_loss, val_mse, val_psnr, val_ssim = self._validate()

            self.cur_epoch += 1  # increment the current epoch counter

            print(
                "Train average metrics: \n"
                f"\tloss (MSE)={np.mean(train_loss):.2e}, "
                f"PSNR={np.mean(train_psnr):.2e}, "
                f"SSIM={np.mean(train_ssim):.2e}\n"
    
                "Validation average metrics:\n"
                f"\tloss (MSE)={np.mean(val_loss):.2e}, "
                f"PSNR={np.mean(val_psnr):.2e}, "
                f"SSIM={np.mean(val_ssim):.2e}\n"

                f"learning rate={np.mean(lrs):.2e}"
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

            # Save the model only approximately ten times during the whole training
            multiple = int(self.num_epochs / 10)
            if self.model_saving_path is not None and self.cur_epoch % multiple == 0:
                self._save_model()

        print(f"Finish Training {self.model_name} model !")

        # Plot training curves
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


    def restore_model(self, model_path: str) -> None:
        """Allows to restore model state at a specific saved epoch"""
        super().restore_model(model_path)
        # Retrieve model saving
        model_save = torch.load(model_path, map_location=torch.device(self.device))
        self.optimizer.load_state_dict(model_save['optimizer_state_dict'])
        self.criterion = model_save['criterion']
        self.schedular.load_state_dict(model_save['schedular_state_dict'])


    def _save_model(self):
        """Save important variable used by the model"""
        state = {
            'model_name': self.model_name,
            'train_set_size': self.train_set_size,
            'val_set_size': self.val_set_size,
            'cur_epoch': self.cur_epoch,
            'batch_size': self.batch_size,
            'training_batch_number': self.training_batch_number,
            'val_batch_number': self.val_batch_number,
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
            'lr_history': self.lr_history
        }

        torch.save(state, path.join(self.model_saving_path, f'training_save_epoch_{self.cur_epoch:02}.tar'))

    def _get_data_loader_from_disk(self, images_folder_path: str, add_noise) -> (DataLoader, DataLoader):
        """Helper function to load data into train and validation Dataloadxer directly from the disk."""

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
