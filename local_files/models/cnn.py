"""
Create CNN class where our model will find all their environment to ct_images or be analyzed.
"""
from abc import ABC

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

from dncnn.model import DnCNN


class CNN(ABC):
    def __init__(self, model_name: str, device: str) -> None:

        self.model_name = model_name
        self.device = device
        self.train_set_size = 0
        self.val_set_size = 0
        self.cur_epoch = 0
        self.batch_size = 0
        self.optimizer = None
        self.criterion = None
        self.schedular = None

        # Set the architecture of model
        if model_name == "DnCNN":
            self.model = DnCNN().to(torch.device(self.device))
        elif model_name == "FFDNet":
            NotImplemented("Please come later for such model.")
        else:
            raise AttributeError("This model has not been implemented. Please use one of the following models :\n"
                                 "DnCNN or FFDNet")

        # Declare statistics metrics array
        self.train_loss_history = []
        self.train_mse_history = []
        self.train_psnr_history = []
        self.train_ssim_history = []

        self.val_loss_history = []
        self.val_mse_history = []
        self.val_psnr_history = []
        self.val_ssim_history = []

        self.lr_history = []

    @torch.no_grad
    def predict(self, images: list[Image]) -> list[Image]:
        """Function to predict output from a given input

        :param images: The images for which to do the prediction
        :return: The list of predicted images
        """
        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        # Convert PIL image to Tensor
        to_tensor = transforms.ToTensor()
        images = torch.stack([to_tensor(img) for img in images])

        # Create DataLoader with all images to predict
        images_dataset = TensorDataset(images)
        images_dataloader = DataLoader(images_dataset, self.batch_size)

        # Create transformer to transform prediction to pil image
        to_pil_img = transforms.ToPILImage()

        predictions = []
        for img in images_dataloader:
            # Move the data to the device
            img = img[0].to(self.device)
            # Compute model output
            prediction = self.model(img)
            # Store the prediction
            predictions.extend([to_pil_img(numpy_img) for numpy_img in prediction.cpu().detach()])

        return predictions

    def restore_model(self, model_path: str) -> None:
        """Allows to restore model state at a specific saved epoch"""
        # Retrieve model saving
        model_save = torch.load(model_path, map_location=torch.device(self.device))
        # Set all parameters
        self.model_name = model_save['model_name']
        self.train_set_size = model_save['train_set_size']
        self.val_set_size = model_save['val_set_size']
        self.cur_epoch = model_save['cur_epoch']
        self.batch_size = model_save['batch_size']
        self.training_batch_number = model_save['training_batch_number']
        self.val_batch_number = model_save['val_batch_number']
        # Reload all model information
        self.model.load_state_dict(model_save['model_state_dict'])
        self.model = self.model.to(torch.device(self.device))  # ensure model is on correct device
        self.optimizer.load_state_dict(model_save['optimizer_state_dict'])
        self.criterion = model_save['criterion']
        self.schedular.load_state_dict(model_save['schedular_state_dict'])
        # Retrieve training history
        self.train_loss_history = model_save['train_loss_history']
        self.train_mse_history = model_save['train_mse_history']
        self.train_psnr_history = model_save['train_psnr_history']
        self.train_ssim_history = model_save['train_ssim_history']
        self.val_loss_history = model_save['val_loss_history']
        self.val_mse_history = model_save['val_mse_history']
        self.val_psnr_history = model_save['val_psnr_history']
        self.val_ssim_history = model_save['val_ssim_history']
        self.lr_history = model_save['lr_history']

    def print_training_stats(self, start: int = 0, end: int = None):
        """Plot train and val metrics evolution"""

        # Define the range of printing
        if end is None:
            end = self.cur_epoch

        # Compute the means and quantiles of each metrics
        # Training loss
        mean_loss_tr = [np.mean(values) for values in self.train_loss_history]
        quantile_20_loss_tr = [np.quantile(values, q=0.2) for values in self.train_loss_history]
        quantile_80_loss_tr = [np.quantile(values, q=0.8) for values in self.train_loss_history]
        # Validation loss
        mean_loss_val = [np.mean(values) for values in self.val_loss_history]
        quantile_20_loss_val = [np.quantile(values, q=0.2) for values in self.val_loss_history]
        quantile_80_loss_val = [np.quantile(values, q=0.8) for values in self.val_loss_history]

        # Training MSE
        mean_mse_tr = [np.mean(values) for values in self.train_mse_history]
        quantile_20_mse_tr = [np.quantile(values, q=0.2) for values in self.train_mse_history]
        quantile_80_mse_tr = [np.quantile(values, q=0.8) for values in self.train_mse_history]
        # Validation MSE
        mean_mse_val = [np.mean(values) for values in self.val_mse_history]
        quantile_20_mse_val = [np.quantile(values, q=0.2) for values in self.val_mse_history]
        quantile_80_mse_val = [np.quantile(values, q=0.8) for values in self.val_mse_history]

        # Training PSNR
        mean_psnr_tr = [np.mean(values) for values in self.train_psnr_history]
        quantile_20_psnr_tr = [np.quantile(values, q=0.2) for values in self.train_psnr_history]
        quantile_80_psnr_tr = [np.quantile(values, q=0.8) for values in self.train_psnr_history]
        # Validation PSNR
        mean_psnr_val = [np.mean(values) for values in self.val_mse_history]
        quantile_20_psnr_val = [np.quantile(values, q=0.2) for values in self.val_mse_history]
        quantile_80_psnr_val = [np.quantile(values, q=0.8) for values in self.val_mse_history]

        # Training SSIM
        mean_ssim_tr = [np.mean(values) for values in self.train_ssim_history]
        quantile_20_ssim_tr = [np.quantile(values, q=0.2) for values in self.train_ssim_history]
        quantile_80_ssim_tr = [np.quantile(values, q=0.8) for values in self.train_ssim_history]
        # Validation SSIM
        mean_ssim_val = [np.mean(values) for values in self.val_ssim_history]
        quantile_20_ssim_val = [np.quantile(values, q=0.2) for values in self.val_ssim_history]
        quantile_80_ssim_val = [np.quantile(values, q=0.8) for values in self.val_ssim_history]

        epoch_evol = range(start + 1, end + 1)

        fig, ax = plt.subplots(2, 2, figsize=(15, 12))

        # Plot loss evolution
        ax[0][0].plot(epoch_evol, mean_loss_tr[start:end], label='Train', color='blue')
        ax[0][0].plot(epoch_evol, mean_loss_val[start:end], label='Val', color='orange')
        ax[0][0].plot(epoch_evol, quantile_20_loss_tr[start:end], color='blue', linestyle='dotted')
        ax[0][0].plot(epoch_evol, quantile_80_loss_tr[start:end], color='blue', linestyle='dotted')
        ax[0][0].plot(epoch_evol, quantile_20_loss_val[start:end], color='orange', linestyle='dotted')
        ax[0][0].plot(epoch_evol, quantile_80_loss_val[start:end], color='orange', linestyle='dotted')
        ax[0][0].legend()
        ax[0][0].set_xlabel('Epoch')
        ax[0][0].set_ylabel('Loss')

        # Plot mse evolution
        ax[0][1].plot(epoch_evol, mean_mse_tr[start:end], label='Train', color='blue')
        ax[0][1].plot(epoch_evol, mean_mse_val[start:end], label='Val', color='orange')
        ax[0][1].plot(epoch_evol, quantile_20_mse_tr[start:end], color='blue', linestyle='dotted')
        ax[0][1].plot(epoch_evol, quantile_80_mse_tr[start:end], color='blue', linestyle='dotted')
        ax[0][1].plot(epoch_evol, quantile_20_mse_val[start:end], color='orange', linestyle='dotted')
        ax[0][1].plot(epoch_evol, quantile_80_mse_val[start:end], color='orange', linestyle='dotted')
        ax[0][1].legend()
        ax[0][1].set_xlabel('Epoch')
        ax[0][1].set_ylabel('MSE')

        # Plot psnr evolution
        ax[1][0].plot(epoch_evol, mean_psnr_tr[start:end], label='Train', color='blue')
        ax[1][0].plot(epoch_evol, mean_psnr_val[start:end], label='Val', color='orange')
        ax[1][0].plot(epoch_evol, quantile_20_psnr_tr[start:end], color='blue', linestyle='dotted')
        ax[1][0].plot(epoch_evol, quantile_80_psnr_tr[start:end], color='blue', linestyle='dotted')
        ax[1][0].plot(epoch_evol, quantile_20_psnr_val[start:end], color='orange', linestyle='dotted')
        ax[1][0].plot(epoch_evol, quantile_80_psnr_val[start:end], color='orange', linestyle='dotted')
        ax[1][0].legend()
        ax[1][0].set_xlabel('Epoch')
        ax[1][0].set_ylabel('PSNR')

        # Plot ssim evolution
        ax[1][1].plot(epoch_evol, mean_ssim_tr[start:end], label='Train', color='blue')
        ax[1][1].plot(epoch_evol, mean_ssim_val[start:end], label='Val', color='orange')
        ax[1][1].plot(epoch_evol, quantile_20_ssim_tr[start:end], color='blue', linestyle='dotted')
        ax[1][1].plot(epoch_evol, quantile_80_ssim_tr[start:end], color='blue', linestyle='dotted')
        ax[1][1].plot(epoch_evol, quantile_20_ssim_val[start:end], color='orange', linestyle='dotted')
        ax[1][1].plot(epoch_evol, quantile_80_ssim_val[start:end], color='orange', linestyle='dotted')
        ax[1][1].legend()
        ax[1][1].set_xlabel('Epoch')
        ax[1][1].set_ylabel('SSIM')

        plt.show()
