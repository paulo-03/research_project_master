"""
Create CNN class where our model will find all their environment to ct_images or be analyzed.
"""
from abc import ABC

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from models.dataset import DeadLeaves
from models.performance_metrics import get_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dncnn.model import DnCNN


class CNN(ABC):
    def __init__(self, model_name: str, device: str) -> None:

        self.model_name = model_name
        self.device = device
        self.train_set_size = 0
        self.val_set_size = 0
        self.cur_epoch = 0
        self.batch_size = 0
        self.training_batch_number = 0
        self.val_batch_number = 0
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
    def test(self, test_ct_path: str, add_noise) -> tuple:
        """Function to predict output from a given input

        :param images: The images for which to do the prediction
        :return: The list of predicted images
        """
        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        # Create DataLoader with all images to predict
        dataset = DeadLeaves(test_ct_path, add_noise)
        dataloader = DataLoader(dataset, self.batch_size)
        batch_number = int(len(dataset) / self.batch_size)

        # Create transformer to transform prediction to pil image
        to_pil_img = transforms.ToPILImage()

        # Initiate dictionnary to store metrics performance within batches
        perf = {
            'mse': {'init': [],
                    'final': []},
            'psnr': {'init': [],
                    'final': []},
            'ssim': {'init': [],
                    'final': []}
        }
        
        # Initiate list to store all images to have a look at the noise reduction made by the model
        data_img = []
        target_img = []
        prediction_img = []
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=batch_number, desc="Progress of predictions"):
            # Move the data to the device
            data, target = data.to(self.device), target.to(self.device)
            # Compute model output
            prediction = self.model(data)
            # Calculate batch metrics
            mse_init, psnr_init, ssim_init = get_metrics(data, target, self.device)
            mse_final, psnr_final, ssim_final = get_metrics(prediction, target, self.device)

            # Store the performance metrics
            perf['mse']['init'].append(mse_init)
            perf['psnr']['init'].append(psnr_init)
            perf['ssim']['init'].append(ssim_init)
            perf['mse']['final'].append(mse_final)
            perf['psnr']['final'].append(psnr_final)
            perf['ssim']['final'].append(ssim_final)
            
            # Store the data, target and prediction in numpy array
            data_img.extend([np.squeeze(numpy_img) for numpy_img in data.cpu().detach().numpy()])
            target_img.extend([np.squeeze(numpy_img) for numpy_img in target.cpu().detach().numpy()])
            prediction_img.extend([np.squeeze(numpy_img) for numpy_img in prediction.cpu().detach().numpy()])

        return perf, data_img, target_img, prediction_img


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

        # Learning rate evolution
        mean_lr = [np.mean(values) for values in self.lr_history]

        # Training PSNR
        mean_psnr_tr = [np.mean(values) for values in self.train_psnr_history]
        quantile_20_psnr_tr = [np.quantile(values, q=0.2) for values in self.train_psnr_history]
        quantile_80_psnr_tr = [np.quantile(values, q=0.8) for values in self.train_psnr_history]
        # Validation PSNR
        mean_psnr_val = [np.mean(values) for values in self.val_psnr_history]
        quantile_20_psnr_val = [np.quantile(values, q=0.2) for values in self.val_psnr_history]
        quantile_80_psnr_val = [np.quantile(values, q=0.8) for values in self.val_psnr_history]

        # Training SSIM
        mean_ssim_tr = [np.mean(values) for values in self.train_ssim_history]
        quantile_20_ssim_tr = [np.quantile(values, q=0.2) for values in self.train_ssim_history]
        quantile_80_ssim_tr = [np.quantile(values, q=0.8) for values in self.train_ssim_history]
        # Validation SSIM
        mean_ssim_val = [np.mean(values) for values in self.val_ssim_history]
        quantile_20_ssim_val = [np.quantile(values, q=0.2) for values in self.val_ssim_history]
        quantile_80_ssim_val = [np.quantile(values, q=0.8) for values in self.val_ssim_history]

        epoch_evol = range(start + 1, end + 1)

        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Plot loss evolution
        plt.figure(figsize=(10, 5))
        
        plt.subplot(2, 2, 1)
        sns.lineplot(x=epoch_evol, y=mean_loss_tr[start:end], label='Train', color='blue')
        sns.lineplot(x=epoch_evol, y=mean_loss_val[start:end], label='Val', color='orange')
        plt.fill_between(epoch_evol, quantile_20_loss_tr[start:end], quantile_80_loss_tr[start:end], 
                         color='blue', alpha=0.2)
        plt.fill_between(epoch_evol, quantile_20_loss_val[start:end], quantile_80_loss_val[start:end], 
                         color='orange', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE-Loss')
        plt.legend()
        
        # Plot lr evolution
        plt.subplot(2, 2, 2)
        sns.lineplot(x=epoch_evol, y=mean_lr[start:end], color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        # Plot psnr evolution
        plt.subplot(2, 2, 3)
        sns.lineplot(x=epoch_evol, y=mean_psnr_tr[start:end], label='Train', color='blue')
        sns.lineplot(x=epoch_evol, y=mean_psnr_val[start:end], label='Val', color='orange')
        plt.fill_between(epoch_evol, quantile_20_psnr_tr[start:end], quantile_80_psnr_tr[start:end], 
                         color='blue', alpha=0.2)
        plt.fill_between(epoch_evol, quantile_20_psnr_val[start:end], quantile_80_psnr_val[start:end], 
                         color='orange', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        
        # Plot ssim evolution
        plt.subplot(2, 2, 4)
        sns.lineplot(x=epoch_evol, y=mean_ssim_tr[start:end], label='Train', color='blue')
        sns.lineplot(x=epoch_evol, y=mean_ssim_val[start:end], label='Val', color='orange')
        plt.fill_between(epoch_evol, quantile_20_ssim_tr[start:end], quantile_80_ssim_tr[start:end], 
                         color='blue', alpha=0.2)
        plt.fill_between(epoch_evol, quantile_20_ssim_val[start:end], quantile_80_ssim_val[start:end], 
                         color='orange', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
