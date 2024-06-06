"""Helpers function to have clean notebook showing only the results without too much code."""

import os
import pydicom
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem, t
from image_utils.utils import *
from tqdm.notebook import tqdm_notebook

from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import v2
from models.cnn_viewer import CnnViewer
from models.noises import gaussian


def load_data(path) -> GroupReal:
    """Helps to load all DICOM images into a list of PIL images."""
    ct_imgs = []
    all_imgs_path = _get_paths(path)

    for img_path in tqdm_notebook(all_imgs_path):
        # Store information of the images that we are loading by looking at its path
        cat, img_type, dose, patient = _defining_img_param(img_path)
        # Load DICOM projection image name in the given path
        dicom_img = pydicom.dcmread(img_path)
        # Convert the format to numpy array type
        pixel_img = dicom_img.pixel_array
        # Rescale the pixels values to the range 0-255
        scaled_pixel_img = pixel_img / pixel_img.max() * 255.0
        # Convert image into ImageCT personal class for simplicity of processing and ease of structure
        pil_img = Image.fromarray(scaled_pixel_img.astype(np.uint8))
        # Create an ImageCT object to store img and its information and append to the other images
        ct_imgs.append(RealImageCT(pil_img, img_path, cat, img_type, dose, patient))

    return GroupReal(ct_imgs)


def _get_paths(path) -> list[str]:
    """Get all path to DICOM images."""
    dicom_img_paths = []
    start_path = '/'.join([path, 'train'])  # data/train
    # filter possible unwanted folder/files
    img_types = [img_type for img_type in os.listdir(start_path) if img_type != '.DS_Store']

    for img_type in img_types:
        type_path = '/'.join([start_path, img_type])  # data/img_cat/img_type
        img_doses = [img_dose for img_dose in os.listdir(type_path) if img_dose != '.DS_Store']

        for img_dose in img_doses:
            dose_path = '/'.join([type_path, img_dose])  # data/img_cat/img_type/img_dose
            img_patients = [img_patient for img_patient in os.listdir(dose_path) if img_patient != '.DS_Store']

            for img_patient in img_patients:
                patient_path = '/'.join([dose_path, img_patient])  # data/img_cat/img_type/img_dose/img_patient
                imgs_full_path = [item for item in sorted(os.listdir(patient_path)) if item.endswith('.IMA')]

                for img_full_path in imgs_full_path:
                    item_path = '/'.join([patient_path, img_full_path])
                    dicom_img_paths.append(item_path)

    return dicom_img_paths


def _defining_img_param(path):
    """Allows to easily retrieve the type of images we are seeing."""
    params = path.split('/')
    cat = params[1]  # train
    img_type = params[2]  # 1mm B30, 1mm D45, 3mm B30, 3mm D45
    dose = 'quarter' if params[3].startswith('quarter') or params[3].startswith('QD') else 'full'  # full or quarter
    patient = params[4]

    return cat, img_type, dose, patient


def store_dl_disk(path, dl_imgs):
    for idx, dl in tqdm_notebook(enumerate(dl_imgs), total=len(dl_imgs),
                                 desc="Writing generated Dead Leaves into disk"):
        img = dl.pil
        img.save(path + f"image_{idx:05}.png")


def display_test_results(perf: dict):
    print("Results (init):\n", 
          f"MSE:  {np.mean(perf['mse']['init']):.2f} ({np.std(perf['mse']['init']):.2f})\n", 
          f"PSNR: {np.mean(perf['psnr']['init']):.2f} ({np.std(perf['psnr']['init']):.2f})\n", 
          f"SSIM: {np.mean(perf['ssim']['init']):.2f} ({np.std(perf['ssim']['init']):.2f})\n")

    print("Results (final):\n", 
          f"MSE:  {np.mean(perf['mse']['final']):.2f} ({np.std(perf['mse']['final']):.2f})\n", 
          f"PSNR: {np.mean(perf['psnr']['final']):.2f} ({np.std(perf['psnr']['final']):.2f})\n", 
          f"SSIM: {np.mean(perf['ssim']['final']):.2f} ({np.std(perf['ssim']['final']):.2f})")


def models_comparaison(models: list[str] = ['dl_no_texture', 'dl_texture', 'ct_baseline_full'],
                       y_label: list[str] = ["None", "DL w/o texture", "DL w/ texture", "CT"],
                       test_ct_path: str = 'data/test/ct_images',
                       add_noise = lambda x: gaussian(x, var=20),
                       save_fig: str = 'models_perf_comparaison',
                       save_data: str = 'models_perf_data.csv',
                       device: str = 'cuda'):
    
    # Compute the prediction of all model from gaussian noised CT images (out-of-sample images)
    models_perf = {'mse': [], 'psnr': [], 'ssim': [], 'model': []}
    models_images = {}
    for model in models:
        # Inititate the model viewer
        cnn = CnnViewer(model_path='models/dncnn/' + model + '/training_save_epoch_50.tar',
                        model_name='DnCNN',
                        device=device)
        # Compute the denoised images
        perf, noised, target, denoised = cnn.test(test_ct_path=test_ct_path, 
                                                  add_noise=add_noise)
        # Store the perf
        models_perf['mse'] += perf['mse']['final']
        models_perf['psnr'] += perf['psnr']['final']
        models_perf['ssim'] += perf['ssim']['final']
        models_perf['model'] += [model] * len(perf['mse']['final'])

        # Store the denoised images
        models_images[model] = {'denoised': denoised, 'noised': noised}

    # Since all models have same groundtruth images, we can store it once.
    models_images['ground_truth'] = target
    
    # Also, store the psnr of noised image to see the improvment performed by each models
    # Increment by the left for plotting purpose
    models_perf['mse'] = perf['mse']['init'] + models_perf['mse']
    models_perf['psnr'] = perf['psnr']['init'] + models_perf['psnr']
    models_perf['ssim'] = perf['ssim']['init'] + models_perf['ssim']
    models_perf['model'] = ['none'] * len(perf['mse']['final']) + models_perf['model']  
    
    # Convert dictionnary into pandas Dataframe to easily use seaborn functions, store in disk the data if requested to
    models_perf = pd.DataFrame(models_perf)
    if save_data is not None:
        models_perf.to_csv(path_or_buf=save_data)

    # Create subplots with 3 rows and 1 column
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
    
    # MSE
    sns.pointplot(x='mse', y='model', data=models_perf, errorbar=('ci', 95), 
                  linestyle='none', marker='|', markersize=15, ax=axs[0])
    sns.stripplot(data=models_perf, x="mse", y="model", dodge=True, alpha=.05, legend=False, ax=axs[0])
    
    # PSNR
    sns.pointplot(x='psnr', y='model', data=models_perf, errorbar=('ci', 95), 
                  linestyle='none', marker='|', markersize=15, ax=axs[1])
    sns.stripplot(data=models_perf, x="psnr", y="model", dodge=True, alpha=.05, legend=False, ax=axs[1])
    
    # SSIM
    sns.pointplot(x='ssim', y='model', data=models_perf, errorbar=('ci', 95), 
                  linestyle='none', marker='|', markersize=15, ax=axs[2])
    sns.stripplot(data=models_perf, x="ssim", y="model", dodge=True, alpha=.05, legend=False, ax=axs[2])
    
    # Customize y-ticks for all subplots
    for ax in axs:
        ax.set_yticks(range(len(y_label)))
        ax.set_yticklabels(y_label, rotation=0)
    
    # Add title and axis labels to the first subplot
    axs[0].set_title("Out-of-Sample Average MSE")
    axs[0].set_xlabel("mse")
    axs[1].set_title("Out-of-Sample Average PSNR")
    axs[1].set_xlabel("psnr")
    axs[2].set_title("Out-of-Sample Average SSIM")
    axs[2].set_xlabel("ssim")
    for ax in axs:
        ax.set_ylabel("datasets")
    
    # Display the plot
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, dpi=1080)
    plt.show()

    return models_perf, models_images
        

def models_evolution(models_reduced: list[str] = ['ct_reduce_1', 'ct_reduce_3', 'ct_reduce_5', 
                                                  'ct_reduce_10', 'ct_reduce_20'],
                     models_mix: list[str] = ['mix_99_dl_texture', 'mix_97_dl_texture', 'mix_95_dl_texture',
                                              'mix_90_dl_texture', 'mix_80_dl_texture'],
                     ct_ratios: list[str] = [1, 3, 5, 10, 20],
                     test_ct_path: str = 'data/test/ct_images',
                     add_noise = lambda x: gaussian(x, var=20),
                     save_fig: str = 'models_perf_evolution',
                     save_data: str = 'models_evolution_data.csv',
                     device: str = 'cuda'):
    
    # Compute the prediction of all model from gaussian noised CT images (out-of-sample images)
    models_perf = {'mse': [], 'psnr': [], 'ssim': [], 'ct_ratio': [], 'type': []}
    for model_reduce, model_mix, ct_ratio in zip(models_reduced, models_mix, ct_ratios):
        # Inititate both models viewer
        cnn_reduce = CnnViewer(model_path='models/dncnn/' + model_reduce + '/training_save_epoch_50.tar',
                               model_name='DnCNN', device=device)
        cnn_mix = CnnViewer(model_path='models/dncnn/' + model_mix + '/training_save_epoch_50.tar',
                            model_name='DnCNN', device=device)
        # Compute both model performances and store the performances
        perf_reduce, _, _, _ = cnn_reduce.test(test_ct_path=test_ct_path, add_noise=add_noise)
        models_perf['mse'] += perf_reduce['mse']['final']
        models_perf['psnr'] += perf_reduce['psnr']['final']
        models_perf['ssim'] += perf_reduce['ssim']['final']
        models_perf['ct_ratio'] += [ct_ratio] * len(perf_reduce['mse']['final'])
        models_perf['type'] += ['reduce'] * len(perf_reduce['mse']['final'])
        
        perf_mix, _, _, _ = cnn_mix.test(test_ct_path=test_ct_path, add_noise=add_noise)
        models_perf['mse'] += perf_mix['mse']['final']
        models_perf['psnr'] += perf_mix['psnr']['final']
        models_perf['ssim'] += perf_mix['ssim']['final']
        models_perf['ct_ratio'] += [ct_ratio] * len(perf_mix['mse']['final'])
        models_perf['type'] += ['mix'] * len(perf_mix['mse']['final'])

    # Convert dictionnary into pandas Dataframe to easily use seaborn functions, store in disk the data if requested to
    models_perf = pd.DataFrame(models_perf)
    if save_data is not None:
        models_perf.to_csv(path_or_buf=save_data)

    # Create subplots with 3 rows and 1 column
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 16))
    
    # MSE
    sns.pointplot(x='ct_ratio', y='mse', hue='type', data=models_perf, errorbar=('ci', 95), 
                  dodge=.4, linestyle='none', marker='_', markersize=15, ax=axs[0])
    sns.stripplot(data=models_perf, x="ct_ratio", y="mse", hue='type', dodge=True, alpha=.05, 
                  legend=False, ax=axs[0])
    
    # PSNR
    sns.pointplot(x='ct_ratio', y='psnr', hue='type', data=models_perf, errorbar=('ci', 95), 
                  dodge=.4, linestyle='none', marker='_', markersize=15, ax=axs[1])
    sns.stripplot(data=models_perf, x="ct_ratio", y="psnr", hue='type', dodge=True, alpha=.05, 
                  legend=False, ax=axs[1])
    
    # SSIM
    sns.pointplot(x='ct_ratio', y='ssim', hue='type', data=models_perf, errorbar=('ci', 95), 
                  dodge=.4, linestyle='none', marker='_', markersize=15, ax=axs[2])
    sns.stripplot(data=models_perf, x="ct_ratio", y="ssim", hue='type', dodge=True, alpha=.05, 
                  legend=False, ax=axs[2])    
    
    # Add title and axis labels to the first subplot
    axs[0].set_title("Out-of-Sample Average MSE")
    axs[0].set_ylabel("mse")
    axs[1].set_title("Out-of-Sample Average PSNR")
    axs[1].set_ylabel("psnr")
    axs[2].set_title("Out-of-Sample Average SSIM")
    axs[2].set_ylabel("ssim")
    for ax in axs:
        ax.set_xlabel("CT ratio")
    
    # Display the plot
    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, dpi=1080)
    plt.show()

    return models_perf


def display_images(images: dict, idx: int, save_fig: str = None):
    
    # Retrieve the images to display
    ground_truth = images['ground_truth'][idx]
    noised = images['ct_baseline_full']['noised'][idx]
    denoised_ct = images['ct_baseline_full']['denoised'][idx]
    denoised_dl = images['dl_no_texture']['denoised'][idx]
    denoised_dl_texture = images['dl_texture']['denoised'][idx]

    # Display results
    fig, axs = plt.subplots(1, 5, figsize=(24, 5))
    axs[0].imshow(noised, cmap='gray')
    axs[0].set_title('Noised Image')
    axs[0].axis('off')  # Turn off axis display
    axs[1].imshow(ground_truth, cmap='gray')
    axs[1].set_title('Ground Truth Image')
    axs[1].axis('off')  # Turn off axis display
    axs[2].imshow(denoised_dl, cmap='gray')
    axs[2].set_title('Denoised Image (DL model)')
    axs[2].axis('off')  # Turn off axis display
    axs[3].imshow(denoised_dl_texture, cmap='gray')
    axs[3].set_title('Denoised Image (DL w/ texture model)')
    axs[3].axis('off')  # Turn off axis display
    axs[4].imshow(denoised_ct, cmap='gray')
    axs[4].set_title('Denoised Image (CT model)')
    axs[4].axis('off')  # Turn off axis display
    if save_fig is not None:
        plt.savefig(save_fig, dpi=1080)
    plt.show()










        
