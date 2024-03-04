"""
Helpers functions of basic functions that are shared across scripts or notebooks.
Authors: Raphaël Achddou (PhD) & Paulo Ribeiro (Master)
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydicom
from PIL import Image


def load_data(path: str) -> list[Image]:
    """Helps to load all DICOM images into a list of pixel numpy array"""
    # List all projection image name in the given path
    files_name = sorted(os.listdir(path))
    # Load images
    dicom_imgs = [pydicom.dcmread('/'.join([path, img_name])) for img_name in files_name]
    # Convert the format to numpy array type
    pixel_imgs = [dicom_img.pixel_array for dicom_img in dicom_imgs]
    # Rescale the pixels values to the range 0-255
    scaled_pixel_imgs = [pixel_img / pixel_img.max() * 255.0 for pixel_img in pixel_imgs]
    # Convert image into PIL for simplicity of processing
    pil_imgs = [Image.fromarray(scaled_pixel_img.astype(np.uint8)) for scaled_pixel_img in scaled_pixel_imgs]

    return pil_imgs


def color_distribution(imgs: list[Image], plot: bool = False) -> [list, list]:
    """Compute the observed color distribution of our CT images sample"""
    # Images are set to uint8 format, then pixels values are in range [0,255]
    pixel_values = np.arange(0, 256)
    # Initialize histogram
    hist = np.zeros(256)
    # Sum all the histograms to compute the mean histogram of our sample
    for img in imgs:
        hist += np.array(img.histogram())
    # Finally display a nice histogram, if set to True, to visualize result
    if plot:
        plt.figure(figsize=(20, 5))
        # Compute the color distribution observed
        sns.lineplot(hist / hist.sum())
        # Add labels and title
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')
        plt.title('Color Distribution of CT images')
        plt.xticks(range(0, 251, 40))
        plt.ylim(0, 0.15)
        plt.xlim(0, 256)
        # Show the plot
        plt.show()

    return pixel_values, hist
