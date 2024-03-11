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


class ImageCT:
    def __init__(self, img, cat, img_type, dose, patient):
        self.img: Image.PIL = img
        self.cat, self.img_type, self.dose, self.patient = cat, img_type, dose, patient


class GroupImageCT:
    def __init__(self, path):
        self.full_path: str = path
        self.cat, self.img_type, self.dose, self.patient = self._defining_folder_param()
        self.imgs: list = self._load_data()
        self.len: int = len(self.imgs)

    def _load_data(self) -> list[ImageCT]:
        """Helps to load all DICOM images into a list of PIL images"""
        # List all projection image name in the given path
        files_name = [img_name for img_name in sorted(os.listdir(self.full_path)) if not img_name.startswith('.')]
        # Load images
        dicom_imgs = [pydicom.dcmread('/'.join([self.full_path, img_name])) for img_name in files_name]
        # Convert the format to numpy array type
        pixel_imgs = [dicom_img.pixel_array for dicom_img in dicom_imgs]
        # Rescale the pixels values to the range 0-255
        scaled_pixel_imgs = [pixel_img / pixel_img.max() * 255.0 for pixel_img in pixel_imgs]
        # Convert image into ImageCT personal class for simplicity of processing and ease of structure
        pil_imgs = [ImageCT(Image.fromarray(scaled_pixel_img.astype(np.uint8)),
                            self.cat, self.img_type, self.dose, self.patient) for scaled_pixel_img in scaled_pixel_imgs]

        return pil_imgs

    def _defining_folder_param(self):
        """Allows to easily retrieve the type of images we are seeing"""
        params = self.full_path.split('/')
        cat = params[1]  # train or test
        img_type = params[2]  # 1mm B30, 1mm D45, 3mm B30, 3mm D45
        dose = 'quarter' if params[3].startswith('quarter') or params[3].startswith('QD') else 'full'  # full or quarter
        patient = params[4]

        return cat, img_type, dose, patient

    def view(self, idxs: list[int], random: bool = False):
        """Allow to quickly and easily have an overview of the CT images of this specific folder"""
        rows_number = len(idxs) // 4    # 4 images showed by row
        fig, axs = plt.subplots(rows_number, 4, figsize=(14, 4 * rows_number))
        fig.suptitle(f'CT Images from {self.full_path}')
        for idx, image in enumerate(idxs):
            row = idx // 4
            col = idx % 4
            axs[row][col].imshow(self.imgs[image], cmap='grey')
            axs[row][col].set_title(f'CT Image number: {image}')
            axs[row][col].grid(None)
        plt.show()

    def color_histogram(self, plot: bool = False) -> [list, list]:
        """Compute the observed color histogram of our CT images sample"""
        # Images are set to uint8 format, then pixels values are in range [0,255]
        pixel_values = np.arange(0, 256)
        # Initialize histogram
        hist = np.zeros(256)
        # Sum all the histograms to compute the mean histogram of our sample
        for img in self.imgs:
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
