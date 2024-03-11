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
    def __init__(self, img, path, cat, img_type, dose, patient):
        self.pil: Image.PIL = img
        self.path = path
        self.cat, self.type, self.dose, self.patient = cat, img_type, dose, patient


class GroupImageCT:
    def __init__(self, data: list[ImageCT]):
        self.imgs: list[ImageCT] = data
        self.len: int = len(self.imgs)

    def filter(self, cat: list[str] = None,
               img_type: list[str] = None,
               dose: list[str] = None,
               patient: list[str] = None):
        """Allows  to easily retrieve a subset of CT images.
        :param cat: category of image (train or test)
        :param img_type: environment used during scan (1mm B30, 1mm D45, 3mm B30, 3mm D45)
        :param dose: dose used during scan (full or quarter)
        :param patient: id of patient (L***)
        """
        # If no condition is given for a parameter, keep all of them
        if cat is None:
            cat = ['train', 'test']
        if img_type is None:
            img_type = ['1mm B30', '1mm D45', '3mm B30', '3mm D45']
        if dose is None:
            dose = ['full', 'quarter']
        if patient is None:
            patient = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']

        # Filter the images
        filtered_img = [img for img in self.imgs if
                        img.cat in cat and
                        img.type in img_type and
                        img.dose in dose and
                        img.patient in patient]

        return GroupImageCT(filtered_img)

    def color_histogram(self, plot: bool = False) -> [list, list]:
        """Compute the observed color histogram of our CT images sample"""
        # Images are set to uint8 format, then pixels values are in range [0,255]
        pixel_values = np.arange(0, 256)
        # Initialize histogram
        hist = np.zeros(256)
        # Sum all the histograms to compute the mean histogram of our sample
        for img in self.imgs:
            hist += np.array(img.pil.histogram())
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

    def directional_gradient(self, plot: bool = False) -> [list, list]:
        """Compute the observed directional gradient of our CT images sample"""
        # Initialize histograms
        histo_x = np.zeros(256)
        histo_y = np.zeros(256)
        bins = None
        # Sum all the histograms to compute the mean histograms of our sample
        for img in self.imgs:
            pil = img.pil
            array = np.array(pil, dtype=np.int16)
            # shift array to see diff with neighbors pixels
            array_shifted_x = np.roll(array, shift=-1, axis=1)
            array_shifted_y = np.roll(array, shift=-1, axis=0)
            # compute gradients
            gradients_x = array - array_shifted_x
            gradients_y = array - array_shifted_y
            # Compute histograms
            histo_x_, bins = np.histogram(gradients_x.flatten(), bins=256, range=(-256, 256))
            histo_y_, _ = np.histogram(gradients_y.flatten(), bins=256, range=(-256, 256))  # same bins as x
            histo_x += histo_x_
            histo_y += histo_y_
        # Convert frequency to distribution
        histo_x /= histo_x.sum()
        histo_y /= histo_y.sum()
        # Finally, if plot set to True, display a nice plot
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(20, 4))
            axs[0].plot(bins[:-1], histo_x, color='blue')
            axs[0].set_title('Directional Gradient along X-axis')
            axs[0].set_xlabel('Gradient Intensity')
            axs[0].set_ylabel('Frequency')
            axs[1].plot(bins[:-1], histo_y, color='red')
            axs[1].set_title('Directional Gradient along Y-axis')
            axs[1].set_xlabel('Gradient Intensity')
            axs[1].set_ylabel('Frequency')

        return histo_x, histo_y, bins
