"""
Helpers functions of basic functions that are shared across scripts or notebooks.
Authors: Raphaël Achddou (PhD) & Paulo Ribeiro (Master)
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import skimage
from scipy.fft import fft2, fftshift
from PIL import Image
from skimage.morphology import opening, closing
from skimage.morphology.footprints import ellipse, disk, square
from skimage.metrics import structural_similarity


class ImageCT:
    def __init__(self, img, arr, path, cat, img_type, dose, patient):
        self.pil: Image.PIL = img
        self.arr: np.ndarray = arr
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
        """Compute the observed color histogram of our CT images sample."""
        # Images are set to uint8 format, then pixels values are in range [0,255]
        pixel_values = np.arange(0, 256)
        # Initialize histogram
        hist = np.zeros(256)
        # Sum all the histograms to compute the mean histogram of our sample
        for img in self.imgs:
            hist += np.array(img.pil.histogram())  # I checked the function, and it will deal correctly our images
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

    def directional_gradient(self, plot: bool = False) -> (list, list, list):
        """Compute the observed directional gradient of our CT images sample."""
        # Initialize histograms
        histo_x = np.zeros(256 * 2)
        histo_y = np.zeros(256 * 2)
        bins = None
        # Sum all the histograms to compute the mean histograms of our sample
        for img in self.imgs:
            pil = img.pil
            array = np.array(pil, dtype=np.int16)
            # Shift array to see diff with neighbors pixels
            array_shifted_x = np.roll(array, shift=-1, axis=1)
            array_shifted_y = np.roll(array, shift=-1, axis=0)
            # Compute gradients
            gradients_x = array - array_shifted_x
            gradients_y = array - array_shifted_y
            # Compute histograms
            histo_x_, bins = np.histogram(gradients_x.flatten(), bins=256 * 2, range=(-256, 256))  # 256*2 because +/-
            histo_y_, _ = np.histogram(gradients_y.flatten(), bins=256 * 2, range=(-256, 256))  # same bins as x
            histo_x += histo_x_
            histo_y += histo_y_
        # Convert frequency to distribution
        histo_x /= histo_x.sum()
        histo_y /= histo_y.sum()
        # Finally, if plot set to True, display a nice plot
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(20, 4))
            axs[0].plot(bins[:-1], histo_x, color='blue')
            axs[0].set_yscale("log")
            axs[0].set_title('Directional Gradient along X-axis')
            axs[0].set_xlabel('Gradient Intensity')
            axs[0].set_ylabel('Frequency (log-scale)')
            axs[0].set_ylim(10 ** (-9), 1)
            axs[1].plot(bins[:-1], histo_y, color='red')
            axs[1].set_yscale("log")
            axs[1].set_title('Directional Gradient along Y-axis')
            axs[1].set_xlabel('Gradient Intensity')
            axs[1].set_ylabel('Frequency (log-scale)')

        return histo_x, histo_y, bins

    def fourier_transformation(self, plot: bool = False) -> np.ndarray:
        """Compute the fourier transformation of our observed CT Images."""
        # Initialize the array that will contain all fourier transform values
        global_fourier = np.zeros((512, 512))
        for img in self.imgs:
            # Compute the 2-dimensional Fourier transform
            f_transform = fft2(img.pil)
            # Shift the zero frequency component to the center of the spectrum
            f_transform_shifted = fftshift(f_transform)
            # Compute the magnitude spectrum (absolute value)
            magnitude_spectrum = np.abs(f_transform_shifted)
            # Sum up the fourier values to compute later the mean
            global_fourier += magnitude_spectrum
        # Compute the means fourier transform values for each pixel
        global_fourier /= self.len

        if plot:
            plt.figure(figsize=(12, 6))
            plt.imshow(np.log(global_fourier))
            plt.title('Mean (Log-values) Fourier Transform of CT Image sample')
            plt.axis('off')
            plt.show()

        return global_fourier

    def opening(self, footprint: skimage.morphology.footprints, keep_class_structure=True):
        """Compute morphological opening of all the images in the GroupImageCT class."""
        opening_img = []
        for img in self.imgs:
            open_img = opening(img.pil, footprint)

            if keep_class_structure:
                open_img = Image.fromarray(open_img.astype(np.uint8))  # convert to pil for data type consistency
                encoded_img = ImageCT(open_img, img.path, img.cat, img.type, img.dose, img.patient)
                opening_img.append(encoded_img)

            else:
                opening_img.append(open_img)

        return GroupImageCT(opening_img) if keep_class_structure else opening_img

    def closing(self, footprint: skimage.morphology.footprints, keep_class_structure=True):
        """Compute morphological closing of all the images in the GroupImageCT class."""
        closing_img = []
        for img in self.imgs:
            close_img = closing(img.pil, footprint)

            if keep_class_structure:
                close_img = Image.fromarray(close_img.astype(np.uint8))  # convert to pil for data type consistency
                encoded_img = ImageCT(close_img, img.path, img.cat, img.type, img.dose, img.patient)
                closing_img.append(encoded_img)

            else:
                closing_img.append(close_img)

        return GroupImageCT(closing_img) if keep_class_structure else closing_img

    def morphological_analysis(self, morphologies=None):
        """Test all th given morphologies and return the object that fits the best the image sample."""
        # Retrieve all original images into array data format
        original = [img.arr for img in self.imgs]
        # Prepare all selected morphologies and add them to the object list used to open or close the images
        results = {}
        for name, morpho in morphologies:
            closing = self.closing(morpho, keep_class_structure=False)
            opening = self.opening(morpho, keep_class_structure=False)
            ssim_closing = structural_similarity(original, closing, data_range=256.0)
            ssim_opening = structural_similarity(original, opening, data_range=256.0)
            results[name] = [ssim_opening, ssim_closing]

        return results
