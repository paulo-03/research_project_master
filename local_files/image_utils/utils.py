"""
Helpers functions of basic functions that are shared across scripts or notebooks.
Authors: Raphaël Achddou (PhD) & Paulo Ribeiro (Master)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage
from tqdm.notebook import tqdm_notebook
from scipy.fft import fft2, fftshift
from PIL import Image
from skimage.morphology import opening, closing
from skimage.metrics import structural_similarity


class ImageCT:
    def __init__(self, img):
        self.pil: Image = img
        self.width: int = self.pil.width
        self.height: int = self.pil.height


class RealImageCT(ImageCT):
    def __init__(self, img, path, cat, img_type, dose, patient):
        super().__init__(img)
        self.path: str = path
        self.cat: str = cat
        self.type: str = img_type
        self.dose: str = dose
        self.patient: str = patient


class SynthImageCT(ImageCT):
    def __init__(self, leave_img, r_mean, disk_number):
        super().__init__(leave_img)
        self.r_mean = r_mean
        self.disk_number = disk_number


class GroupImageCT:
    def __init__(self, data: list[ImageCT] | list[RealImageCT] | list[SynthImageCT]):
        self.imgs: list[ImageCT] = data
        self.len: int = len(self.imgs)

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
            axs[1].set_ylim(10 ** (-9), 1)

        return histo_x, histo_y, bins

    def fourier_transform_analysis(self, num_rings: int = 10, plot: bool = False) -> float:
        """Process the fourier transform to retrieve our alpha value."""
        # Compute fourier transform
        ft2 = self._fourier_transformation(plot=plot)
        # Create an array with the distances from the center of each pixel
        distances = self._radius_distance_from_center(ft2)
        max_distances = distances.max() + 1  # +1 to be sure the most far away pixels will be selected
        # Compute rings border [1. 2. 3. 4.] and then the rings itself [[1., 2.], [2., 3.], [3., 4.]]
        _, rings_border = np.histogram([0, max_distances], bins=num_rings)
        rings = [[rings_border[idx], rings_border[idx + 1]] for idx in range(len(rings_border) - 1)]
        # Initialize list that will store mean fourier values and radius position
        avg_radius_position = []
        avg_values = []
        for [min_radius, max_radius] in rings:
            # Prepare the mask and apply it to compute only within the pixel in the current radius
            mask = (distances >= min_radius) & (distances < max_radius)
            # Store the position and values
            avg_radius_position.append((min_radius + max_radius) * 0.5)
            avg_values.append(float(np.mean(ft2, where=mask)))

        # Perform linear regression
        log_avg_radius_position, log_avg_values = np.log(avg_radius_position), np.log(avg_values)
        slope, intercept = np.polyfit(log_avg_radius_position, log_avg_values, 1)

        if plot:
            plt.figure(figsize=(7, 3))
            plt.scatter(log_avg_radius_position, log_avg_values, label='empirical average value')
            plt.plot(log_avg_radius_position,
                     [(lambda x: slope * x + intercept)(rad) for rad in log_avg_radius_position],
                     label=fr'linear regression: $\alpha$={slope:.2f}')
            plt.title('Average Fourier Transformation')
            plt.ylabel('Fourier Transform (Log-scale)')
            plt.xlabel('Radius Distance from the Center (Log-scale)')
            plt.legend()
            plt.show()

        return slope

    @staticmethod
    def _radius_distance_from_center(ft2: np.ndarray) -> np.ndarray:
        """Return an array given the distance of each pixel from its center."""
        # Calculate the center of the array
        center_x, center_y = ft2.shape[1] // 2, ft2.shape[0] // 2
        # Generate a grid of coordinates for all points in the array
        y, x = np.ogrid[:ft2.shape[0], :ft2.shape[1]]
        # Calculate the distance of each point from the center
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        return distances

    def _fourier_transformation(self, plot: bool = False) -> np.ndarray:
        """Compute the fourier transformation of our observed CT Images."""
        # Initialize the array that will contain all fourier transform values
        global_fourier = np.zeros(self.imgs[0].pil.size)
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


class GroupReal(GroupImageCT):
    def __init__(self, data: list[RealImageCT]):
        super().__init__(data)

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

        return GroupReal(filtered_img)

    def opening(self, footprint: skimage.morphology.footprints, keep_class_structure=True):
        """Compute morphological opening of all the images in the GroupImageCT class."""
        opening_img = []
        for img in self.imgs:
            open_img = opening(img.pil, footprint)

            if keep_class_structure:
                open_img = Image.fromarray(open_img.astype(np.uint8))  # convert to pil for data type consistency
                encoded_img = RealImageCT(open_img, img.path, img.cat, img.type, img.dose, img.patient)
                opening_img.append(encoded_img)

            else:
                opening_img.append(open_img)

        return GroupReal(opening_img) if keep_class_structure else opening_img

    def closing(self, footprint: skimage.morphology.footprints, keep_class_structure=True):
        """Compute morphological closing of all the images in the GroupImageCT class."""
        closing_img = []
        for img in self.imgs:
            close_img = closing(img.pil, footprint)

            if keep_class_structure:
                close_img = Image.fromarray(close_img.astype(np.uint8))  # convert to pil for data type consistency
                encoded_img = RealImageCT(close_img, img.path, img.cat, img.type, img.dose, img.patient)
                closing_img.append(encoded_img)

            else:
                closing_img.append(close_img)

        return GroupReal(closing_img) if keep_class_structure else closing_img

    def morphological_analysis(self, morphologies: dict = None):
        """Test all th given morphologies and return the object ssim scores within the image sample."""
        # Retrieve all original images into array data format
        originals = [np.array(img.pil) for img in self.imgs]
        # Prepare all selected morphologies and add them to the object list used to open or close the images
        results = {}
        for name, morpho in tqdm_notebook(morphologies.items()):
            closing_imgs = self.closing(morpho, keep_class_structure=False)
            opening_imgs = self.opening(morpho, keep_class_structure=False)
            ssim_closing = 0
            ssim_opening = 0

            for closing_img, opening_img, original in tqdm_notebook(zip(closing_imgs, opening_imgs, originals),
                                                                    total=self.len):
                # Compute the similarity score (1 -> similar and 0 -> no similarity)
                ssim_closing += structural_similarity(original, closing_img, data_range=255)
                ssim_opening += structural_similarity(original, opening_img, data_range=255)

            results[name] = [ssim_opening / self.len, ssim_closing / self.len]

        return pd.DataFrame.from_dict(results, orient='index', columns=['ssim_opening_avg', 'ssim_closing_avg'])


class GroupSynth(GroupImageCT):
    def __init__(self, data: list[SynthImageCT], r_min: int, r_max: int, alpha: float):
        super().__init__(data)
        self.r_min = r_min
        self.r_max = r_max
        self.alpha = alpha


def noise_stat_analysis(full: GroupReal, quarter: GroupReal) -> dict:
    """This function analyses noise between a full and quarter dose in CT Images. It uses a pixel intensity
    approach, meaning it looks at all the pixels with same frequency in full dose images and compute the mean and
    standard deviation of the pixel frequency in the quarter dose images.
    """
    # Initialize dict that will contain all pixels variance
    var_dict = {}
    for intensity in range(256):
        var_dict[intensity] = []  # initialize an empty list to store all variances for pixels from this intensity

    for f_img, q_img in tqdm_notebook(zip(full.imgs, quarter.imgs), total=full.len):
        # Start by validating the image pair
        _valid_img_pair(f_img, q_img)
        # Image pair is valid, then we start the analysis
        var_dict = _noise_analysis_current_img(full_img=f_img, quarter_img=q_img, var_dict=var_dict)

    # Finally compute the mean variance for all pixels intensities
    var_dict_mean = {}
    for intensity in range(256):
        var_dict_mean[intensity] = sum(var_dict[intensity]) / len(var_dict[intensity])

    return var_dict_mean


def _noise_analysis_current_img(full_img: RealImageCT, quarter_img: RealImageCT, var_dict: dict) -> dict:
    """Compute variance of pixel value respectively for all pixel intensity"""
    # Compute mask to know which pixel to drop during analysis
    mask = _create_mask(img=full_img)
    # Start the analysis
    f_img = full_img.pil
    q_img = quarter_img.pil
    for x in range(full_img.width):
        for y in range(full_img.height):
            if mask[x][y]:
                f_intensity = f_img.getpixel((x, y))
                q_intensity = q_img.getpixel((x, y))
                intensity_var = (f_intensity - q_intensity) ** 2
                var_dict[f_intensity].append(intensity_var)

    return var_dict


# TODO: Still need to think about the occlusion in CT Image. Currently setting no mask.
def _create_mask(img: RealImageCT) -> np.ndarray:
    """Compute the mask of the current image. Indeed, real CT Images are not square image, then black pixels are added
    circularly around the image. We don't want to analyze these pixels that do not carry useful information.
    """
    return np.ones(shape=(img.height, img.width))


def _valid_img_pair(full_img: RealImageCT, quarter_img: RealImageCT):
    """Make sure that full and quarter image are their respective equivalent"""
    f_patient, q_patient = full_img.patient, quarter_img.patient
    f_type, q_type = full_img.type, quarter_img.type
    f_cat, q_cat = full_img.cat, quarter_img.cat
    if f_patient != q_patient or f_type != q_type or f_cat != q_cat:
        return ValueError('Images that are trying to be processed for noise analysis are not validate pair.\n'
                          f'full is patient "{f_patient}", cat "{f_cat}" and type "{f_type}"\n'
                          f'quarter is patient "{q_patient}", cat "{q_cat}" and type "{q_type}"')