"""Explanation"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from img_class import *
from tqdm.notebook import tqdm_notebook


def load_data(path) -> list[ImageCT]:
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
        ct_imgs.append(ImageCT(pil_img, scaled_pixel_img, img_path, cat, img_type, dose, patient))

    return ct_imgs


def _get_paths(path) -> list[str]:
    """Get all path to DICOM images."""
    img_cats = ['train', 'test']
    dicom_img_paths = []

    for img_cat in img_cats:
        start_path = '/'.join([path, img_cat])  # data/img_cat
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
    cat = params[1]  # train or test
    img_type = params[2]  # 1mm B30, 1mm D45, 3mm B30, 3mm D45
    dose = 'quarter' if params[3].startswith('quarter') or params[3].startswith('QD') else 'full'  # full or quarter
    patient = params[4]

    return cat, img_type, dose, patient


def fourier_transform_analysis(ft2: np.ndarray, num_rings: int = 10, plot: bool = False) -> float:
    """Process the fourier transform to retrieve our alpha value."""
    # Create an array with the distances from the center of each pixel
    distances = radius_distance_from_center(ft2)
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
        plt.plot(log_avg_radius_position, [(lambda x: slope * x + intercept)(rad) for rad in log_avg_radius_position],
                 label=fr'linear regression: $\alpha$={slope:.2f}')
        plt.title('Average Fourier Transformation')
        plt.ylabel('Fourier Transform (Log-scale)')
        plt.xlabel('Radius Distance from the Center (Log-scale)')
        plt.legend()
        plt.show()

    return slope


def radius_distance_from_center(ft2: np.ndarray) -> np.ndarray:
    """Return an array given the distance of each pixel from its center."""
    # Calculate the center of the array
    center_x, center_y = ft2.shape[1] // 2, ft2.shape[0] // 2
    # Generate a grid of coordinates for all points in the array
    y, x = np.ogrid[:ft2.shape[0], :ft2.shape[1]]
    # Calculate the distance of each point from the center
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    return distances
