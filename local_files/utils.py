"""Explanation"""

import os
import pydicom
import torch

from image_utils.utils import *
from tqdm.notebook import tqdm_notebook


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
