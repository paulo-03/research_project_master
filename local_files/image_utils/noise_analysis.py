"""
Noise analysis functions.
"""

import numpy as np
from PIL import Image


def de_trending(image: Image, roi_size):
    """Removing the structural noise using a de-trending technique in which a polynomial fitted image Cp(x, y) was
    subtracted from the original CT image C(x, y)."""
    # Convert PIL image to NumPy array
    image_arr = np.array(image)

    # Extract ROI from the center of the image
    center_x = image_arr.shape[0] // 2
    center_y = image_arr.shape[1] // 2
    half_roi = roi_size // 2
    roi = image_arr[center_x - half_roi: center_x + half_roi,
                    center_y - half_roi: center_y + half_roi]

    # Perform surface fitting on the ROI data using polynomial function
    x = np.arange(roi.shape[0])
    y = np.arange(roi.shape[1])
    xx, yy = np.meshgrid(x, y)
    z = roi
    coeffs = np.polyfit((xx, yy), z, 2)  # Adjust degree as needed
    fitted_surface = np.polyval(coeffs, (xx, yy))

    # Subtract fitted surface from original image
    detrended_array = image_arr - fitted_surface

    # Convert NumPy array back to PIL image
    detrended_image = Image.fromarray(detrended_array)

    return detrended_image


def nps_2d(image):
    """Compute the Noise Power Spectrum (NPS) of CT Images."""
    NotImplementedError


def nps_1d(nps2d):
    """1D NPS is obtained by taking several interpolated sampling of NPS2d along the radial direction at varying angles
    (0 to π), and performing the average of them."""
    NotImplementedError


def neq():
    """Estimate the Noise Equivalent Quanta (NEQ) from NPS"""
    NotImplementedError


def mtf():
    """Compute Modulation Transfer Function (MTF) from NPS"""
    NotImplementedError


def noise_model_param():
    """Curve fitting in NEQ estimations to retrieve noise model parameters"""
    NotImplementedError
