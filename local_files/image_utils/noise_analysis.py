"""
Noise analysis functions.
"""

import numpy as np
from skimage.transform import radon, iradon
from skimage.util import crop
import pyxu
from PIL import Image
import pyxu.abc as pxa
import pyxu.operator as pxo
from pyxu.operator.interop import from_source
import skimage.transform as skt


def ct_to_sino(img: np.ndarray, angles: int = 512):
    sino = radon(img, theta=np.linspace(0, 180, angles), circle=False, preserve_range=True)
    return sino


def sino_to_ct(sino: np.ndarray):
    ct = iradon(sino, preserve_range=True)
    # Crop the image to select only the 512x512 square in center
    ct = ct[106:618, 106:618]
    # # Normalize the pixel values to be in the range [0, 1]
    # ct = ct / np.max(ct)
    # # Scale the values to the range [0, 255]
    # ct = ct * 255
    # # Convert back to uint8 data type (0-255 range)
    # ct = ct.astype(np.uint8)
    return ct.astype(np.uint8)
