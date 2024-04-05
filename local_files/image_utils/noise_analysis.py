"""
Noise analysis functions.
"""

import numpy as np
import pyxu
from PIL import Image
import pyxu.abc as pxa
import pyxu.operator as pxo
from pyxu.operator.interop import from_source
import skimage.transform as skt


def ct_to_sino(shape: tuple = (500, 500), angles: int = 90, wsize: int = 5):
    # Radon Operator (imported from `skt` since Pyxu did not ship with a Radon transform at the time of this writing.)
    size = shape[0] * shape[1]
    Radon = from_source(cls=pxa.LinOp,
                        shape=(size, size),
                        apply=lambda _, arr: skt.radon(arr.reshape(shape),
                                                       theta=np.linspace(0, 180, angles),
                                                       circle=True).ravel(),
                        adjoint=lambda _, arr: skt.iradon(arr.reshape(sino.shape),
                                                          filter_name=None,
                                                          circle=True).ravel(),
                        vectorize=["apply", "adjoint"],
                        vmethod="scan",
                        enforce_precision=["apply", "adjoint"])

    # 1D Filtering
    boxcar = np.asarray(sp.signal.get_window("boxcar", wsize))
    boxcar /= wsize
    BoxCar1D = pxo.Stencil(kernel=[boxcar, np.array([1.0])], center=(wsize // 2, 0), arg_shape=shape, )

    # Partial Masking
    Mask = pxo.DiagonalOp(mask.ravel())

    # Tapering
    taper = np.outer(sp.signal.get_window("hamming", shape[0]), np.ones(shape[1]))
    Taper = pxo.DiagonalOp(taper.ravel())

    # Compose operators
    Phi = Taper * Mask * BoxCar1D * Radon

    return NotImplementedError


def sino_to_ct():
    return NotImplementedError
