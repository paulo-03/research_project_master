"""
Helpers functions of basic functions that are shared across scripts or notebooks.
Authors: Raphaël Achddou (PhD) & Paulo Ribeiro (Master)
"""
from PIL import Image


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
