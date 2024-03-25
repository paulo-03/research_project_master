"""
Paython file with function to analyze noise in CT Images from full to quarter dose.
"""
import numpy as np
from tqdm.notebook import tqdm_notebook
from group_imgs_class import GroupReal
from img_class import RealImageCT


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
