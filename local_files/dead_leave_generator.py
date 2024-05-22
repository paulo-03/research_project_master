"""
This code allows you to compute your own dead leaves (dl) images.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm_notebook
from PIL import Image, ImageDraw
from image_utils.utils import GroupSynth, SynthImageCT


def dead_leaves_generator(infos: tuple):
    """Generator of dead leaves images following the dead leaves (dl) arguments."""
    # Retrieve the information give to the function
    number, dl_args, seed = infos
    # Fix the randomness of generation
    np.random.seed(seed)
    # Start computing the dl images
    dl_imgs = []
    for _ in tqdm_notebook(range(number)):
        dl_imgs.append(_dead_leaves(**dl_args))

    return GroupSynth(dl_imgs, dl_args['r_min'], dl_args['r_max'], dl_args['alpha'])


def _dead_leaves(alpha, r_min, r_max, color_distribution, width, height, max_objects, noise_var=600, n=2):
    """Function compute one dead leaves image."""
    # Start by up-scaling the size of image (then we resize it to keep a blurry, then more real image)
    width_up = width * n
    height_up = height * n

    # Generate the empty/background image
    image = Image.new("L", size=(width_up, height_up), color=0)
    draw = ImageDraw.Draw(image)

    # Initiate an array that will track where circle have been drawn or not and the variable that will stop generation
    drawn_tracker = Image.new("1", size=(width_up, height_up), color=0)
    track = ImageDraw.Draw(drawn_tracker)
    ratio_drawn = 0

    # Use PIL image to retrieve the position of the circle just added to know where to add the Brownian noise(texture)
    circle_tracker = Image.new("I", size=(width_up, height_up), color=0)
    bn_track = ImageDraw.Draw(circle_tracker)
    brown_noise_track = 1
    brown_noise_array = np.zeros((width_up, height_up), dtype=np.float64)

    # Also initiate some interesting information
    disk_number = 0
    radius_mean_size = 0
    objects = 0

    while ratio_drawn < 0.998 and objects <= max_objects:
        # Compute the new parameters of the new object that will be drawn
        radius = _object_radius(r_min, r_max, alpha)
        x, y = _object_position(width_up, height_up)
        color = _object_color(color_distribution)

        # Draw the new object (circle here from morphological analysis) on the image
        xy = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(xy=xy,
                     fill=color)

        # Add a brownian noise for texture at each disk, or uncomment below for a general brownian noise
        # brown_noise_array = _brownian_noise_array_per_disk(brown_noise_array=brown_noise_array,
        #                                                    xy=xy,
        #                                                    noise_type=noise_type,
        #                                                    noise_var=noise_var,
        #                                                    circle_tracker=circle_tracker,
        #                                                    bn_track=bn_track,
        #                                                    brown_noise_track=brown_noise_track)

        # Update  the draw tracker and other metrics
        ratio_drawn = _ratio_drawn(drawn_tracker, track, xy, n)
        disk_number += 1
        radius_mean_size += radius
        objects += 1
        brown_noise_track += 1

    # Add a brownian noise general or per disk and gradient layer to the overall synthetic image, just uncomment
    # what you desire
    brown_noise_array = _brownian_noise_array_general(size=brown_noise_array.shape,
                                                      noise_var=noise_var)
    image = _add_brownian_noise(image=image, brown_noise_array=brown_noise_array)

    # image = _gradient_layer(image=image)

    radius_mean_size /= disk_number

    return SynthImageCT(image.resize((width, height)), radius_mean_size, disk_number, ratio_drawn)


def _object_radius(r_min, r_max, alpha) -> int:
    """Simply returns a radius based on a power law distribution of the radius in [r_min, r_max] range."""
    vamin = 1 / (r_max ** (alpha - 1))
    vamax = 1 / (r_min ** (alpha - 1))
    r = vamin + (vamax - vamin) * np.random.random()
    r = int(1 / (r ** (1. / (alpha - 1))))
    return r


def _object_position(width, height) -> [int, int]:
    """Compute randomly the position of the object added to the image. It follows a uniform distribution."""
    x = np.random.randint(low=0, high=width)
    y = np.random.randint(low=0, high=height)
    return x, y


def _object_color(color_distribution) -> int:
    """Select randomly a color for the object. We used real CT images to compute the real color distribution."""
    color_list = range(0, len(color_distribution))
    if color_distribution.sum() > 1:  # make sure to have a distribution and not a simple histogram
        color_distribution /= color_distribution.sum()
    else:
        pass
    return int(np.random.choice(color_list, p=color_distribution))


def _ratio_drawn(drawn_tracker, tracker, xy, n):
    """Tracker to know where an object has already been added, then we know when to stop adding objects."""
    # Update potential 0 pixels into ones pixels, telling the tracker that an object have been drawn
    tracker.ellipse(xy=xy, fill=1)
    # Then we check if all values are set to 1, meaning to True
    return np.array(drawn_tracker, dtype=bool).sum() / (512 * n * 512 * n)


def _add_brownian_noise(image: Image, brown_noise_array: np.ndarray) -> Image:
    """Add the brownian noise that was track during the whole DL generation."""
    # Convert the PIL into array
    image_array = np.array(image, dtype=np.float64)

    # Add the noise
    image_array += brown_noise_array

    # Make sure no values exceed 255
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Convert NumPy array back to PIL image
    image_noised = Image.fromarray(image_array, mode='L')

    return image_noised


def _brownian_noise_array_general(size: tuple, noise_var: int) -> Image:
    """Add one general brownian noise to the dead leave image"""
    brownian_noise = _create_brownian_noise(size=size,
                                            noise_var=noise_var)

    return brownian_noise


def _brownian_noise_array_per_disk(brown_noise_array: np.ndarray, xy: tuple, noise_var: int,
                                   circle_tracker: Image, bn_track: ImageDraw, brown_noise_track: int) -> Image:
    """Add the brownian noise to the current added object in the array to track it. Decided to do this way for
    computational optimization."""
    # Use PIL image to retrieve the position of the circle just added
    bn_track.ellipse(xy=xy,
                     fill=brown_noise_track)
    mask = np.array(circle_tracker) == brown_noise_track

    # Compute the brownian noise to add into image
    brownian_noise = _brownian_noise(size=brown_noise_array.shape,
                                     noise_var=noise_var,
                                     mask=mask)

    # Add the noise to the image
    brown_noise_array[mask] = brownian_noise[mask]

    return brown_noise_array


def _brownian_noise(size: tuple, noise_var: int, mask: np.ndarray) -> np.ndarray:
    """Create an array of image size with noise only where the disk has been added."""
    # First compute the big brownian noise of image size
    brownian_noise = _create_brownian_noise(size=size,
                                            noise_var=noise_var)
    # Then only keep the region of interest (roi) where the disk has been added
    brownian_noise[~mask] = 0

    return brownian_noise


def _create_brownian_noise(size: tuple, noise_var: int) -> np.ndarray:
    """Create a brownian noise to add texture to the synthetic image.
        reference: https://stackoverflow.com/questions/70085015/how-to-generate-2d-colored-noise"""
    # Color factor 
    color_factor = np.random.choice(np.linspace(1, 1.5, 9))
    # Generate white noise image
    whitenoise = np.random.normal(size=size) * (noise_var + 3000 * (color_factor - 1))
    # Compute 2D Fourier transform and shift it to the center
    ft_arr = np.fft.fftshift(np.fft.fft2(whitenoise))
    # Pass the white noise into colored noise
    _x, _y = np.mgrid[0:ft_arr.shape[0], 0:ft_arr.shape[1]]
    f = np.hypot(_x - ft_arr.shape[0] / 2, _y - ft_arr.shape[1] / 2)
    f[f == 0] = 1  # just to delete the 0 in the center
    f = f ** color_factor
    color_ft_arr = np.nan_to_num(ft_arr / f, nan=0, posinf=0, neginf=0)
    # Come back to image level
    color_noise = np.fft.ifft2(np.fft.ifftshift(color_ft_arr)).real

    return color_noise


def _gradient_layer(image: Image):
    """Create the gradient layer"""
    # Retrieve image info
    image_array = np.array(image, dtype=np.float64)
    width, height = image.size
    # Select randomly the gradient pattern
    is_horizontal = np.random.choice([True, False])
    is_left_to_right = np.random.choice([True, False])
    start = np.random.choice(range(-30, -25))
    stop = np.random.choice(range(25, 30))
    # Choose if the gradient is horizontal or vertical
    if is_horizontal:
        gradient_layer = np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        gradient_layer = np.tile(np.linspace(start, stop, width), (height, 1)).T
    # Define the gradient direction, left to right or inverse
    if not is_left_to_right:
        gradient_layer *= -1

    # Add the noise to the image
    image_array += gradient_layer

    # Make sure no values exceed 255
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Convert NumPy array back to PIL image
    image = Image.fromarray(image_array, mode='L')

    return image


def visualize_generation(dead_leaves: SynthImageCT, idx: int):
    """Helper function to visualize the generated image and its used parameters."""
    print("Radius min: ", dead_leaves.r_min)
    print("Radius max: ", dead_leaves.r_max)
    print("Alpha used: ", dead_leaves.alpha)
    print("Number of images: ", dead_leaves.len)
    print(f"----------- idx: {idx} -----------")
    print("Disk number: ", dead_leaves.imgs[idx].disk_number)
    print("Width: ", dead_leaves.imgs[idx].width)
    print("Height: ", dead_leaves.imgs[idx].height)
    print("Radius mean: ", dead_leaves.imgs[idx].r_mean)
    print(f"Ratio drawn: {dead_leaves.imgs[idx].ratio_drawn*100:.2f}%")
    return dead_leaves.imgs[idx].pil
