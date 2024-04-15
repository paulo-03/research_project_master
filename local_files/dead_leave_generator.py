"""
This code allows you to compute your own dead leaves (dl) images.
"""

import numpy as np
from tqdm.notebook import tqdm_notebook
from PIL import Image, ImageDraw
from image_utils.utils import GroupSynth, SynthImageCT


def dead_leaves_generator(infos: tuple):
    """Generator of dead leaves images following the dead leaves (dl) arguments."""
    # Retrieve the information give to the function
    number, dl_args = infos
    # Start computing the dl images
    dl_imgs = []
    for _ in tqdm_notebook(range(number)):
        dl_imgs.append(_dead_leaves(**dl_args))

    return GroupSynth(dl_imgs, dl_args['r_min'], dl_args['r_max'], dl_args['alpha'])


def _dead_leaves(alpha, r_min, r_max, color_distribution, width, height, max_objects):
    """Function compute one dead leaves image."""
    # Start by up-scaling the size of image (then we resize it to keep a blurry, then more real image)
    width_up = width * 4
    height_up = height * 4

    # Generate the empty/background image
    image = Image.new("L", size=(width_up, height_up), color=0)
    draw = ImageDraw.Draw(image)

    # Initiate an array that will track where circle have been drawn or not and the variable that will stop generation
    drawn_tracker = Image.new("1", size=(width_up, height_up), color=0)
    track = ImageDraw.Draw(drawn_tracker)
    ratio_drawn = 0

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

        # Update  the draw tracker and other metrics
        ratio_drawn = _ratio_drawn(drawn_tracker, track, xy)
        disk_number += 1
        radius_mean_size += radius
        objects += 1

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


def _ratio_drawn(drawn_tracker, tracker, xy):
    """Tracker to know where an object has already been added, then we know when to stop adding objects."""
    # Update potential 0 pixels into ones pixels, telling the tracker that an object have been drawn
    tracker.ellipse(xy=xy, fill=1)
    # Then we check if all values are set to 1, meaning to True
    return np.array(drawn_tracker, dtype=bool).sum()/(512*4*512*4)
