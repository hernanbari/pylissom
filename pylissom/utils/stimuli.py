"""
Provides several functions that create and manipulate matrices representing different stimuli, mainly guassians disks
TODO: only use cv2 or scikit-image, not both
"""

import random

import cv2
import numpy as np
from skimage.transform import rotate

from pylissom.math import gaussian


def translate(bar, step=1, over_x=True):
    images = []
    cols = bar.shape[0]
    half_cols = cols // 2
    for x in range(-half_cols, half_cols + 1 if cols % 2 == 1 else half_cols, step):
        if over_x:
            matrix = np.float32([[1, 0, x], [0, 1, 0]])
        else:
            matrix = np.float32([[1, 0, 0], [0, 1, x]])
        dst = cv2.warpAffine(bar, matrix, bar.shape)
        images.append(dst)
    return images


def random_translation(img, x_offset=5, y_offset=5):
    random_x = random.uniform(-x_offset, x_offset)
    random_y = random.uniform(-y_offset, y_offset)
    matrix = np.float32([[1, 0, random_x], [0, 1, random_y]])
    return cv2.warpAffine(img, matrix, img.shape)


def generate_horizontal_bar(size):
    line = cv2.line(np.zeros((size, size), dtype=np.float32), (0, size // 2), (size, size // 2), 1, 1)
    return cv2.GaussianBlur(line, (7, 7), 0)


def generate_gaussian(shape, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=None):
    return np.fromfunction(function=lambda x, y: gaussian(x, y, mu_x, mu_y, sigma=sigma_x, sigma_y=sigma_y),
                           shape=shape, dtype=int)


def rotations(img, num=13):
    r"""
    Returns: Returns a dictionary of len 180 / num, representing {rotation_degrees: rotated_img}
    """
    return {(int(degree), rotate(img, degree, mode='constant').astype(img.dtype)) for degree in
                 np.linspace(0, 180, num)}


def gaussian_generator(size, mu_x, mu_y, sigma_x, sigma_y, orientation):
    r"""
    Args:
        orientation: It's actually redundant because orientation is a function of the sigmas, but make it easier to use
    Returns: A numpy matrix representing a gaussian of shape = (size, size)
    """
    g = generate_gaussian((size, size), mu_x, mu_y, sigma_x, sigma_y)
    rot = rotate(g, orientation, mode='constant')
    return rot.astype(g.dtype)


def generate_random_gaussian(size):
    r"""
    Returns: A numpy matrix with a random gaussian of shape = (size, size)
    """
    return gaussian_generator(size,
                              mu_x=random.uniform(0, size - 1), mu_y=random.uniform(0, size - 1),
                              sigma_x=0.75, sigma_y=2.5,
                              orientation=random.uniform(0, 180)
                              )


def random_gaussians_generator(size, gaussians=1):
    r"""
    Args:
        size: img will have shape = (size, size)
        gaussians: How many gaussians per matrix

    Returns: Yields a squared numpy matrix with gaussians disks
    """
    while True:
        yield combine_matrices(*(generate_random_gaussian(size) for _ in range(gaussians)))


def generate_random_faces(size):
    # TODO: sigma should be a function of size
    return generate_three_dots(size,
                               mu_x=random.uniform(0, size - 1),
                               mu_y=random.uniform(0, size - 1),
                               # sigma_x=2 * 1.21 / size / 1.7,
                               sigma_x=1.2,
                               # sigma_x=2.0,
                               orientation=random.uniform(-15, 15)
                               )


def generate_three_dots(size, mu_x, mu_y, sigma_x, orientation):
    # TODO: The spacing between eyes and mouth should be a function of the image size or sigma
    r"""
    Returns: A numpy matrix with 3 gaussian disks representing a face
    """
    lefteye = generate_gaussian((size, size), mu_x=mu_x, mu_y=mu_y + 2.5 * 2, sigma_x=sigma_x)
    righteye = generate_gaussian((size, size), mu_x=mu_x, mu_y=mu_y - 2.5 * 2, sigma_x=sigma_x)
    mouth = generate_gaussian((size, size), mu_x=mu_x + 5 * 2, mu_y=mu_y, sigma_x=sigma_x)
    eyes = combine_matrices(lefteye, righteye)
    face = combine_matrices(eyes, mouth)
    return rotate(face, orientation, mode='constant').astype(face.dtype)


def faces_generator(size, num=1):
    r"""
    Args:
        size: img will have shape = (size, size)
        gaussians: How many faces per matrix

    Returns: Yields a squared numpy matrix with 3-gaussians faces
    """
    while True:
        yield combine_matrices(*(generate_random_faces(size) for _ in range(num)))


def combine_matrices(*matrices):
    r"""
    Returns: Merges matrices in one matrix using the maximum values in each pixel
    """
    return np.maximum.reduce(matrices)


def sine_grating(size, freq, phase):
    vals = np.linspace(-np.pi, np.pi, size)
    xgrid, ygrid = np.meshgrid(vals, vals)
    return 0.5 + 0.5 * np.sin(freq * ygrid + phase, dtype=np.float32)
