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
            M = np.float32([[1, 0, x], [0, 1, 0]])
        else:
            M = np.float32([[1, 0, 0], [0, 1, x]])
        dst = cv2.warpAffine(bar, M, bar.shape)
        images.append(dst)
    return images


def random_translation(img, x_offset=5, y_offset=5):
    random_x = random.uniform(-x_offset, x_offset)
    random_y = random.uniform(-y_offset, y_offset)
    M = np.float32([[1, 0, random_x], [0, 1, random_y]])
    return cv2.warpAffine(img, M, img.shape)


def generate_horizontal_bar(size):
    line = cv2.line(np.zeros((size, size), dtype=np.float32), (0, size // 2), (size, size // 2), 1, 1)
    return cv2.GaussianBlur(line, (7, 7), 0)


def generate_gaussian(shape, mu_x=0, mu_y=0, sigma_x=1.0, sigma_y=None):
    return np.fromfunction(function=lambda x, y: gaussian(x, y, mu_x, mu_y, sigma=sigma_x, sigma_y=sigma_y),
                           shape=shape, dtype=int)


def rotations(img, num=13):
    return dict([(int(degree), rotate(img, degree, mode='constant').astype(img.dtype)) for degree in
                 np.linspace(0, 180, num)])


def gaussian_generator(size, mu_x, mu_y, sigma_x, sigma_y, orientation):
    g = generate_gaussian((size, size), mu_x, mu_y, sigma_x, sigma_y)
    rot = rotate(g, orientation, mode='constant')
    return rot.astype(g.dtype)


def generate_random_gaussians(size, count):
    return (
        gaussian_generator(size,
                           mu_x=random.uniform(0, size - 1), mu_y=random.uniform(0, size - 1),
                           sigma_x=0.75, sigma_y=2.5,
                           orientation=random.uniform(0, 180)
                           )
        for _ in range(count)
    )


def two_random_gaussians_generator(size, count):
    stream_one = generate_random_gaussians(size, count)
    stream_two = generate_random_gaussians(size, count)
    for one, two in zip(stream_one, stream_two):
        yield combine_matrices(one, two)


def generate_random_faces(size, count):
    return (
        generate_three_dots(size,
                            mu_x=random.uniform(0, size - 1),
                            mu_y=random.uniform(0, size - 1),
                            sigma_x=2 * 1.21 / size / 1.7)
        for _ in range(count)
    )


def generate_three_dots(size, mu_x, mu_y, sigma_x, orientation):
    lefteye = generate_gaussian((size, size), mu_x=mu_x, mu_y=mu_y + 2.5 * 4, sigma_x=sigma_x)
    righteye = generate_gaussian((size, size), mu_x=mu_x, mu_y=mu_y - 2.5 * 4, sigma_x=sigma_x)
    mouth = generate_gaussian((size, size), mu_x=mu_x + 5 * 4, mu_y=mu_y, sigma_x=sigma_x)
    eyes = combine_matrices(lefteye, righteye)
    face = combine_matrices(eyes, mouth)
    return rotate(face, orientation, mode='constant').astype(face.dtype)


def faces_generator(size, num=1):
    while True:
        faces = (
            generate_three_dots(size,
                                mu_x=random.uniform(10, size - 10), mu_y=random.uniform(10, size - 10),
                                sigma_x=3.0,
                                orientation=random.uniform(-15, 15)
                                )
            for _ in range(num)
        )
        combined = combine_matrices(faces)
        yield combined


def combine_matrices(*matrices):
    return np.maximum.reduce(matrices)


def sine_grating(size, freq, phase):
    vals = np.linspace(-np.pi, np.pi, size)
    xgrid, ygrid = np.meshgrid(vals, vals)
    return 0.5 + 0.5 * np.sin(freq * ygrid + phase, dtype=np.float32)