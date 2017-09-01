import numpy as np


def get_zeros(shape):
    return np.zeros(shape=shape, dtype=np.float32)


def get_ones(shape):
    return np.ones(shape=shape, dtype=np.float32)


def get_uniform(shape):
    random = np.random.uniform(size=shape, low=0.0, high=0.5)
    return random


def mask_distance_gt_radius(x, y, mu_x, mu_y, radius):
    return np.sqrt(((x - mu_x) ** 2 + (y - mu_y) ** 2)) > radius


def normalize(input):
    return np.divide(input, np.linalg.norm(input, ord=1, axis=0))