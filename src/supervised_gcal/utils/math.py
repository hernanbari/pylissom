"""
This module contains auxiliary math functions.
"""

import numpy as np
import torch


def euclidian_distances(x, y, mu_x, mu_y):
    """
    This function implements the euclidean distance between two 2-dimensional vectors.
    """
    return np.sqrt(((x - mu_x) ** 2 + (y - mu_y) ** 2))

def euclidean_distance_general(x, y):
    """
    This function implements the euclidean distance between two n-dimensional vectors as numpy arrays.
    """
    return np.sqrt(np.sum((x - y) ** 2))

def gaussian(x, y, mu_x, mu_y, sigma):
    num = np.power(x - mu_x, 2) + np.power(y - mu_y, 2)
    den = np.power(sigma, 2)
    ans = np.float32(np.exp(-np.divide(num, den)))
    return ans


def normalize(matrix, norm=1, axis=0):
    return matrix.div(matrix.norm(p=norm, dim=axis).unsqueeze(1))