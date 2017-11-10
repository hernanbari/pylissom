"""
This module contains auxiliary math functions.
"""

import numpy as np
import torch


# TODO: use pytorch lib
def euclidian_distances(x, y, mu_x, mu_y):
    """
    This function implements the euclidean distance between two 2-dimensional vectors.
    :param x: first element of the first vector
    :param y: second element of the first vector
    :param mu_x: first element of the second vector
    :param mu_y: second element of the second vector
    :return: euclidean distance
    """
    return np.sqrt(((x - mu_x) ** 2 + (y - mu_y) ** 2))


# TODO: use pytorch lib
def euclidean_distance_general(x, y):
    """
    This function implements the euclidean distance between two n-dimensional vectors as numpy arrays.
    :param x: First vector (numpy array)
    :param y: Second vector (numpy array)
    :return: euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def gaussian(x, y, mu_x, mu_y, sigma, sigma_y=None):
    """
    This function implements a circular gaussian function.
    :param x:
    :param y:
    :param mu_x: Center
    :param mu_y: Center
    :param sigma:
    :return: Gaussian
    """
    if sigma_y is None:
        sigma_y = sigma

    x_w = np.divide(x - mu_x, sigma)
    y_h = np.divide(y - mu_y, sigma_y)
    return np.float32(np.exp(-0.5 * x_w * x_w + -0.5 * y_h * y_h))


# TODO: use pytorch lib
def normalize(matrix, norm=1, axis=1):
    """
    This function implements a normalization of the row or column vectors of a matrix
    (by default, normalizes the columns and uses norm 1).
    :param matrix: input matrix
    :param norm: Dimension of the norm
    :param axis:  0 is column, 1 is row
    :return: a matrix normalized by columns or rows
    """
    norm = matrix.norm(p=norm, dim=axis)
    norm = norm if axis == 0 else norm.unsqueeze(1)
    return matrix.div(norm)
