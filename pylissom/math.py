"""
This module contains auxiliary math functions.
"""

import numpy as np
import torch


# TODO: use pytorch lib
def euclidian_distances(x, y, mu_x, mu_y):
    """
    This function implements the euclidean distance between two 2-dimensional vectors.

    Args:
        x: first element of the first vector
        y: second element of the first vector
        mu_x: first element of the second vector
        mu_y: second element of the second vector

    Returns:
        Euclidean distance
    """
    return np.sqrt(((x - mu_x) ** 2 + (y - mu_y) ** 2))


# TODO: use pytorch lib
def euclidean_distance_general(x, y):
    """
    This function implements the euclidean distance between two n-dimensional vectors as numpy arrays.

    Args:
        x: First vector (numpy array)
        y: Second vector (numpy array)

    Returns:
        euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def gaussian(x, y, mu_x, mu_y, sigma, sigma_y=None):
    """
    This function implements a circular gaussian function.

    Args:
        x:
        y:
        mu_x: Center
        mu_y: Center
        sigma:
        sigma_y:

    Returns:
        Gaussian
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

    Args:
        matrix: input matrix
        norm: Dimension of the norm
        axis: 0 is column, 1 is row

    Returns:
        A matrix normalized by columns or rows
    """
    matrix_norms = matrix.norm(p=norm, dim=axis)
    return matrix.div(matrix_norms.expand_as(matrix))
