"""
This module contains auxiliary math functions.
"""

import numpy as np
import torch


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


def euclidean_distance_general(x, y):
    """
    This function implements the euclidean distance between two n-dimensional vectors as numpy arrays.
    :param x: First vector (numpy array)
    :param y: Second vector (numpy array)
    :return: euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def gaussian(x, y, mu_x, mu_y, sigma):
    """
    This function implements a circular gaussian function.
    :param x:
    :param y:
    :param mu_x: Center
    :param mu_y: Center
    :param sigma:
    :return: Gaussian
    """
    num = np.power(x - mu_x, 2) + np.power(y - mu_y, 2)
    den = np.power(sigma, 2)
    ans = np.float32(np.exp(-np.divide(num, den)))
    return ans


def normalize(matrix, norm=1, axis=0):
    """
    This function implements a normalization of the row or column vectors of a matrix
    (by default, normalizes the columns and uses norm 1).
    :param matrix: input matrix
    :param norm: Dimension of the norm
    :param axis:  0 is column, 1 is row
    :return: a matrix normalized by columns or rows
    """
    return matrix.div(matrix.norm(p=norm, dim=axis).unsqueeze(1))
