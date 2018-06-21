"""
Some plotting functions that receive :py:mod:`numpy` matrices as input
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(img, vmin=0, vmax=1):
    r"""Plots a numpy matrix in grayscale normalizing with vmin, vmax"""
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


def plot_list_matrices(imgs):
    for i in imgs:
        plot_matrix(i)


def plot_dict_matrices(imgs):
    """
    Args:
        imgs: Dictionary of {title: numpy matrix} items
    """
    for k, i in imgs.items():
        plt.title(k)
        plot_matrix(i)


def tensor_to_numpy_matrix(tensor, shape):
    """
    Returns: Numpy version of tensor with shape = shape
    """
    tensor = tensor.cpu() if tensor.is_cuda else tensor
    return np.reshape(tensor.numpy(), shape)


def plot_tensor(tensor, shape, vmin=0, vmax=1):
    img = tensor_to_numpy_matrix(tensor, shape)
    plot_matrix(img, vmin, vmax)
