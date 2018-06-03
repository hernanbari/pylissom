import matplotlib.pyplot as plt
import numpy as np


def debug():
    from IPython.core.debugger import Pdb
    Pdb().set_trace()


def plot_matrix(img, vmin=0, vmax=1):
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


def plot_array_matrix(imgs):
    for i in imgs:
        plot_matrix(i)


def plot_dict_matrix(imgs):
    for k, i in imgs.items():
        plt.title(k)
        plot_matrix(i)


def tensor_to_numpy_matrix(tensor, shape):
    tensor = tensor.cpu() if tensor.is_cuda else tensor
    return np.reshape(tensor.numpy(), shape)


def plot_tensor(tensor, shape, vmin=0, vmax=1):
    img = tensor_to_numpy_matrix(tensor, shape)
    plot_matrix(img, vmin, vmax)