import numpy as np

import torch


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


def gaussian(x, y, mu_x, mu_y, sigma):
    num = np.power(x - mu_x, 2) + np.power(y - mu_y, 2)
    den = np.power(sigma, 2)
    ans = np.float32(np.exp(-np.divide(num, den)))
    return ans


def get_gaussian(shape, sigma):
    dims = shape[0]
    half_dims = int(np.sqrt(dims))
    dims2 = shape[1]
    half_dims2 = int(np.sqrt(dims2))
    tmp_shape = (half_dims, half_dims, half_dims2, half_dims2)

    mat = np.fromfunction(function=lambda x, y, mu_x, mu_y: gaussian(x, y, mu_x, mu_y, sigma=sigma),
                           shape=tmp_shape, dtype=int)
    mat = np.reshape(mat, shape)
    return mat


def circular_mask(mat, radius):
    if radius is None:
        return mat
    dims = mat.shape[0]
    half_dims = int(np.sqrt(dims))
    dims2 = mat.shape[1]
    half_dims2 = int(np.sqrt(dims2))
    tmp_shape = (half_dims, half_dims, half_dims2, half_dims2)

    # When the distance between the points of the two matrices is greater than radius, set to 0
    mask = np.fromfunction(function=lambda x, y, mu_x, mu_y: mask_distance_gt_radius(x, y, mu_x, mu_y, radius),
                           shape=tmp_shape, dtype=int)
    mask = np.reshape(mask, mat.shape)
    masked_mat = np.ma.masked_where(condition=mask, a=mat)
    return masked_mat.filled(0)


epoch = 1
final_epoch = 8.0 - 1


def linear_neighbors_decay(w, start):
    global epoch
    radius = start + epoch * (1.0 - start) / final_epoch
    w.data = torch.Tensor(normalize(circular_mask(w.data.cpu().numpy(), radius=radius)))
    w.data = w.data.cuda()
    epoch += 1
    return


def kill_neurons(w, threshold):
    return w.masked_fill_(mask=torch.lt(w, threshold), value=0)


def custom_sigmoid(min_theta, max_theta, activation):
    activation = torch.nn.functional.threshold(activation, min_theta, value=0.0)
    activation.masked_fill_(
        mask=torch.gt(activation, max_theta),
        value=1)

    activation.sub_(min_theta).div_(max_theta - min_theta)
    return activation