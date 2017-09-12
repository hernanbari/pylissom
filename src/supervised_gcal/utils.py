import numpy as np
from scipy.sparse import csr_matrix

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


def gaussian(x, y, mu_x, mu_y, sigma, radius):
    num = np.power(x - mu_x, 2) + np.power(y - mu_y, 2)
    den = np.power(sigma, 2)
    ans = np.float32(np.exp(-np.divide(num, den)))
    mask = np.sqrt(((x - mu_x) ** 2 + (y - mu_y) ** 2)) > radius
    masked_mat = np.ma.masked_where(condition=mask, a=ans)
    ans = masked_mat.filled(0)
    return ans


def get_gaussian(shape, sigma, radius):
    dims = shape[0]
    half_dims = int(np.sqrt(dims))
    dims2 = shape[1]
    half_dims2 = int(np.sqrt(dims2))
    step = half_dims/half_dims2
    tmp_idx = []
    tmp_map = []
    for i in np.arange(0, half_dims, step):
        for j in np.arange(0, half_dims, step):
            var = np.fromfunction(function=lambda x, y:  gaussian(x, y, int(i), int(j), sigma=sigma, radius=radius),
                           shape=(half_dims, half_dims), dtype=int)
            var = np.reshape(var, dims)
            var = np.divide(var, np.linalg.norm(var, ord=1, axis=0))
            nnz_idx = np.nonzero(var)
            tmp_idx.append(nnz_idx)
            tmp_map.append(var[nnz_idx])
    tmp_map = np.array([x for r in tmp_map for x in r])
    tmp_idx_cols = np.array([x for r in tmp_idx for x in r[0]])
    tmp_idx_rows = [[idx_r]*len(r[0]) for idx_r, r in enumerate(tmp_idx)]
    tmp_idx_rows = np.array([x for r in tmp_idx_rows for x in r])
    idx = np.array([tmp_idx_rows, tmp_idx_cols])
    var_sparse = torch.sparse.FloatTensor(torch.from_numpy(idx), torch.from_numpy(tmp_map), torch.Size([int(dims2), int(dims)]))
    return var_sparse.t()


def circular_mask(mat, radius):
    if radius is None:
        return mat
    dims = mat.shape[0]
    half_dims = int(np.sqrt(dims))
    dims2 = mat.shape[1]
    half_dims2 = int(np.sqrt(dims2))
    step = half_dims/half_dims2
    tmp_map = []
    for i in np.arange(0, half_dims, step):
        for j in np.arange(0, half_dims, step):
            var = np.fromfunction(function=lambda x, y:  mask_distance_gt_radius(x, y, int(i), int(j), radius=radius),
                           shape=(half_dims, half_dims), dtype=int)
            tmp_map.append(np.reshape(var, dims))
    tmp_map = np.asarray(tmp_map)
    tmp_map = np.transpose(tmp_map)
    masked_mat = np.ma.masked_where(condition=tmp_map, a=mat)
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