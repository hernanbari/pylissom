from functools import lru_cache

import numpy as np

import torch
from src.supervised_gcal.utils.math import gaussian, euclidian_distances


def apply_fn_to_weights_between_maps(rows_dims_source, rows_dims_output, fn, **kwargs):
    dims = rows_dims_source ** 2
    step = rows_dims_source / rows_dims_output
    tmp_map = []
    for i in np.arange(0, rows_dims_source, step):
        for j in np.arange(0, rows_dims_source, step):
            weights_matrix = np.fromfunction(function=lambda x, y: fn(x, y, int(i), int(j), **kwargs),
                                             shape=(rows_dims_source, rows_dims_source), dtype=int)
            weights_row = np.reshape(weights_matrix, dims)
            tmp_map.append(weights_row)
    tmp_map = np.asarray(tmp_map)
    return tmp_map


@lru_cache()
def get_gaussian_weights_wrapped(shape_source, shape_output, sigma):
    rows_source = shape_source[0]
    rows_output = shape_output[0]
    ans = torch.from_numpy(apply_fn_to_weights_between_maps(rows_source, rows_output, gaussian, sigma=sigma))
    return ans


def get_gaussian_weights(shape_source, shape_output, sigma):
    return get_gaussian_weights_wrapped(shape_source, shape_output, sigma).clone()


def apply_circular_mask_to_weights(matrix, radius):
    if radius is None:
        return matrix
    orig_rows_dims_source = int(matrix.shape[1] ** 0.5)
    orig_rows_dims_output = int(matrix.shape[0] ** 0.5)
    distances = apply_fn_to_weights_between_maps(orig_rows_dims_source, orig_rows_dims_output, euclidian_distances)
    mask = distances > radius
    matrix.masked_fill_(torch.from_numpy(mask.astype('uint8')), 0)
    return matrix


def dense_weights_to_sparse(matrix):
    """
    Transforms a torch dense tensor to sparse
    http://pytorch.org/docs/master/sparse.html
    """
    nnz_mask = matrix != 0
    nnz_values = matrix[nnz_mask]
    nnz_indexes = nnz_mask.nonzero()
    return torch.sparse.FloatTensor(nnz_indexes.t(), nnz_values,
                                    torch.Size([int(matrix.shape[0]), int(matrix.shape[1])]))
