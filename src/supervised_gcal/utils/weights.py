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
def get_gaussian_weights(shape_source, shape_output, sigma):
    rows_source = shape_source[0]
    rows_output = shape_output[0]
    ans = torch.from_numpy(apply_fn_to_weights_between_maps(rows_source, rows_output, gaussian, sigma=sigma))
    return ans


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
    indexes_columns = []
    nnz_values = []
    for weights in matrix:
        nnz_idx = np.nonzero(weights)[0]
        indexes_columns.append(nnz_idx)
        nnz_values.append(weights[nnz_idx])
    flatten_nnz_values = np.asarray(nnz_values).flatten()

    flatten_idx_cols = np.asarray(indexes_columns).flatten()

    indexes_rows = [[idx_r] * len(r[0]) for idx_r, r in enumerate(indexes_columns)]
    flatten_idx_rows = np.asarray(indexes_rows).flatten()

    tuple_flatten_indexes = np.array([flatten_idx_rows, flatten_idx_cols])
    return torch.sparse.FloatTensor(torch.from_numpy(tuple_flatten_indexes), torch.from_numpy(flatten_nnz_values),
                                    torch.Size([int(matrix.shape[0]), int(matrix.shape[1])]))
