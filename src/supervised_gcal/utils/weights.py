"""
This module contains functions that modify the weights of the neural network.
"""

import numpy as np

import torch
from functools import lru_cache
from src.supervised_gcal.utils.math import gaussian, euclidian_distances


def apply_fn_to_weights_between_maps(rows_dims_source, rows_dims_output, fn, **kwargs):
    """
    The goal of this function is to apply a function fn, to all the elements of an array of dimension
    rows_dims_source x rows_dims_source (the lower array) centered on an element of the superior array.
    The elements of the array would be the weights of the superior layer, with the inferior layer, i.e.,
    it modifies the weights of each one of the neurons of the superior layer, with respect to all the neurons
    of the inferior layer.
    :param rows_dims_source: The dimension of the inferior array.
    :param rows_dims_output: The dimension of the superior array.
    :param fn: The function applied to the weights.
    :param kwargs: The actual weights.
    :return: An array containing the new weights of the superior layer.
    PROBLEMAS? OJO QUE EL STEP PUEDE SER UN FLOAT
    """
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
    uniform = torch.FloatTensor(ans.size()).uniform_(0, 1)
    ans = ans * uniform
    return ans


def get_gaussian_weights(shape_source, shape_output, sigma):
    return get_gaussian_weights_wrapped(shape_source, shape_output, sigma).clone()


def apply_circular_mask_to_weights(matrix, radius):
    """
    This functions applies a circular mask to a matrix of weights. The weights of the neurons that are
    more far than the radius, will have its weight set to zero.
    :param matrix: Tensor of weights. The rows are the neurons. The columns the weights of the neuron.
    :param radius: The radius of neighborhood.
    :return:
    """
    if radius is None:
        return matrix
    orig_rows_dims_source = int(matrix.shape[1] ** 0.5)
    orig_rows_dims_output = int(matrix.shape[0] ** 0.5)
    distances = apply_fn_to_weights_between_maps(orig_rows_dims_source, orig_rows_dims_output, euclidian_distances)
    mask = distances > radius
    tensor = torch.from_numpy(mask.astype('uint8'))
    matrix.masked_fill_(tensor.cuda() if matrix.is_cuda else tensor, 0)
    return matrix


def dense_weights_to_sparse(matrix):
    """
    Transforms a torch dense tensor to sparse
    http://pytorch.org/docs/master/sparse.html
    """
    nnz_mask = matrix != 0
    nnz_values = matrix[nnz_mask]
    nnz_indexes = nnz_mask.nonzero()
    params = [nnz_indexes.t(), nnz_values, torch.Size([int(matrix.shape[0]), int(matrix.shape[1])])]
    if matrix.is_cuda:
        return torch.cuda.sparse.FloatTensor(*params)
    else:
        return torch.sparse.FloatTensor(*params)
