"""
This module contains functions that modify the weights of the neural network.
"""

from functools import lru_cache

import numpy as np
import torch

from pylissom.math import gaussian, euclidian_distances


# TODO: use pytorch, not numpy
def apply_fn_to_weights_between_maps(in_features, out_features, fn, **kwargs):
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
    # ASSUMES SQUARE MAPS
    rows_dims_source = int(in_features ** 0.5)
    rows_dims_output = int(out_features ** 0.5)
    dims = in_features
    tmp_map = []
    for i in np.linspace(0, rows_dims_source - 1, rows_dims_output):
        for j in np.linspace(0, rows_dims_source - 1, rows_dims_output):
            weights_matrix = np.fromfunction(function=lambda x, y: fn(x, y, i, j, **kwargs),
                                             shape=(rows_dims_source, rows_dims_source), dtype=int)
            weights_row = np.reshape(weights_matrix, dims)
            tmp_map.append(weights_row)
    tmp_map = np.asarray(tmp_map)
    return tmp_map


@lru_cache(maxsize=0)
# TODO: use clear cache to free memory after layers initializations
def get_gaussian_weights_wrapped(in_features, out_features, sigma):
    ans = torch.from_numpy(apply_fn_to_weights_between_maps(in_features, out_features, gaussian, sigma=sigma))
    return ans


def get_gaussian_weights(in_features, out_features, sigma):
    return get_gaussian_weights_wrapped(in_features, out_features, sigma).clone()


# TODO: use clear cache to free memory
@lru_cache(maxsize=3)
def circular_mask(in_features, out_features, radius, cuda=False):
    distances = apply_fn_to_weights_between_maps(in_features=in_features, out_features=out_features,
                                                 fn=euclidian_distances)
    mask = distances > radius
    torch_mask = torch.from_numpy(mask.astype('uint8'))
    return torch_mask.cuda() if cuda else torch_mask


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
    tensor = circular_mask(matrix.size()[1], matrix.size()[0], radius, matrix.is_cuda)
    matrix.masked_fill_(tensor, 0)
    return matrix


# TODO: remove if not used
def dense_weights_to_sparse(matrix):
    """
    Transforms a torch dense tensor to sparse
    http://pytorch.org/docs/master/sparse.html
    """
    nnz_mask = matrix != 0
    nnz_values = matrix[nnz_mask]
    nnz_indexes = nnz_mask.nonzero()
    params = [nnz_indexes.t(), nnz_values, torch.Size([int(matrix.size()[0]), int(matrix.size()[1])])]
    if matrix.is_cuda:
        return torch.cuda.sparse.FloatTensor(*params)
    else:
        return torch.sparse.FloatTensor(*params)
