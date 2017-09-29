from functools import lru_cache

import numpy as np

import torch
from src.supervised_gcal.utils.math import normalize
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights, get_gaussian_weights, \
    dense_weights_to_sparse


class Layer(torch.nn.Module):
    def __init__(self, input_shape, self_shape, name=''):
        super().__init__()
        self.name = name
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(self.self_shape))))
        self.weights = []
        self.epoch = -1
        self.batch_idx = 0
        self._setup_variables()

    def _setup_variables(self):
        raise NotImplementedError

    def custom_sigmoid(self, min_theta, max_theta, activation):
        activation = torch.nn.functional.threshold(activation, min_theta, value=0.0)
        activation.masked_fill_(
            mask=torch.gt(activation, max_theta),
            value=1)

        activation.sub_(min_theta).div_(max_theta - min_theta)
        return activation

    def train(self, mode=True):
        if mode or self.epoch == -1:
            self.epoch += 1
        self.batch_idx = 0
        super().train(mode=mode)


# Outside class only for caching purposes
@lru_cache()
def get_gaussian_weights_variable_wrapped(input_shape, output_shape, sigma, radius, sparse=False):
    weights = normalize(apply_circular_mask_to_weights(get_gaussian_weights(input_shape,
                                                                            output_shape,
                                                                            sigma=sigma),
                                                       radius=radius),
                        axis=1)
    if sparse:
        weights = dense_weights_to_sparse(weights)
    return weights


def get_gaussian_weights_variable(input_shape, output_shape, sigma, radius, sparse=False):
    return get_gaussian_weights_variable_wrapped(input_shape, output_shape, sigma, radius, sparse=sparse).clone()
