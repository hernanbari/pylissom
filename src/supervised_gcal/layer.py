import numpy as np

import torch
from src.supervised_gcal.utils.functions import afferent_normalize, piecewise_sigmoid


class Layer(torch.nn.Module):
    def __init__(self, input_shape, self_shape, min_theta, max_theta, strength, radius, afferent_normalization=False,
                 sparse=False, name=''):
        super(Layer).__init__()
        self.radius = radius
        self.sparse = sparse
        self.strength = strength
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.name = name
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(self.self_shape))))
        self.weights = None
        self.epoch = -1
        self.batch_idx = 0
        self.input = None
        self.activation = None
        self.afferent_normalization = afferent_normalization
        self._setup_weights()

    def _setup_weights(self):
        raise NotImplementedError

    def train(self, mode=True):
        if mode or self.epoch == -1:
            self.epoch += 1
        self.batch_idx = 0
        super(self).train(mode=mode)

    def forward(self, stimulus):
        self.input = stimulus
        weighted_sum = self.strength * self.matmul(stimulus.data, self.weights.data)
        if self.afferent_normalization:
            weighted_sum = afferent_normalize(self.radius, self.afferent_normalization_strength, stimulus, weighted_sum)
        activation = piecewise_sigmoid(self.min_theta, self.max_theta, weighted_sum)
        self.activation = torch.autograd.Variable(activation)
        return self.activation

    def matmul(self, vector, matrix):
        if not self.sparse:
            return torch.matmul(vector, matrix)
        else:
            # Pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense,
            # That's why it's reversed
            return torch.matmul(matrix.t(), vector.t()).t()
