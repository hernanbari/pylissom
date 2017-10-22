import numpy as np
import torch
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights


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

    @staticmethod
    def piecewise_sigmoid(min_theta, max_theta, activation):
        mask_zeros = torch.le(activation, min_theta)
        mask_ones = torch.ge(activation, max_theta)
        activation.sub_(min_theta).div_(max_theta - min_theta)
        activation.masked_fill_(mask=mask_zeros, value=0)
        activation.masked_fill_(mask=mask_ones, value=1)
        return activation

    def train(self, mode=True):
        if mode or self.epoch == -1:
            self.epoch += 1
        self.batch_idx = 0
        super(self).train(mode=mode)

    def forward(self, stimulus):
        self.input = stimulus
        weighted_sum = self.strength * self.matmul(stimulus.data, self.weights.data)
        if self.afferent_normalization:
            weighted_sum = self.afferent_normalize(stimulus, weighted_sum)
        activation = self.piecewise_sigmoid(self.min_theta, self.max_theta, weighted_sum)
        self.activation = torch.autograd.Variable(activation)
        return self.activation

    def afferent_normalize(self, stimulus, weighted_sum):
        reshaped_input = stimulus.data.repeat(stimulus.data.shape[1], 1)
        masked_input = apply_circular_mask_to_weights(reshaped_input, self.radius)
        sums = masked_input.sum(1).unsqueeze(1).t()
        den = 1 + self.afferent_normalization_strength * sums
        weighted_sum = weighted_sum / den
        return weighted_sum

    def matmul(self, vector, matrix):
        if not self.sparse:
            return torch.matmul(vector, matrix)
        else:
            # Pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense,
            # That's why it's reversed
            return torch.matmul(matrix.t(), vector.t()).t()
