import numpy as np

import torch


class Layer(torch.nn.Module):
    def __init__(self, input_shape, self_shape, min_theta, max_theta, name=''):
        super().__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.name = name
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(self.self_shape))))
        self.weights = []
        self.epoch = -1
        self.batch_idx = 0
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

