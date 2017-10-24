import numpy as np

import torch


class AbstractLayer(torch.nn.Module):
    def __init__(self, input_shape, self_shape):
        super(AbstractLayer, self).__init__()
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(self.self_shape))))
        self.input = None
        self.activation = None
        self.epoch = -1
        self.batch_idx = 0
        self._setup_weights()

    def _setup_weights(self):
        raise NotImplementedError

    def train(self, mode=True):
        if mode or self.epoch == -1:
            self.epoch += 1
        self.batch_idx = 0
        super(self).train(mode=mode)


