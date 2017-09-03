import numpy as np

import torch


class Layer(torch.nn.Module):
    def __init__(self, input_shape, self_shape):
        super().__init__()
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.afferent_weights_shape = (np.prod(self.input_shape), np.prod(self.self_shape))
        self.lateral_weights_shape = (np.prod(self.self_shape), np.prod(self.self_shape))
        self.activations_shape = (1, np.prod(self.self_shape))
        self._setup_variables()

    def _setup_variables(self):
        raise NotImplementedError
