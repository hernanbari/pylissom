import numpy as np

import torch


class Layer(torch.nn.Module):
    def __init__(self, input_shape, self_shape):
        super().__init__()
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.weights_shape = (np.prod(self.input_shape), np.prod(self.self_shape))
        # TODO: check if self.previous_activation_shape = self.input_shape works with bulk images
        # I think it wont't work
        self.previous_activations_shape = (1, np.prod(self.self_shape))
        self._setup_variables()

    def _setup_variables(self):
        raise NotImplementedError
