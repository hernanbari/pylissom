import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils.weights import get_gaussian_weights_variable


class CortexLayer(Layer):
    def __init__(self, input_shape, self_shape, sigma=1.0, sparse=False, **kwargs):
        self.sigma = sigma
        self.sparse = sparse
        super(CortexLayer, self).__init__(input_shape, self_shape, sparse, **kwargs)

    def _setup_weights(self):
        gaussian_weights = get_gaussian_weights_variable(input_shape=self.self_shape,
                                                         output_shape=self.self_shape,
                                                         sigma=self.sigma,
                                                         radius=self.radius,
                                                         sparse=self.sparse)
        uniform_noise = torch.FloatTensor(gaussian_weights.size()).uniform_(0, 1)
        self.weights = torch.nn.Parameter((uniform_noise * gaussian_weights).t())
