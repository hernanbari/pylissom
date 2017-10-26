import torch
from src.supervised_gcal.layers.abstract_layer import AbstractLayer
from src.supervised_gcal.utils.functions import afferent_normalize, piecewise_sigmoid


class SimpleLayer(AbstractLayer):
    def __init__(self, input_shape, self_shape, min_theta=0.0, max_theta=1.0, strength=1.0, radius=1.0,
                 afferent_normalization=False, sparse=False, name=''):
        self.sparse = sparse
        self.radius = radius
        self.strength = strength
        self.name = name
        self.weights = None
        self.afferent_normalization = afferent_normalization
        self.max_theta = max_theta
        self.min_theta = min_theta
        super(SimpleLayer, self).__init__(input_shape, self_shape)

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

