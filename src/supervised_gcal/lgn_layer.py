import torch
from src.supervised_gcal.layer import Layer, get_gaussian_weights_variable


class LGNLayer(Layer):
    def __init__(self, input_shape, self_shape, on, sigma_center=0.1, sigma_sorround=10, min_theta=0.0, max_theta=1.0,
                 lgn_factor=1.0, radius=10, sparse=False, name=''):
        self.sparse = sparse
        self.radius = radius
        self.lgn_factor = lgn_factor
        self.on = on
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.sigma_sorround = sigma_sorround
        self.sigma_center = sigma_center
        super().__init__(input_shape, self_shape, ('on' if on else 'off') if name == '' else name)

    def _setup_variables(self):
        sigma_center_weights_matrix = get_gaussian_weights_variable(self.input_shape, self.self_shape,
                                                                    self.sigma_center,
                                                                    self.radius,
                                                                    self.sparse)
        sigma_sorround_weights_matrix = get_gaussian_weights_variable(self.input_shape, self.self_shape,
                                                                      self.sigma_sorround,
                                                                      self.radius,
                                                                      self.sparse)
        if self.on:
            diff = (sigma_center_weights_matrix - sigma_sorround_weights_matrix).t()
        else:
            diff = (sigma_sorround_weights_matrix - sigma_center_weights_matrix).t()
        self.register_buffer(name='afferent_weights', tensor=diff)
        self.weights = [('afferent_weights', self.afferent_weights)]

    def forward(self, lgn_input):
        self.input = lgn_input
        matmul = self.matmul(self.input, self.afferent_weights)
        # Custom sigmoid returns a variable
        self.activation = self.custom_sigmoid(self.min_theta, self.max_theta,
                                              self.lgn_factor * matmul)
        return self.activation

    def matmul(self, vector, matrix):
        if not self.sparse:
            return torch.matmul(vector.data, matrix)
        else:
            # Pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense,
            # That's why it's reversed
            return torch.matmul(matrix.t(), vector.data.t()).t()
