import numpy as np
import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils import custom_sigmoid, get_gaussian, circular_mask, normalize


class LissomLGNLayer(Layer):
    def __init__(self, input_shape, self_shape, sigma_center=0.4, sigma_sorround=1.2, min_theta=0.0, max_theta=1.0,
                 lgn_factor=1.0, radius=10, on=False):
        self.radius = radius
        self.lgn_factor = lgn_factor
        self.on = on
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.sigma_sorround = sigma_sorround
        self.sigma_center = sigma_center
        super().__init__(input_shape, self_shape)

    def _get_weight_variable(self, shape, sigma, radius):
        return torch.nn.Parameter(get_gaussian(shape, sigma, radius=radius))

    def _setup_variables(self):
        sigma_center_weights_matrix = self._get_weight_variable(self.afferent_weights_shape, self.sigma_center, self.radius)
        sigma_sorround_weights_matrix = self._get_weight_variable(self.afferent_weights_shape, self.sigma_sorround, self.radius)
        if self.on:
            self.weights = torch.nn.Parameter((sigma_center_weights_matrix - sigma_sorround_weights_matrix).data)
        else:
            self.weights = torch.nn.Parameter((sigma_sorround_weights_matrix - sigma_center_weights_matrix).data)

    def forward(self, input):
        # TODO: pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense
        processed_input = self.process_input(input)
        # import ipdb; ipdb.set_trace()
        var = torch.matmul( self.weights.data.t(), processed_input.t())
        var = custom_sigmoid(self.min_theta, self.max_theta,
                             self.lgn_factor * var)
        return var
