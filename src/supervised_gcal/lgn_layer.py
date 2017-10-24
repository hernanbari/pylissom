import torch
from src.supervised_gcal.layer import SimpleLayer
from src.supervised_gcal.utils.weights import get_gaussian_weights_variable


class LGNLayer(SimpleLayer):
    def __init__(self, on, input_shape, self_shape, sigma_center=0.1, sigma_surround=10, **kwargs):
        self.on = on
        self.sigma_surround = sigma_surround
        self.sigma_center = sigma_center
        super(LGNLayer, self).__init__(input_shape, self_shape, **kwargs)

    def _setup_weights(self):
        sigma_center_weights_matrix = get_gaussian_weights_variable(self.input_shape, self.self_shape,
                                                                    self.sigma_center,
                                                                    self.radius,
                                                                    self.sparse)
        sigma_surround_weights_matrix = get_gaussian_weights_variable(self.input_shape, self.self_shape,
                                                                      self.sigma_surround,
                                                                      self.radius,
                                                                      self.sparse)
        if self.on:
            diff = (sigma_center_weights_matrix - sigma_surround_weights_matrix).t()
        else:
            diff = (sigma_surround_weights_matrix - sigma_center_weights_matrix).t()
        self.weights = torch.nn.Parameter(diff)
