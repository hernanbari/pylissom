from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils.weights import get_gaussian_weights_variable


class LGNLayer(Layer):
    def __init__(self, input_shape, self_shape, on, sigma_center=0.1, sigma_surround=10, min_theta=0.0, max_theta=1.0,
                 strength=1.0, radius=10, sparse=False, name=''):
        self.on = on
        self.sigma_surround = sigma_surround
        self.sigma_center = sigma_center
        name = ('on' if on else 'off') if name == '' else name
        super(LGNLayer).__init__(input_shape, self_shape, min_theta, max_theta, strength, radius, sparse, name)

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
        self.weights = diff
