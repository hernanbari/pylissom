import torch
from src.supervised_gcal.layer import Layer, get_gaussian_weights_variable
from src.supervised_gcal.utils.images import image_decorator


@image_decorator
class CortexLayer(Layer):
    # The relationship between the excitatoriy radius, inhib_factor and excit_fator is really important for patchy map
    def __init__(self, input_shape, self_shape, min_theta=0.0, max_theta=1.0, afferent_radius=None,
                 excitatory_radius=8.0,
                 inhibitory_radius=None, settling_steps=10, inhib_factor=1.5, excit_factor=1.05, name=''):
        self.max_theta = max_theta
        self.excit_factor = excit_factor
        self.inhib_factor = inhib_factor
        self.settling_steps = settling_steps
        self.inhibitory_radius = inhibitory_radius
        self.excitatory_radius = excitatory_radius
        self.afferent_radius = afferent_radius
        self.min_theta = min_theta
        super().__init__(input_shape, self_shape, name)

    def _get_weight_variable(self, input_shape, weights_shape, radius):
        sigma = (radius / 5 if radius / 5 > 1 else 1) if radius is not None else 2
        return torch.nn.Parameter(get_gaussian_weights_variable(input_shape, weights_shape, sigma, radius).t())

    def _setup_variables(self):
        self.inhibitory_weights = self._get_weight_variable(input_shape=self.self_shape,
                                                            weights_shape=self.self_shape,
                                                            radius=self.inhibitory_radius)

        self.excitatory_weights = self._get_weight_variable(input_shape=self.self_shape,
                                                            weights_shape=self.self_shape,
                                                            radius=self.inhibitory_radius)

        self.afferent_weights = self._get_weight_variable(input_shape=self.input_shape,
                                                          weights_shape=self.self_shape,
                                                          radius=self.inhibitory_radius)

    def forward(self, cortex_input):
        self.cortex_input = cortex_input
        self.afferent_activation = torch.matmul(self.cortex_input, self.afferent_weights)

        current_activation = self.afferent_activation
        for _ in range(self.settling_steps):
            self.excitatory_activation = torch.matmul(current_activation, self.excitatory_weights)
            self.inhibitory_activation = torch.matmul(current_activation, self.inhibitory_weights)

            sum_activations = self.afferent_activation \
                              + self.excit_factor * self.excitatory_activation \
                              - self.inhib_factor * self.inhibitory_activation

            current_activation = self.custom_sigmoid(self.min_theta, self.max_theta, sum_activations)
        self.activation = current_activation
        return self.activation
