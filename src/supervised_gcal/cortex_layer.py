import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils import get_zeros, get_uniform, normalize, circular_mask, get_gaussian


class LissomCortexLayer(Layer):
    # The relationship between the excitatoriy radius, inhib_factor and excit_fator is really important for patchy map
    def __init__(self, input_shape, self_shape, min_theta=0.0, max_theta=1.0, afferent_radius=None, excitatory_radius=8.0,
                 inhibitory_radius=None, settling_steps=100, inhib_factor=1.35, excit_factor=1.05):
        self.max_theta = max_theta
        self.excit_factor = excit_factor
        self.inhib_factor = inhib_factor
        self.settling_steps = settling_steps
        self.inhibitory_radius = inhibitory_radius
        self.excitatory_radius = excitatory_radius
        self.afferent_radius = afferent_radius
        self.min_theta = min_theta
        super().__init__(input_shape, self_shape)

    def _get_weight_variable(self, shape, radius):
        # TODO: learn what Parameter means
        sigma = (radius/5 if radius / 5 > 1 else 1) if radius is not None else 4
        return torch.nn.Parameter(torch.Tensor(normalize(circular_mask(get_gaussian(shape, sigma),
                                                                       radius=radius))))

    def _setup_variables(self):
        self.inhibitory_weights = self._get_weight_variable(shape=self.lateral_weights_shape,
                                                            radius=self.inhibitory_radius)

        self.excitatory_weights = self._get_weight_variable(shape=self.lateral_weights_shape,
                                                            radius=self.excitatory_radius)

        self.retina_weights = self._get_weight_variable(shape=self.afferent_weights_shape, radius=self.afferent_radius)

        # Variable que guarda activaciones previas
        self.previous_activations = torch.Tensor(get_zeros(self.activations_shape))

    def _afferent_activation(self, input, weights):
        return torch.matmul(input, weights.data)

    def _lateral_activation(self, previous_activations, weights):
        return torch.matmul(previous_activations, weights.data)

    def process_input(self, input, normalize=False):
        var = input
        if normalize:
            var = var / torch.norm(input, p=2, dim=1)
        var = var.data.view(self.input_shape)
        return var

    def custom_sigmoid(self, new_activations):
        new_activations = torch.nn.functional.threshold(new_activations, self.min_theta, value=0.0)
        new_activations.masked_fill_(
            mask=torch.gt(new_activations, self.max_theta),
            value=1.0)

        new_activations.sub_(self.min_theta).div_(self.max_theta - self.min_theta)
        return new_activations

    def forward(self, input):
        processed_input = self.process_input(input)
        retina = processed_input
        self.retina = retina
        self.retina_activation = self._afferent_activation(retina, self.retina_weights)
        self.afferent_activation = self.retina_activation

        self.previous_activations = self.afferent_activation
        for _ in range(self.settling_steps):
            self.excitatory_activation = self._lateral_activation(self.previous_activations,
                                                                  self.excitatory_weights)
            self.inhibitory_activation = self._lateral_activation(self.previous_activations,
                                                                  self.inhibitory_weights)

            new_activations = self.afferent_activation + self.excit_factor * self.excitatory_activation - self.inhib_factor * self.inhibitory_activation
            new_activations = self.custom_sigmoid(new_activations).data

            self.previous_activations = new_activations
        return new_activations
