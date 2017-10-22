import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights, get_gaussian_weights_variable


class ReducedLissom(Layer):
    # The relationship between the excitatoriy radius, inhib_factor and excit_fator is really important for patchy map
    def __init__(self, input_shape, self_shape,
                 min_theta=0.0, max_theta=1.0,
                 afferent_radius=5, excitatory_radius=2, inhibitory_radius=5,
                 aff_factor=1, inhib_factor=1, excit_factor=1.5,
                 afferent_normalization=False, afferent_normalization_factor=1.0,
                 settling_steps=10, sparse=False, name='cortex'):
        self.afferent_normalization_factor = afferent_normalization_factor
        self.afferent_normalization = afferent_normalization
        self.aff_factor = aff_factor
        self.sparse = sparse
        self.excit_factor = excit_factor
        self.inhib_factor = inhib_factor
        self.settling_steps = settling_steps
        self.inhibitory_radius = inhibitory_radius
        self.excitatory_radius = excitatory_radius
        self.afferent_radius = afferent_radius
        super(ReducedLissom).__init__(input_shape, self_shape, min_theta, max_theta, name)

    def _get_weight_variable(self, input_shape, weights_shape, radius):
        sigma = (radius / 5 if radius / 5 > 1 else 1) if radius is not None else 2
        return torch.nn.Parameter(
            get_gaussian_weights_variable(input_shape, weights_shape, 2, radius, self.sparse).t())

    def _setup_weights(self):
        self.inhibitory_weights = self._get_weight_variable(input_shape=self.self_shape,
                                                            weights_shape=self.self_shape,
                                                            radius=self.inhibitory_radius)

        self.excitatory_weights = self._get_weight_variable(input_shape=self.self_shape,
                                                            weights_shape=self.self_shape,
                                                            radius=self.excitatory_radius)

        self.afferent_weights = self._get_weight_variable(input_shape=self.input_shape,
                                                          weights_shape=self.self_shape,
                                                          radius=self.afferent_radius)
        self.weights += [('afferent_weights', self.afferent_weights),
                         ('excitatory_weights', self.excitatory_weights),
                         ('inhibitory_weights', self.inhibitory_weights)]

    def forward(self, cortex_input):
        self.input = cortex_input
        self.afferent_activation = self.aff_factor * self.matmul(self.input, self.afferent_weights)
        if self.afferent_normalization:
            reshaped_input = self.input.repeat(self.input.shape[1], 1)
            masked_input = apply_circular_mask_to_weights(reshaped_input, self.afferent_radius)
            sums = masked_input.sum(1).unsqueeze(1).t()
            den = 1 + self.afferent_normalization_factor * sums
            self.afferent_activation = self.afferent_activation / den

        current_activation = self.piecewise_sigmoid(self.min_theta, self.max_theta, self.afferent_activation)
        for _ in range(self.settling_steps):
            self.excitatory_activation = self.matmul(current_activation, self.excitatory_weights)
            self.inhibitory_activation = self.matmul(current_activation, self.inhibitory_weights)

            sum_activations = self.afferent_activation \
                              + self.excit_factor * self.excitatory_activation \
                              - self.inhib_factor * self.inhibitory_activation

            current_activation = self.piecewise_sigmoid(self.min_theta, self.max_theta, sum_activations)
        self.activation = current_activation
        return self.activation

    def matmul(self, vector, matrix):
        if not self.sparse:
            matmul = torch.matmul(vector, matrix)
        else:
            # Pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense,
            # That's why it's reversed
            if isinstance(matrix, torch.autograd.Variable):
                matrix = matrix.data
            if isinstance(vector, torch.autograd.Variable):
                vector = vector.data
            matmul = torch.matmul(matrix.t(), vector.t()).t()
        return matmul
