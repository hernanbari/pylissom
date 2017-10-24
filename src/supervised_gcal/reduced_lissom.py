from src.supervised_gcal.cortex_layer import CortexLayer
from src.supervised_gcal.layer import AbstractLayer
from src.supervised_gcal.utils.functions import piecewise_sigmoid


class ReducedLissom(AbstractLayer):
    def __init__(self, input_shape, self_shape, min_theta=1.0, max_theta=1.0, afferent_params=None,
                 excitatory_params=None,
                 inhibitory_params=None, settling_steps=10):
        self.settling_steps = settling_steps
        if inhibitory_params is None:
            inhibitory_params = {}
        if excitatory_params is None:
            excitatory_params = {}
        if afferent_params is None:
            afferent_params = {}
        afferent_params.update({'input_shape': input_shape,
                                'self_shape': self_shape})
        excitatory_params.update({'input_shape': self_shape,
                                  'self_shape': self_shape})

        inhibitory_params.update({'input_shape': self_shape,
                                  'self_shape': self_shape})
        self.inhibitory_params = inhibitory_params
        self.excitatory_params = excitatory_params
        self.afferent_params = afferent_params
        self.max_theta = max_theta
        self.min_theta = min_theta
        super(ReducedLissom, self).__init__(input_shape, self_shape)

    def _setup_weights(self):
        self.afferent_layer = CortexLayer(**self.afferent_params)
        self.excitatory_layer = CortexLayer(**self.excitatory_params)
        self.inhibitory_layer = CortexLayer(**self.inhibitory_params)

    def forward(self, cortex_input):
        self.input = cortex_input
        afferent_activation = piecewise_sigmoid(self.min_theta, self.max_theta, self.afferent_layer(self.input))
        current_activation = afferent_activation.clone()
        for _ in range(self.settling_steps):
            excitatory_activation = self.excitatory_layer(current_activation)
            inhibitory_activation = self.inhibitory_layer(current_activation)

            sum_activations = afferent_activation + excitatory_activation - inhibitory_activation

            current_activation = piecewise_sigmoid(self.min_theta, self.max_theta, sum_activations)
        self.activation = current_activation
        return self.activation
