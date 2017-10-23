import torch
from src.supervised_gcal.cortex_layer import CortexLayer
from src.supervised_gcal.utils.functions import piecewise_sigmoid


class ReducedLissom(torch.nn.Module):
    def __init__(self, input_shape, self_shape, min_theta=0.0, max_theta=1.0,
                 afferent_params=None, excitatory_params=None, inhibitory_params=None,
                 settling_steps=10):
        super(ReducedLissom, self).__init__()
        if inhibitory_params is None:
            inhibitory_params = {}
        if excitatory_params is None:
            excitatory_params = {}
        if afferent_params is None:
            afferent_params = {}
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.settling_steps = settling_steps
        afferent_params.update({'input_shape': input_shape,
                                'self_shape': self_shape})
        self.afferent_layer = CortexLayer(**afferent_params)
        excitatory_params.update({'input_shape': self_shape,
                                  'self_shape': self_shape})
        self.excitatory_layer = CortexLayer(**excitatory_params)
        inhibitory_params.update({'input_shape': self_shape,
                                  'self_shape': self_shape})
        self.inhibitory_layer = CortexLayer(**inhibitory_params)
        self.input = None
        self.activation = None

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
