import torch
from src.supervised_gcal.cortex_layer import ReducedLissom
from src.supervised_gcal.lissom import Lissom


# TODO: test and define an optimizer
class HLissom(torch.nn.Module):
    def __init__(self, input_shape, lgn_shape, v1_shape, lgn_params=None, v1_params=None):
        super().__init__()
        if lgn_params is None:
            lgn_params = {}
        if v1_params is None:
            v1_params = {}
        self.full_lissom = Lissom(input_shape, lgn_shape, v1_shape, lgn_params, v1_params)
        self.cortexes = [self.full_lissom.v1]

    def add_cortex_layer(self, cortex_shape):
        last_shape = self.cortexes[-1].shape
        self.cortexes.append(ReducedLissom(input_shape=last_shape, self_shape=cortex_shape))

    def forward(self, retina):
        self.activations = [self.full_lissom(retina)]
        for layer in self.cortexes:
            self.activations.append(layer(self.activations[-1]))
        return self.activations[-1]
