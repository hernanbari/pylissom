import torch
import numpy as np
from src.supervised_gcal.cortex_layer import CortexLayer
from src.supervised_gcal.lgn_layer import LGNLayer
from src.supervised_gcal.optimizers import CortexHebbian, NeighborsDecay, ConnectionDeath, \
    CombinedCortexOptimizer


class FullLissom(torch.nn.Module):
    def __init__(self, input_shape, lgn_shape, v1_shape, lgn_params=None, v1_params=None):
        super().__init__()
        if lgn_params is None:
            lgn_params = {}
        if v1_params is None:
            v1_params = {}
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(v1_shape))))
        on = LGNLayer(input_shape=input_shape, self_shape=lgn_shape, on=True, **lgn_params)
        off = LGNLayer(input_shape=input_shape, self_shape=lgn_shape, on=False, **lgn_params)
        v1 = CortexLayer(input_shape=lgn_shape, self_shape=v1_shape, **v1_params)

        for key, module in [('on', on), ('off', off), ('v1', v1)]:
            self.add_module(key, module)

    def forward(self, retina):
        on_output = self.on(retina)
        off_output = self.off(retina)
        self.lgn_activation = on_output + off_output
        return self.v1(self.lgn_activation)


# TODO: test and define an optimizer
class HLissom(torch.nn.Module):
    def __init__(self, input_shape, lgn_shape, v1_shape, lgn_params=None, v1_params=None):
        super().__init__()
        if lgn_params is None:
            lgn_params = {}
        if v1_params is None:
            v1_params = {}
        self.full_lissom = FullLissom(input_shape, lgn_shape, v1_shape, lgn_params, v1_params)
        self.cortexes = [self.full_lissom.v1]

    def add_cortex_layer(self, cortex_shape):
        last_shape = self.cortexes[-1].shape
        self.cortexes.append(CortexLayer(input_shape=last_shape, self_shape=cortex_shape))

    def forward(self, retina):
        self.activations = [self.full_lissom(retina)]
        for layer in self.cortexes:
            self.activations.append(layer(self.activations[-1]))
        return self.activations[-1]


class HebbianAndPruners(CombinedCortexOptimizer):
    optimizers = [CortexHebbian, ConnectionDeath, NeighborsDecay]
