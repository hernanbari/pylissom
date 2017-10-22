import numpy as np

import torch
from src.supervised_gcal.cortex_layer import ReducedLissom
from src.supervised_gcal.lgn_layer import LGNLayer


class Lissom(torch.nn.Module):
    def __init__(self, input_shape, lgn_shape, v1_shape, lgn_params=None, v1_params=None):
        super(Lissom).__init__()
        if lgn_params is None:
            lgn_params = {}
        if v1_params is None:
            v1_params = {}
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(v1_shape))))
        self.on = LGNLayer(input_shape=input_shape, self_shape=lgn_shape, on=True, **lgn_params)
        self.off = LGNLayer(input_shape=input_shape, self_shape=lgn_shape, on=False, **lgn_params)
        self.v1 = ReducedLissom(input_shape=lgn_shape, self_shape=v1_shape, **v1_params)

    def forward(self, retina):
        on_output = self.on(retina)
        off_output = self.off(retina)
        self.lgn_activation = on_output + off_output
        return self.v1(self.lgn_activation)

    def register_forward_hook(self, hook):
        handles = []
        for m in self.children():
            handles.append(m.register_forward_hook(hook))
        return handles
