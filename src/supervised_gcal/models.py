import numpy as np

import torch
from src.main import args
from src.supervised_gcal.cortex_layer import CortexLayer
from src.supervised_gcal.lgn_layer import LGNLayer
from src.supervised_gcal.optimizers import SequentialOptimizer, CortexHebbian, NeighborsDecay


class FullLissom(torch.nn.Module):
    def __init__(self, input_shape, lgn_shape, v1_shape, lgn_params=None, v1_params=None):
        super().__init__()
        if lgn_params is None:
            lgn_params = {}
        if v1_params is None:
            v1_params = {}
        self.input_shape = input_shape
        self.activation_shape = torch.Size((1, int(np.prod(v1_shape))))
        self.on = LGNLayer(input_shape=input_shape, self_shape=lgn_shape, on=True, **lgn_params)
        self.off = LGNLayer(input_shape=input_shape, self_shape=lgn_shape, on=False, **lgn_params)
        self.v1 = CortexLayer(input_shape=lgn_shape, self_shape=v1_shape, **v1_params)

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


def get_cortex(input_shape, cortex_shape):
    # Cortex Layer
    model = CortexLayer(input_shape, cortex_shape)
    optimizer = SequentialOptimizer(
        CortexHebbian(cortex_layer=model.v1),
        NeighborsDecay(cortex_layer=model.v1,
                       pruning_step=args.log_interval, final_epoch=args.epochs))
    return model, optimizer, None


def get_full_lissom(input_shape, lgn_shape, cortex_shape, pruning_step=None, final_epoch=None, lgn_params=None,
                    v1_params=None):
    # Full Lissom
    model = FullLissom(input_shape, lgn_shape, cortex_shape, lgn_params=lgn_params, v1_params=v1_params)
    optimizer = SequentialOptimizer(
        CortexHebbian(cortex_layer=model.v1),
        NeighborsDecay(cortex_layer=model.v1,
                       pruning_step=pruning_step, final_epoch=final_epoch))
    return model, optimizer, None


def get_net(net_input_shape, classes):
    # 1 Layer Net
    net = torch.nn.Sequential(
        torch.nn.Linear(net_input_shape, classes),
        torch.nn.LogSoftmax()
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_fn = torch.nn.functional.nll_loss
    return net, optimizer, loss_fn


def get_supervised(input_shape, lgn_shape, cortex_shape, pruning_step, final_epoch, classes, v1_params=None):
    # Lissom
    lissom, optimizer_lissom, _ = get_full_lissom(input_shape, lgn_shape, cortex_shape, pruning_step, final_epoch,
                                                  v1_params=v1_params)
    # Net
    net_input_shape = lissom.activation_shape[1]
    net, optimizer_nn, loss_fn = get_net(net_input_shape, classes)
    # Supervised Lissom
    model = torch.nn.Sequential(
        lissom,
        net
    )
    optimizer = SequentialOptimizer(
        optimizer_lissom,
        optimizer_nn
    )
    return model, optimizer, loss_fn
