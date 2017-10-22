import torch
from src.supervised_gcal.reduced_lissom import ReducedLissom
from src.supervised_gcal.lissom import Lissom
from src.supervised_gcal.optimizers import SequentialOptimizer, CortexHebbian, NeighborsDecay


def get_reduced_lissom(input_shape, cortex_shape, pruning_step=None, final_epoch=None, v1_params=None,
                       learning_rate=None):
    # Cortex Layer
    model = ReducedLissom(input_shape, cortex_shape, **v1_params)
    optimizer = SequentialOptimizer(
        CortexHebbian(cortex_layer=model, learning_rate=learning_rate),
        NeighborsDecay(cortex_layer=model,
                       pruning_step=pruning_step, final_epoch=final_epoch))
    return model, optimizer, None


def get_lissom(input_shape, lgn_shape, cortex_shape, pruning_step=None, final_epoch=None, lgn_params=None,
               v1_params=None):
    # Full Lissom
    model = Lissom(input_shape, lgn_shape, cortex_shape, lgn_params=lgn_params, v1_params=v1_params)
    optimizer = SequentialOptimizer(
        CortexHebbian(cortex_layer=model.v1),
        NeighborsDecay(cortex_layer=model.v1,
                       pruning_step=pruning_step, final_epoch=final_epoch))
    return model, optimizer, None


def get_net(net_input_shape, classes):
    # 1 Layer Net
    net = torch.nn.Sequential(
        torch.nn.Linear(net_input_shape, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, classes),
        torch.nn.LogSoftmax()
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_fn = torch.nn.functional.nll_loss
    return net, optimizer, loss_fn


def get_supervised(input_shape, lgn_shape, cortex_shape, pruning_step, final_epoch, classes, v1_params=None):
    # Lissom
    lissom, optimizer_lissom, _ = get_lissom(input_shape, lgn_shape, cortex_shape, pruning_step, final_epoch,
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
