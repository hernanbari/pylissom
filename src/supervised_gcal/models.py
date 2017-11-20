import torch

from src.supervised_gcal.modules.lissom import ReducedLissom, Lissom, Cortex
from src.supervised_gcal.optimizers import SequentialOptimizer, CortexHebbian, NeighborsDecay
from supervised_gcal.utils.config import global_config


def get_reduced_lissom(retinal_density='DEFAULT', cortical_density='DEFAULT',
                       rlissom_params='rlissom', optim_params='optim', cfg_path=None):
    config = global_config(infile=cfg_path).eval_dict()
    if not isinstance(retinal_density, int):
        retinal_density = config[retinal_density]['retinal_density']
    if not isinstance(cortical_density, int):
        cortical_density = config[cortical_density]['cortical_density']

    in_features = retinal_density ** 2
    out_features = cortical_density ** 2
    rlissom_params = config[rlissom_params]

    afferent_module = Cortex(in_features, out_features, **(rlissom_params['afferent_module']))
    excitatory_module = Cortex(out_features, out_features, **(rlissom_params['excitatory_module']))
    inhibitory_module = Cortex(out_features, out_features, **(rlissom_params['inhibitory_module']))
    model = ReducedLissom(afferent_module, excitatory_module, inhibitory_module, **(rlissom_params['others']))

    optim_params = config[optim_params]
    optimizer = SequentialOptimizer(
        CortexHebbian(cortex=afferent_module, **(optim_params['afferent'])),
        CortexHebbian(cortex=excitatory_module, **(optim_params['excitatory'])),
        CortexHebbian(cortex=inhibitory_module, **(optim_params['inhibitory'])),
    )
    return model, optimizer, None


def get_lissom(input_shape, lgn_shape, cortex_shape, pruning_step=None, final_epoch=None, lgn_params=None,
               v1_params=None):
    # Full Lissom
    on = LGN(self, in_features, out_features, on, radius, sigma_surround,
             sigma_center=1.0, min_theta=0.0, max_theta=1.0, strength=1.0,
             diff_of_gauss_cls=DifferenceOfGaussiansLinear, pw_sigmoid_cls=PiecewiseSigmoid)
    off = LGN(self, in_features, out_features, on, radius, sigma_surround,
              sigma_center=1.0, min_theta=0.0, max_theta=1.0, strength=1.0,
              diff_of_gauss_cls=DifferenceOfGaussiansLinear, pw_sigmoid_cls=PiecewiseSigmoid)
    v1, _, _ = get_reduced_lissom()
    model = Lissom(on, off, v1)
    optimizer = SequentialOptimizer(
        CortexHebbian(cortex=model.v1),
        NeighborsDecay(cortex=model.v1,
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
