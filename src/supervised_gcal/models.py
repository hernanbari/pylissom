import torch

from src.supervised_gcal.modules.lissom import ReducedLissom, Lissom, Cortex, LGN
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


def get_lissom(retinal_density='DEFAULT', lgn_density='DEFAULT', cortical_density='DEFAULT',
               lgn_params='lgn', rlissom_params='rlissom', optim_params='optim', cfg_path=None):
    config = global_config(infile=cfg_path).eval_dict()
    if not isinstance(retinal_density, int):
        retinal_density = config[retinal_density]['retinal_density']
    if not isinstance(lgn_density, int):
        lgn_density = config[lgn_density]['lgn_density']
    # Full Lissom

    in_features = retinal_density ** 2
    out_features = lgn_density ** 2

    lgn_params = config[lgn_params]
    on = LGN(in_features, out_features, on=True, **lgn_params)
    off = LGN(in_features, out_features, on=False, **lgn_params)
    v1, optimizer, _ = get_reduced_lissom(retinal_density, cortical_density, rlissom_params, optim_params, cfg_path)
    model = Lissom(on, off, v1)
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


def get_supervised(retinal_density='DEFAULT', lgn_density='DEFAULT', cortical_density='DEFAULT',
                   lgn_params='lgn', rlissom_params='rlissom', optim_params='optim', cfg_path=None, classes=10):
    # Lissom
    lissom, optimizer_lissom, _ = get_lissom(retinal_density, lgn_density, cortical_density,
                                             lgn_params, rlissom_params, optim_params, cfg_path)
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
