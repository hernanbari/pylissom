from __future__ import print_function

import argparse
import copy
import os
import shutil

import numpy as np

import torch
from src.supervised_gcal.lgn_layer import LGNLayer
from src.supervised_gcal.models import FullLissom, get_cortex, get_full_lissom, get_net, get_supervised
from src.supervised_gcal.optimizers import CortexHebbian, SequentialOptimizer, NeighborsDecay
from src.utils.datasets import get_dataset, CKDataset, CVSubjectIndependent
from src.utils.pipeline import Pipeline
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--ipdb', action='store_true', default=False,
                    help='activate ipdb set_trace()')
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'ck', 'number_one'],
                    help='which dataset iterate')
parser.add_argument('--logdir', default='runs',
                    help='log dir for tensorboard')
parser.add_argument('--model', required=True,
                    choices=['lgn', 'cortex', 'lissom', 'supervised', 'control', 'lgn-grid-search',
                             'lissom-grid-search', 'supervised-grid-search', 'cv'],
                    help='which model to evaluate')
parser.add_argument('--shape', type=int, default=28, metavar='N',
                    help='# of rows of square maps')
parser.add_argument('--save_images', action='store_false', default=True,
                    help='save images for tensorboard')
parser.add_argument('--dataset-len', type=int, default=None, metavar='N',
                    help='max batch len')

args = parser.parse_args()

if os.path.exists(args.logdir):
    shutil.rmtree(args.logdir)

import src.supervised_gcal.utils.images as images

images.logdir = args.logdir
images.log_interval = args.log_interval

if not args.ipdb:
    import IPython.core.debugger as dbg

    dbg.Pdb.set_trace = lambda s: 0
    import ipdb

    ipdb.set_trace = lambda: 0

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

classes = 10 if not args.dataset == 'ck ' else 7
input_shape = (28, 28) if not args.dataset == 'ck' else (96, 96)
batch_input_shape = torch.Size((args.batch_size, int(np.prod(input_shape))))
model = None
optimizer = None
loss_fn = None
lgn_shape = (args.shape, args.shape)
cortex_shape = (args.shape, args.shape)

if args.model == 'lgn':
    # LGN layer
    model = LGNLayer(input_shape, lgn_shape, on=True)

if args.model == 'cortex':
    model, optimizer = get_cortex(input_shape, cortex_shape)

if args.model == 'lissom':
    model, optimizer = get_full_lissom(input_shape, lgn_shape, cortex_shape, args.log_interval, args.epochs)

handles = None
if args.save_images and args.model in ['lgn', 'cortex', 'lissom']:
    handles = model.register_forward_hook(images.generate_images)

if args.model == 'control':
    net_input_shape = batch_input_shape[1]
    model, optimizer, loss_fn = get_net(net_input_shape, classes)

if args.model == 'supervised':
    model, optimizer, loss_fn = get_supervised(input_shape, lgn_shape, cortex_shape, args.log_interval, args.epochs,
                                               classes)
    model[0].register_forward_hook(images.generate_images)

if args.model != 'cv' and 'grid-search' not in args.model:
    test_loader = get_dataset(train=False, args=args)
    train_loader = get_dataset(train=True, args=args)
    pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval, dataset_len=args.dataset_len,
                        cuda=args.cuda)

    for epoch in range(1, args.epochs + 1):
        pipeline.train(train_data_loader=train_loader, epoch=epoch)
        pipeline.test(test_data_loader=test_loader, epoch=epoch)

if args.model == 'lgn-grid-search':
    counter = 0

    test_loader = get_dataset(train=False, args=args)
    for sigma_center in np.arange(0.1, 10, step=0.5):
        for sigma_sorround in [1.5, 2, 3, 5, 8, 10]:
            for radius in [3, 4, 5, 8, 10, 15, 20]:
                lgn_shape = (args.shape, args.shape)
                model = FullLissom(input_shape, lgn_shape, (1, 1),
                                   lgn_params={'sigma_center': sigma_center, 'sigma_sorround': sigma_sorround,
                                               'radius': radius})
                pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval,
                                    dataset_len=args.dataset_len,
                                    cuda=args.cuda)
                if args.save_images:
                    def hardcoded_counter(self, input, output):
                        self.batch_idx = counter
                        images.generate_images(self, input, output)


                    model.register_forward_hook(hardcoded_counter)
                pipeline.test(test_data_loader=test_loader)
                print("Iteration", counter)
                print(sigma_center, sigma_sorround, radius)
                counter += 1
    exit()

if args.model == 'lissom-grid-search':
    counter = 0
    train_loader = get_dataset(train=True, args=args)
    for afferent_radius in [3, 5, 10, 15, 20]:
        for excitatory_radius in [2, 4, 9, 14]:
            for inhib_factor in [1.0, 1.5, 3.0]:
                for excit_factor in [1.0, 1.5, 3.0]:
                    if excitatory_radius > afferent_radius:
                        continue

                    lgn_shape = (args.shape, args.shape)
                    lissom_shape = (args.shape, args.shape)
                    inhibitory_radius = afferent_radius
                    params_list = [(name, eval(name)) for name in
                                   ['afferent_radius', 'excitatory_radius',
                                    'inhib_factor', 'excit_factor']]
                    model = FullLissom(input_shape, lgn_shape, lissom_shape,
                                       v1_params=dict(params_list))
                    optimizer = SequentialOptimizer(
                        CortexHebbian(cortex_layer=model.v1),
                        NeighborsDecay(cortex_layer=model.v1, pruning_step=args.log_interval, final_epoch=5)
                    )
                    pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval,
                                        dataset_len=args.dataset_len,
                                        cuda=args.cuda)
                    if args.save_images:
                        def hardcoded_counter(self, input, output):
                            self.batch_idx = counter
                            images.generate_images(self, input, output)


                        model.register_forward_hook(hardcoded_counter)
                    for epoch in range(1, args.epochs + 1):
                        pipeline.train(train_data_loader=train_loader, epoch=epoch)
                    print("Iteration", counter)
                    print(params_list)
                    counter += 1
    exit()

# TODO: train lissom first and then net
if args.model == 'supervised-grid-search':
    counter = 0
    test_loader = get_dataset(train=False, args=args)
    train_loader = get_dataset(train=True, args=args)
    params_names = ('afferent_radius', 'excitatory_radius', 'inhib_factor', 'excit_factor')
    for params_values in [(3, 2, 1, 1.5), (5, 2, 1, 1.5), (5, 4, 1, 1.5), (5, 4, 3, 3)]:
        params_dict = dict(zip(params_names, params_values))
        lgn_shape = (args.shape, args.shape)
        lissom_shape = (args.shape, args.shape)
        inhibitory_radius = params_dict['afferent_radius']
        lissom = FullLissom(input_shape, lgn_shape, lissom_shape,
                            v1_params=params_dict)
        net_input_shape = lissom.activation_shape[1]
        net = torch.nn.Sequential(
            torch.nn.Linear(net_input_shape, classes),
            torch.nn.LogSoftmax()
        )
        loss_fn = torch.nn.functional.nll_loss
        model = torch.nn.Sequential(
            lissom,
            net
        )
        optimizer = SequentialOptimizer(
            CortexHebbian(cortex_layer=lissom.v1),
            NeighborsDecay(cortex_layer=lissom.v1, pruning_step=args.log_interval, final_epoch=args.epochs),
            torch.optim.SGD(net.parameters(), lr=0.1)
        )
        pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval,
                            dataset_len=args.dataset_len,
                            cuda=args.cuda)
        if args.save_images:
            def hardcoded_counter(self, input, output):
                images.logdir = args.logdir + '/counter_' + str(counter)
                images.generate_images(self, input, output)


            lissom.register_forward_hook(hardcoded_counter)
        for epoch in range(1, args.epochs + 1):
            pipeline.train(train_data_loader=train_loader, epoch=epoch)
            pipeline.test(test_data_loader=test_loader, epoch=epoch)
        print("Iteration", counter)
        print(params_dict)
        counter += 1
    exit()

# TODO: train lissom first and then net
if args.model == 'cv':

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    ck_dataset = CKDataset()
    cv = CVSubjectIndependent(ck_dataset)
    best_accuracy = 0
    best_model = None
    for fold, (train_sampler, val_sampler) in enumerate(cv.train_val_samplers()):
        train_loader = DataLoader(ck_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(ck_dataset, batch_size=args.batch_size, sampler=val_sampler, **kwargs)
        model, optimizer, loss_fn = get_supervised(input_shape, lgn_shape, cortex_shape, args.log_interval,
                                                   args.epochs,
                                                   classes)
        pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval, dataset_len=args.dataset_len,
                            cuda=args.cuda, prefix='fold_' + str(fold))
        if args.save_images:
            model[0].register_forward_hook(lambda *x: images.generate_images(*x, prefix='fold_' + str(fold)))
        for epoch in range(1, args.epochs + 1):
            pipeline.train(train_data_loader=train_loader, epoch=epoch)
            curr_accuracy = pipeline.test(test_data_loader=val_loader, epoch=epoch)

        if best_model is None or curr_accuracy > best_accuracy:
            best_model = copy.deepcopy(model.state_dict())
            best_accuracy = curr_accuracy
    fold = 'test'
    model, optimizer, loss_fn = get_supervised(input_shape, lgn_shape, cortex_shape, args.log_interval,
                                               args.epochs,
                                               classes)
    model.load_state_dict(best_model)
    if args.save_images:
        model[0].register_forward_hook(lambda *x: images.generate_images(*x, prefix='fold_' + str(fold)))
    test_sampler = cv.test_sampler()
    test_loader = DataLoader(ck_dataset, batch_size=args.batch_size, sampler=test_sampler, **kwargs)
    pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval, dataset_len=args.dataset_len,
                        cuda=args.cuda, prefix='fold_' + str(fold))
    pipeline.test(test_data_loader=test_loader, epoch=1)
    exit()
