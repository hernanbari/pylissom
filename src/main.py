from __future__ import print_function

import argparse
import os
import shutil

import numpy as np

import torch
from src.supervised_gcal.layers.lgn_layer import LGNLayer
from src.supervised_gcal.models import get_reduced_lissom, get_lissom, get_net, get_supervised
from src.utils.cross_validation import CVSubjectIndependent, run_cross_validation
from src.utils.datasets import get_dataset, CKDataset
from src.utils.grid_search import run_lgn_grid_search, run_lissom_grid_search, run_supervised_grid_search
from src.utils.pipeline import Pipeline

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
parser.add_argument('--cv', action='store_true', default=False,
                    help='Run cross validation')
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
    model, optimizer, _ = get_reduced_lissom(input_shape, cortex_shape, args)

if args.model == 'lissom':
    model, optimizer, _ = get_lissom(input_shape, lgn_shape, cortex_shape, args.log_interval, args.epochs)

handles = None
if args.save_images and args.model in ['lgn', 'cortex', 'lissom']:
    handles = model.register_forward_hook(images.generate_images)

if args.model == 'control':
    net_input_shape = batch_input_shape[1]
    model, optimizer, loss_fn = get_net(net_input_shape, classes)
    model_fn = lambda: get_net(net_input_shape, classes)

if args.model == 'supervised':
    model, optimizer, loss_fn = get_supervised(input_shape, lgn_shape, cortex_shape, args.log_interval, args.epochs,
                                               classes)
    model[0].register_forward_hook(images.generate_images)
    model_fn = lambda: get_supervised(input_shape, lgn_shape, cortex_shape, args.log_interval,
                                      args.epochs,
                                      classes)

if not args.cv and 'grid-search' not in args.model:
    test_loader = get_dataset(train=False, args=args)
    train_loader = get_dataset(train=True, args=args)
    pipeline = Pipeline(model, optimizer, loss_fn, log_interval=args.log_interval, dataset_len=args.dataset_len,
                        cuda=args.cuda)
    # TODO: Change epochs to 0
    for epoch in range(1, args.epochs + 1):
        pipeline.train(train_data_loader=train_loader, epoch=epoch)
        pipeline.test(test_data_loader=test_loader, epoch=epoch)
    exit()

# TODO: train lissom first and then net
if args.cv:
    assert model_fn is not None, "Cross validation only with supervised or control models"
    ck_dataset = CKDataset()
    cv = CVSubjectIndependent(ck_dataset)
    run_cross_validation(model_fn, ck_dataset, cv, args)
    exit()

# TODO: test
if args.model == 'lgn-grid-search':
    run_lgn_grid_search(input_shape, lgn_shape, args)
    exit()

# TODO: test
if args.model == 'lissom-grid-search':
    run_lissom_grid_search(input_shape, lgn_shape, cortex_shape, args)
    exit()

# TODO: test
# TODO: train lissom first and then net
if args.model == 'supervised-grid-search':
    run_supervised_grid_search(input_shape, lgn_shape, cortex_shape, args)
    exit()
