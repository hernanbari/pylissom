from __future__ import print_function

import argparse
import os
import shutil

import numpy as np
from tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
from src.supervised_gcal.cortex_layer import LissomCortexLayer
from src.supervised_gcal.hebbian_optimizer import LissomHebbianOptimizer
from src.supervised_gcal.lgn_layer import LissomLGNLayer
from src.utils_pipeline import summary_images, get_dataset, summary_lgn, summary_weights
from torch.autograd import Variable

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
parser.add_argument('--ck', action='store_true', default=False,
                    help='uses cohn-kanade dataset')
parser.add_argument('--two-layers', action='store_true', default=False,
                    help='uses 2 lissom layers')
parser.add_argument('--only-one', action='store_true', default=False,
                    help='trains with number one')
parser.add_argument('--logdir', default='runs',
                    help='log dir for tensorboard')

args = parser.parse_args()

if os.path.exists(args.logdir):
    shutil.rmtree(args.logdir)

if not args.ipdb:
    import ipdb

    ipdb.set_trace = lambda: 0

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = get_dataset(train=True, args=args)
# Lissom Model
classes = 10
lissom_shape = (70, 70)
input_shape = (28, 28) if not args.ck else (96, 96)
lissom_neurons = int(np.prod(lissom_shape))
input_neurons = int(np.prod(input_shape))
# lissom_model = LissomCortexLayer(lissom_shape, lissom_shape)
# optimizer = LissomHebbianOptimizer()
batch_idx, (data, target) = next(enumerate(train_loader))
data, target = Variable(data), Variable(target)
writer = SummaryWriter(log_dir='runs/epoch_' + str(batch_idx))
i = batch_idx
from torchvision import utils as vutils

# for sigma_center in np.arange(0.1, 2, step=0.2):
#     for sigma_sorround in [1.5, 2, 3, 5, 8, 10]:
#         for radius in [3, 5, 8, 10, 15, 20]:
#             on_layer = LissomLGNLayer(input_shape, lissom_shape, radius=radius, sigma_center=sigma_center, sigma_sorround=sigma_sorround*sigma_center, on=True)
#             off_layer = LissomLGNLayer(input_shape, lissom_shape, radius=radius, sigma_center=sigma_center, sigma_sorround=sigma_sorround*sigma_center, on=False)
#
#             output_on = on_layer(data)
#             output_off = off_layer(data)
#             summary_lgn(off_layer, input_shape, lissom_shape, batch_idx, data, output_off, writer, 'off')
#             summary_lgn(on_layer, input_shape, lissom_shape, batch_idx, data, output_on, writer, 'on')
#             im = (output_on + output_off).view((1, 1) + lissom_shape)
#             im = vutils.make_grid(im.data, range=(0, 1))
#             writer.add_image('lgn_activation', im, batch_idx)
#             batch_idx += 1
#

# if args.cuda:
    # lissom_model.cuda()
    # on_layer.cuda()
    # off_layer.cuda()

if args.two_layers:
    lissom_model_2 = LissomCortexLayer(lissom_shape, lissom_shape)
    optimizer_2 = LissomHebbianOptimizer()

    if args.cuda:
        lissom_model_2.cuda()


def train_lissom(epoch):
    # lissom_model.train()
    writer = SummaryWriter(log_dir='runs/epoch_' + str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 2000:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if args.only_one and target.data[0] == 1:
            for i in range(250):
                output = lissom_model(data)
                optimizer.update_weights(lissom_model, step=batch_idx)
                summary_images(lissom_model, batch_idx, data, output, writer)
            break

        output_on = on_layer(data)
        output_off = off_layer(data)
        summary_lgn(off_layer, input_shape, lissom_shape, batch_idx, data, output_off, writer, 'off')
        summary_lgn(on_layer, input_shape, lissom_shape, batch_idx, data, output_on, writer, 'on')
        im = (output_on + output_off).view((1, 1) + lissom_shape)
        im = vutils.make_grid(im.data, range=(0, 1))
        writer.add_image('lgn_activation', im, batch_idx)
        import ipdb;
        ipdb.set_trace()
        # data = output_on + output_off
        # output = lissom_model(data)
        # optimizer.update_weights(lissom_model, step=batch_idx)
        # summary_images(lissom_model, batch_idx, data, output, writer)

        if args.two_layers:
            output_2 = lissom_model_2(Variable(output))
            optimizer_2.update_weights(lissom_model_2, step=batch_idx)

            # if batch_idx % (args.log_interval * 50) == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader)))
    writer.close()


# for epoch in range(1, args.epochs + 1):
#     train_lissom(epoch)

train_loader = get_dataset(train=True, args=args)

# 2 Layer Net
hidden_neurons = 20
perceptron_model = torch.nn.Sequential(
    torch.nn.Linear(lissom_neurons, classes),
    torch.nn.LogSoftmax()
)
optimizer_nn = torch.optim.SGD(perceptron_model.parameters(), lr=0.1)

if args.cuda:
    perceptron_model.cuda()


def train_nn(epoch, control=False):
    perceptron_model.train()
    # Train network
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 2000:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if not control:
            output = lissom_model(data)
            if args.two_layers:
                output = lissom_model_2(Variable(output))
        else:
            output = data.data.view(torch.Size((data.shape[0], input_neurons)))
        nn_output = perceptron_model(torch.autograd.Variable(output))
        loss = F.nll_loss(nn_output, target)
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()

        if batch_idx % (args.log_interval * 50) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


test_loader = get_dataset(train=False, args=args)


def test(control=False):
    perceptron_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if not control:
            output = lissom_model(data)
            if args.two_layers:
                output = lissom_model_2(Variable(output))
        else:
            output = data.data.view(torch.Size((data.shape[0], input_neurons)))
        nn_output = perceptron_model(torch.autograd.Variable(output))
        test_loss += F.nll_loss(nn_output, target, size_average=False).data[0]  # sum up batch loss
        pred = nn_output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# for epoch in range(1, args.epochs + 1):
#     train_nn(epoch)
#     test()

# Control
hidden_neurons = 20
perceptron_model = torch.nn.Sequential(
    torch.nn.Linear(input_neurons, classes),
    torch.nn.LogSoftmax()
)
optimizer_nn = torch.optim.SGD(perceptron_model.parameters(), lr=0.1)

if args.cuda:
    perceptron_model.cuda()

for epoch in range(1, args.epochs + 1):
    train_nn(epoch, control=True)
    test(control=True)
