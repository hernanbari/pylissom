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
from src.supervised_gcal.utils import summary_images
from torch.autograd import Variable
from torchvision import datasets, transforms

if os.path.exists('runs'):
    shutil.rmtree('runs')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--ipdb', action='store_true', default=False,
                    help='activate ipdb set_trace()')
args = parser.parse_args()

if not args.ipdb:
    import ipdb

    ipdb.set_trace = lambda: 0

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()
                   ),
    batch_size=args.batch_size, shuffle=False, **kwargs)

# Lissom Model
classes = 10
lissom_shape = (20, 20)
input_shape = (28, 28)
lissom_neurons = int(np.prod(lissom_shape))
input_neurons = int(np.prod(input_shape))
lissom_model = LissomCortexLayer(input_shape, lissom_shape)
optimizer = LissomHebbianOptimizer()

if args.cuda:
    lissom_model.cuda()


def train_lissom(epoch):
    lissom_model.train()
    writer = SummaryWriter(log_dir='runs/epoch_' + str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 2000:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = lissom_model(data)
        optimizer.update_weights(lissom_model, step=batch_idx)
        summary_images(lissom_model, batch_idx, data, output, writer)

        if batch_idx % (args.log_interval * 50) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader)))
    writer.close()


for epoch in range(1, args.epochs + 1):
    train_lissom(epoch)

# Increases batch sizes for perceptron
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()
                   ),
    batch_size=1, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=1, shuffle=False, **kwargs)

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
        else:
            output = data
        nn_output = perceptron_model(torch.autograd.Variable(output))
        loss = F.nll_loss(nn_output, target)
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()

        if batch_idx % (args.log_interval * 50) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


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
        else:
            output = data
        nn_output = perceptron_model(torch.autograd.Variable(output))
        test_loss += F.nll_loss(nn_output, target, size_average=False).data[0]  # sum up batch loss
        pred = nn_output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs * 10 + 1):
    train_nn(epoch)
    test()

# Control
for epoch in range(1, args.epochs * 10 + 1):
    train_nn(epoch, control=True)
    test(control=True)
