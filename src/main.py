from __future__ import print_function

import argparse
import os
import shutil

import numpy as np
from tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from src.supervised_gcal.cortex_layer import LissomCortexLayer
from src.supervised_gcal.hebbian_optimizer import LissomHebbianOptimizer
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
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

# Lissom Model
classes = 10
lissom_shape = (20, 20)
input_shape = (28, 28)
lissom_neurons = int(np.prod(lissom_shape))
input_neurons = int(np.prod(input_shape))
model = LissomCortexLayer((1, input_neurons), lissom_shape)
optimizer = LissomHebbianOptimizer()

# 2 Layer Net
hidden_neurons = 20
model_nn = torch.nn.Sequential(
    torch.nn.Linear(lissom_neurons, classes),
    torch.nn.LogSoftmax()
)
optimizer_nn = torch.optim.SGD(model_nn.parameters(), lr=0.1)

if args.cuda:
    model.cuda()
    model_nn.cuda()


def train_nn(epoch):
    model_nn.train()
    # Train network
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 2000:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        nn_output = model_nn(torch.autograd.Variable(output))
        loss = F.nll_loss(nn_output, target)
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()

        if batch_idx % (args.log_interval * 64) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def train_lissom(epoch):
    model.train()
    writer = SummaryWriter(log_dir='runs/epoch_'+str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 2000:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        optimizer.update_weights(model, step=batch_idx)
        summary_images(batch_idx, data, output, writer)

        if batch_idx % (args.log_interval * 64) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader)))
    writer.close()


def summary_images(batch_idx, data, output, writer):
    images_numpy = [x.view((1, 1) + lissom_shape) for x in
                    [output, model.afferent_activation, model.inhibitory_activation,
                     model.excitatory_activation, model.retina_activation]]
    images_numpy.append(data.data.view((1, 1) + input_shape))
    for title, im in zip(['output', 'model.afferent_activation', 'model.inhibitory_activation',
                          'model.excitatory_activation', 'model.retina_activation', 'input'], images_numpy):
        im = vutils.make_grid(im)
        writer.add_image(title, im, batch_idx)
    orig_weights = [model.inhibitory_weights, model.excitatory_weights]
    weights = [w for w in
               map(summary_weights, orig_weights)]
    weights.append(summary_weights(model.retina_weights, afferent=True))
    for title, im in zip(['model.inhibitory_weights', 'model.excitatory_weights',
                          'model.retina_weights'], weights):
        im = vutils.make_grid(im, nrow=int(np.sqrt(im.shape[0])))
        writer.add_image(title, im, batch_idx)


def summary_weights(input, afferent=False):
    shape = input_shape if afferent else lissom_shape
    input = input * shape[0]
    return torch.t(input).contiguous().data.view((input.shape[1], 1) + shape)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        nn_output = model_nn(torch.autograd.Variable(output))
        test_loss += F.nll_loss(nn_output, target, size_average=False).data[0]  # sum up batch loss
        pred = nn_output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train_lissom(epoch)

for epoch in range(1, args.epochs*10 + 1):
    train_nn(epoch)
    test()
