import torch
import numpy as np

# from src.supervised_gcal.utils.images import get_writer
from torch.autograd import Variable
import torch.nn.functional as F


class Pipeline(object):
    def __init__(self, model, optimizer=None, loss_fn=None, log_interval=10, dataset_len=None, cuda=False, prefix='',
                 use_writer=False):
        self.use_writer = use_writer
        self.prefix = prefix
        self.dataset_len = dataset_len
        self.log_interval = log_interval
        self.loss_fn = loss_fn
        self.cuda = cuda
        self.optimizer = optimizer
        self.model = model if not cuda else model.cuda()

    def train(self, train_data_loader, epoch):
        self.model.train()
        self.epoch = epoch
        return self._run(train_data_loader, train=True)

    def test(self, test_data_loader, epoch):
        self.model.eval()
        self.epoch = epoch
        self.test_loss = 0
        return self._run(test_data_loader, train=False)

    # TODO: check this
    @staticmethod
    def process_input(input, normalize=False):
        batch_input_shape = torch.Size((1, int(np.prod(input.data.shape))))
        var = input
        if normalize:
            var = var / torch.norm(input, p=2, dim=1)
        var = var.view(batch_input_shape)
        return var

    def _run(self, data_loader, train):
        self.correct = 0
        if self.use_writer:
            self.writer = get_writer(train=train, epoch=0, prefix=self.prefix)
        for batch_idx, (data, target) in enumerate(data_loader):
            if self.dataset_len is not None and batch_idx >= self.dataset_len:
                break
            loss = None
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=not train), Variable(target)
            data = self.process_input(data)
            self.optimizer.zero_grad() if self.optimizer else None
            output = self.model(data)
            if self.loss_fn:
                loss = self.loss_fn(output, target)
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                self.correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if train:
                if self.loss_fn:
                    if self.use_writer:
                        self.writer.add_scalar('loss', loss.data[0],
                                           global_step=batch_idx + len(data_loader) * (self.epoch - 1))
                    loss.backward()
                self.optimizer.step() if self.optimizer else None
                if batch_idx % self.log_interval == 0:
                    self._train_log(batch_idx, data_loader, loss)
            elif self.loss_fn:
                self.test_loss += loss.data[0]  # sum up batch loss

        if self.loss_fn:
            if not train:
                self._test_log(data_loader)
            if self.use_writer:
                self.writer.add_scalar('accuracy', self.accuracy(data_loader), global_step=self.epoch-1)
            return self.accuracy(data_loader)
        return None

    def _test_log(self, data_loader):
        self.test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
            self.test_loss, self.accuracy(data_loader)))

    def accuracy(self, data_loader):
        return 100. * self.correct / len(data_loader)

    def _train_log(self, batch_idx, data_loader, loss):
        if batch_idx % self.log_interval == 0:
            print('Train Epoch: {} Iterations: {:.0f}%'.format(
                self.epoch,
                100. * batch_idx / len(data_loader)))
            if self.loss_fn:
                print('Accuracy: {:.0f}% Loss: {:.6f}'.format(
                    self.accuracy(data_loader), loss.data[0]))
