import numpy as np
from tensorboard import SummaryWriter

import torch
from src.supervised_gcal.lgn_layer import LGNLayer
from torchvision import utils as vutils

logdir = 'runs'
log_interval = 10


def generate_images(self, input, output):
    if self.batch_idx % log_interval != 0:
        self.batch_idx += 1
        return
    writer = SummaryWriter(log_dir=logdir + ('/train/' if self.training else '/test/') + 'epoch_' + str(self.epoch))

    for name, mat in self.weights + [('activation', self.activation.t()), ('input', self.input.t())]:
        if isinstance(mat, torch.autograd.Variable):
            mat = mat.data
        if 'weights' in name and self.sparse:
            mat = mat.to_dense()
        if isinstance(self, LGNLayer) and 'weights' in name:
            range = (-1, 1)
        else:
            range = (0, 1)
        image = images_matrix(mat, range)
        title = self.name + '_' + name
        writer.add_image(title, image, self.batch_idx)
    writer.close()
    self.batch_idx += 1


def images_matrix(matrix, range):
    neurons = matrix.shape[1]
    input_neurons = matrix.shape[0]
    weights_shape = (int(np.sqrt(input_neurons)), int(np.sqrt(input_neurons)))
    reshaped_weights = matrix.t().contiguous().view((neurons, 1) + weights_shape)
    im = vutils.make_grid(reshaped_weights, normalize=True, nrow=int(np.sqrt(reshaped_weights.shape[0])), range=range)
    return im
