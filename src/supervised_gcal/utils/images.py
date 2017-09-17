import numpy as np
from tensorboard import SummaryWriter

import torch
from torchvision import utils as vutils

logdir = 'runs'
log_interval = 10


def generate_images(self, input, output):
    if not self.training:
        return
    if self.batch_idx % log_interval != 0:
        self.batch_idx += 1
        return
    writer = SummaryWriter(log_dir=logdir + '/epoch_' + str(self.epoch))
    for name, mat in self.weights + [('activation', self.activation.t()), ('input', input[0].t()), ('output', output.t()),
                                     ('self.input', self.input.t())]:
        if isinstance(mat, torch.autograd.Variable):
            mat = mat.data
        image = images_matrix(mat)
        title = self.name + '_' + name
        writer.add_image(title, image, self.batch_idx)
    writer.close()
    self.batch_idx += 1


def images_matrix(matrix):
    neurons = matrix.shape[1]
    input_neurons = matrix.shape[0]
    weights_shape = (int(np.sqrt(input_neurons)), int(np.sqrt(input_neurons)))
    reshaped_weights = matrix.t().contiguous().view((neurons, 1) + weights_shape)
    im = vutils.make_grid(reshaped_weights, nrow=int(np.sqrt(reshaped_weights.shape[0])), range=(0, 1))
    return im
