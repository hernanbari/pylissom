import numpy as np
from tensorboard import SummaryWriter

import torch
from src.supervised_gcal.layers.lgn_layer import LGNLayer
from src.supervised_gcal.layers.modules.lissom import DiffOfGaussians
from src.supervised_gcal.layers.modules.simple import DifferenceOfGaussiansLinear
from src.utils.images import plot_array_matrix, plot_dict_matrix, plot_matrix
from torchvision import utils as vutils

logdir = 'runs'
log_interval = 10


def generate_images(self, input, output, prefix=''):
    if self.batch_idx % log_interval != 0:
        self.batch_idx += 1
        return
    writer = get_writer(self.training, self.epoch, prefix)

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


def get_writer(train, epoch, prefix):
    log_path = logdir \
               + ('/' + prefix + '/' if prefix != '' else '') \
               + ('/train/' if train else '/test/') \
               + 'epoch_' + str(epoch)
    writer = SummaryWriter(log_dir=log_path)
    return writer


def images_matrix(matrix, range=None):
    out_features = matrix.shape[0]
    in_features = matrix.shape[1]
    weights_shape = (int(np.sqrt(in_features)), int(np.sqrt(in_features)))
    reshaped_weights = matrix.contiguous().view((out_features, 1) + weights_shape)
    im = vutils.make_grid(reshaped_weights, normalize=True, nrow=int(np.sqrt(reshaped_weights.shape[0])), range=range,
                          pad_value=0.5 if range is not None and range[0] >= 0 else 0)
    return im


def weights_to_numpy_matrix(weights, values_range):
    grid = images_matrix(weights, values_range)
    grid = grid.cpu() if grid.is_cuda else grid
    return np.transpose(grid.numpy(), (1, 2, 0))


def plot_layer_weights(layer, use_range=True, prefix=''):
    if use_range:
        values_range = (-1, 1) if isinstance(layer, DifferenceOfGaussiansLinear) \
                                  or isinstance(layer, DiffOfGaussians) else (0, 1)
    else:
        values_range = None
    plot_dict_matrix({prefix + k: weights_to_numpy_matrix(w.data, values_range) for k, w in layer.named_parameters()})
    for k, c in layer.named_children():
        plot_layer_weights(c, prefix=k + '.')
    return


def tensor_to_numpy_matrix(tensor, shape):
    tensor = tensor.cpu() if tensor.is_cuda else tensor
    return np.reshape(tensor.numpy(), shape)


def plot_tensor(tensor, shape):
    img = tensor_to_numpy_matrix(tensor, shape)
    plot_matrix(img)


def plot_layer_activation(layer, prefix=''):
    try:
        inp_mat = tensor_to_numpy_matrix(layer.input.data, layer.input_shape)
        act_mat = tensor_to_numpy_matrix(layer.activation.data, layer.self_shape)
        plot_dict_matrix(dict([(prefix + 'input', inp_mat), (prefix + 'activation', act_mat)]))
    except Exception:
        pass
    for k, c in layer.named_children():
        plot_layer_activation(c, prefix=k + '.')
    return


def plot_cortex_activations(cortex):
    inp_act_plots = plot_layer_activation(cortex)
    activations = [tensor_to_numpy_matrix(act, cortex.self_shape) for act in [cortex.afferent_activation,
                                                                              cortex.inhibitory_activation,
                                                                              cortex.excitatory_activation]]
    return inp_act_plots + plot_array_matrix(activations)
