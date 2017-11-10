import numpy as np
from tensorboard import SummaryWriter

import torch
from src.supervised_gcal.layers.modules.lissom import DifferenceOfGaussiansLinear
from src.supervised_gcal.layers.modules.simple import UnnormalizedDifferenceOfGaussiansLinear
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


def plot_layer_weights(layer, use_range=True, recursive=False, prefix=''):
    if use_range:
        values_range = (-1, 1) if isinstance(layer, DifferenceOfGaussiansLinear) \
                                  or isinstance(layer, UnnormalizedDifferenceOfGaussiansLinear) else (0, 1)
    else:
        values_range = None
    plot_dict_matrix({prefix + '.' + k: weights_to_numpy_matrix(w.data, values_range) for k, w in layer.named_parameters()})
    if recursive:
        for k, c in layer.named_children():
            plot_layer_weights(c, prefix=k + '.')
    return


def tensor_to_numpy_matrix(tensor, shape):
    tensor = tensor.cpu() if tensor.is_cuda else tensor
    return np.reshape(tensor.numpy(), shape)


def plot_tensor(tensor, shape, vmin=0, vmax=1):
    img = tensor_to_numpy_matrix(tensor, shape)
    plot_matrix(img, vmin, vmax)


def simple_plot_layer_activation(layer, prefix=''):
    try:
        input_rows = int(layer.in_features ** 0.5)
        output_rows = int(layer.out_features ** 0.5)
        inp_mat = tensor_to_numpy_matrix(layer.input.data, (input_rows, input_rows))
        act_mat = tensor_to_numpy_matrix(layer.output.data, (output_rows, output_rows))
        plot_dict_matrix(dict([(prefix + '.' + 'input', inp_mat), (prefix + '.' + 'activation', act_mat)]))
    except Exception:
        pass


def named_apply(module, fn, prefix):
    for k, m in module.named_children():
        named_apply(m, fn, prefix + '.' + k)
    fn(module, prefix)


def plot_layer_activation(layer, prefix=''):
    named_apply(layer, simple_plot_layer_activation, prefix)


def plot_cortex_activations(cortex):
    inp_act_plots = plot_layer_activation(cortex)
    activations = [tensor_to_numpy_matrix(act, cortex.self_shape) for act in [cortex.afferent_activation,
                                                                              cortex.inhibitory_activation,
                                                                              cortex.excitatory_activation]]
    return inp_act_plots + plot_array_matrix(activations)
