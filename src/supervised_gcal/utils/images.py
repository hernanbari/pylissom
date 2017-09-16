import types

import numpy as np
from tensorboard import SummaryWriter

from src.supervised_gcal.layer import Layer
from torchvision import utils as vutils

logdir = 'runs'


def image_decorator(Cls, logdir=logdir):
    class ImagesWrapper(object):
        def __init__(self, *args, **kwargs):
            self.wrapped_layer = Cls(*args, **kwargs)
            assert isinstance(self.wrapped_layer, Layer)
            self.batch_idx = 0
            self.epoch = 0

        def __getattr__(self, name):
            """
            this is called whenever any attribute of a ImagesWrapper object is accessed. This function returns the
            attribute from self.wrapped_layer (an instance of the decorated class).
            """
            return self.wrapped_layer.__getattribute__(name)

        def forward(self, *args, **kwargs):
            self.wrapped_layer.forward(*args, **kwargs)
            if self.epoch != self.wrapped_layer.epoch:
                self.epoch = self.wrapped_layer.epoch
                self.batch_idx = 0
            writer = SummaryWriter(log_dir=logdir + '/epoch_' + str(self.epoch))
            title = self.wrapped_layer.name
            for name, mat in self.wrapped_layer.weights + [('activation', self.wrapped_layer.activation.data.t())]:
                image = images_matrix(mat)
                title += '/' + name
                writer.add_image(title, image, self.batch_idx)
            writer.close()
            self.batch_idx += 1

    return ImagesWrapper


def images_matrix(matrix):
    neurons = matrix.shape[1]
    input_neurons = matrix.shape[0]
    weights_shape = (int(np.sqrt(input_neurons)), int(np.sqrt(input_neurons)))
    reshaped_weights = matrix.t().contiguous().view((neurons, 1) + weights_shape)
    im = vutils.make_grid(reshaped_weights, nrow=int(np.sqrt(reshaped_weights.shape[0])), range=(0, 1))
    return im


def reshape_weights(input_shape, lissom_shape, weights, afferent=False):
    shape = input_shape if afferent else lissom_shape
    weights = weights * shape[0]
    return weights.t().contiguous().view((weights.shape[1], 1) + shape)


def summary_images(lissom_model, batch_idx, data, output, writer):
    lissom_shape = lissom_model.self_shape
    images_numpy = [x.view((1, 1) + lissom_shape) for x in
                    [output, lissom_model.afferent_activation, lissom_model.inhibitory_activation,
                     lissom_model.excitatory_activation, lissom_model.retina_activation]]
    input_shape = lissom_model.orig_input_shape
    images_numpy.append(data.data.view((1, 1) + input_shape))
    for title, im in zip(['output', 'model.afferent_activation', 'model.inhibitory_activation',
                          'model.excitatory_activation', 'model.retina_activation', 'input'], images_numpy):
        im = vutils.make_grid(im, range=(0, 1))
        writer.add_image(title, im, batch_idx)
    orig_weights = [lissom_model.inhibitory_weights, lissom_model.excitatory_weights]
    weights = [w for w in
               map(lambda w: reshape_weights(input_shape, lissom_shape, w), orig_weights)]
    weights.append(
        reshape_weights(input_shape=input_shape, lissom_shape=lissom_shape, weights=lissom_model.retina_weights,
                        afferent=True))
    for title, im in zip(['model.inhibitory_weights', 'model.excitatory_weights',
                          'model.retina_weights'], weights):
        im = vutils.make_grid(im, nrow=int(np.sqrt(im.shape[0])), range=(0, 1))
        writer.add_image(title, im, batch_idx)


def summary_lgn(lgn_layer, input_shape, lissom_shape, batch_idx, data, output, writer, name):
    reshaped_weights = reshape_weights(input_shape, lissom_shape, lgn_layer.weights.data, afferent=True)
    im = vutils.make_grid(reshaped_weights, nrow=int(np.sqrt(reshaped_weights.shape[0])), range=(0, 1))
    writer.add_image('lgn weights_' + name, im, batch_idx)
    im = vutils.make_grid(data.data, range=(0, 1))
    writer.add_image('input_lgn_' + name, im, batch_idx)
    im = vutils.make_grid(output.data.view(1, 1, lissom_shape[0], lissom_shape[0]), range=(0, 1))
    writer.add_image('lgn activation_' + name, im, batch_idx)
