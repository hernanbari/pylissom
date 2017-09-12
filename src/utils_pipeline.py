import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import utils as vutils, datasets, transforms


# def tensor_to_im(tensor, shape, writer):
#     tensor = tensor.view((1, 1), shape)
#     im = vutils.make_grid(tensor, range=(0, 1))
#     writer.add_image()

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
               map(lambda w: summary_weights(input_shape, lissom_shape, w), orig_weights)]
    weights.append(
        summary_weights(input_shape=input_shape, lissom_shape=lissom_shape, weights=lissom_model.retina_weights,
                        afferent=True))
    for title, im in zip(['model.inhibitory_weights', 'model.excitatory_weights',
                          'model.retina_weights'], weights):
        im = vutils.make_grid(im, nrow=int(np.sqrt(im.shape[0])), range=(0, 1))
        writer.add_image(title, im, batch_idx)


def summary_weights(input_shape, lissom_shape, weights, afferent=False):
    shape = input_shape if afferent else lissom_shape
    weights = weights * shape[0]
    return torch.t(weights).data.to_dense().view((weights.shape[1], 1) + shape)


def get_dataset(train, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.ck:
        return torch.utils.data.DataLoader(CKDataset(train=train), batch_size=args.batch_size, shuffle=False,
                                           **kwargs)
    else:
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)


class CKDataset(Dataset):
    def __init__(self, train, path_images='/home/rosana/data/X.npy', path_labels='/home/rosana/data/y.npy',
                 train_pct=0.75):
        self.train_pct = train_pct
        self.path_labels = path_labels
        self.path_images = path_images
        X = np.load(self.path_images)
        y = np.load(self.path_labels)
        # subjs = np.load('/home/rosana/data/subjs.npy')

        self.x_shape = X[0].shape

        train_len = int(len(X) * self.train_pct)
        if train:
            self.X = X[:train_len]
            self.y = y[:train_len]
        else:
            self.X = X[train_len:]
            self.y = y[train_len:]

    def __getitem__(self, item):
        return torch.Tensor(self.X[item]), int(self.y[item])

    def __len__(self):
        return len(self.X)


def summary_lgn(lgn_layer, input_shape, lissom_shape, batch_idx, data, output, writer, name):
    im = summary_weights(input_shape, lissom_shape, lgn_layer.weights, afferent=True)
    im = vutils.make_grid(im, nrow=int(np.sqrt(im.shape[0])), range=(0, 1))
    writer.add_image('lgn weights_'+name, im, batch_idx)
    im = vutils.make_grid(data.data, range=(0, 1))
    writer.add_image('input_lgn_'+name, im, batch_idx)
    im = vutils.make_grid(output.view(1, 1, lissom_shape[0], lissom_shape[0]).data, range=(0, 1))
    writer.add_image('lgn activation_'+name, im, batch_idx)