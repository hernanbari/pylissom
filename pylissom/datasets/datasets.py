from random import shuffle

import numpy as np

import torch
from pylissom.utils.stimuli import random_gaussians_generator, faces_generator
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

Dataset.__module__ = 'torch.utils.data'


def get_dataset(train, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'ck':
        dataset = CKDataset()
        return DataLoader(CKDataset(), batch_size=args.batch_size,
                          sampler=train_test_CK_samplers(dataset, train),
                          **kwargs)
    elif args.dataset == 'mnist':
        return DataLoader(
            datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, **kwargs)
    elif args.dataset == 'number_one':
        raise NotImplementedError


class RandomDataset(Dataset):
    def __init__(self, length):
        self._lenght = length

    def __len__(self):
        return self._lenght

    def __getitem__(self, item):
        # Foo target
        return torch.from_numpy(next(self.gen)), torch.Tensor(2)

    @property
    def gen(self):
        raise NotImplementedError


class OrientatedGaussians(RandomDataset):
    @property
    def gen(self):
        return random_gaussians_generator(self.size, len(self), self.gaussians)

    def __init__(self, size, length, gaussians=2):
        super(OrientatedGaussians, self).__init__(length)
        self.gaussians = gaussians
        self.size = size


class ThreeDotFaces(RandomDataset):
    @property
    def gen(self):
        return faces_generator(self.size, self.faces)

    def __init__(self, size, length, faces=2):
        super(ThreeDotFaces, self).__init__(length)
        self.faces = faces
        self.size = size


class CKDataset(Dataset):
    def __init__(self, path_images='/home/hbari/data/X.npy', path_labels='/home/hbari/data/y.npy',
                 path_subjects='/home/hbari/data/subjs.npy'):
        self.path_labels = path_labels
        self.path_images = path_images
        self.X = np.load(self.path_images)
        # Substract 1 bc labels are 1-7 and need to start from 0
        self.y = np.load(self.path_labels) - 1
        self.subjs = np.load(path_subjects)

    def __getitem__(self, item):
        return torch.Tensor(self.X[item]), int(self.y[item])

    def __len__(self):
        return len(self.X)


def train_test_CK_samplers(ck_dataset, train, train_pct=0.5):
    train_idxs, test_idxs = subj_indep_train_test_samplers(ck_dataset.subjs, pct=train_pct)
    if train:
        return SubsetRandomSampler(train_idxs)
    else:
        return SubsetRandomSampler(test_idxs)


def subj_indep_train_test_samplers(subjs, pct):
    set_subjs = list(set(subjs))
    shuffle(set_subjs)
    split = int(len(set_subjs) * pct)
    train_subjs = set_subjs[:split]

    train_idxs = []
    test_idxs = []
    for idx, subj in enumerate(subjs):
        if subj in train_subjs:
            train_idxs.append(idx)
        else:
            test_idxs.append(idx)
    return train_idxs, test_idxs
