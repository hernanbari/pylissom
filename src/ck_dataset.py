from random import shuffle

import numpy as np

import torch
from torch.utils.data import Dataset


class CKDataset(Dataset):
    def __init__(self, train, path_images='/home/rosana/data/X.npy', path_labels='/home/rosana/data/y.npy',
                 train_pct=0.25):
        self.train_pct = train_pct
        self.path_labels = path_labels
        self.path_images = path_images
        X = np.load(self.path_images)
        y = np.load(self.path_labels)
        # subjs = np.load('/home/rosana/data/subjs.npy')

        X, y = self.shuffle(X, y)

        self.X, self.y = self._split(X, y, train)

    def __getitem__(self, item):
        return torch.Tensor(self.X[item]), int(self.y[item])

    def __len__(self):
        return len(self.X)

    @staticmethod
    def shuffle(X, y):
        x_y = list(zip(X, y))
        shuffle(list(x_y))
        return zip(*x_y)

    def _split(self, X, y, train):
        train_len = int(len(X) * self.train_pct)
        if train:
            return X[:train_len], y[:train_len]
        else:
            return X[train_len:], y[train_len:]
