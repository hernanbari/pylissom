import torch
from src.ck_dataset import CKDataset
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_dataset(train, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'ck':
        return torch.utils.data.DataLoader(CKDataset(train=train), batch_size=args.batch_size, shuffle=False,
                                           **kwargs)
    elif args.dataset == 'mnist':
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'number_one':
        raise NotImplementedError
