import time

import torch


def save_model(model, optimizer=None, fname='best_model.pth.tar'):
    # TODO: Add asserts of model is Module and optimizer is Optimizer
    state = {'model': model}
    if optimizer is not None:
        state.update({
            'optimizer': optimizer})
    torch.save(state, fname)


def load_model(fname='model.pth.tar'):
    state = torch.load(fname)
    return state['model'], state['optimizer'] if 'optimizer' in state else None


def debug():
    r"""Calls a debugger that works with jupyter notebooks"""
    from IPython.core.debugger import Pdb
    Pdb().set_trace()


class TimeIt(object):
    r"""At instantiation starts a timer and prints value when end() is called"""
    def __init__(self):
        self.t0 = time.time()

    def end(self):
        t1 = time.time()
        print(t1 - self.t0)
