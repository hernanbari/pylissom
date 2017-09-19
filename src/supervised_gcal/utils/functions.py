import time

import torch
from src.supervised_gcal.utils.math import normalize
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights


def linear_decay(w, start, epoch, final_epoch):
    radius = start + epoch * (1.0 - start) / final_epoch
    w.data = torch.Tensor(normalize(apply_circular_mask_to_weights(w.data.cpu().numpy(), radius=radius)))
    w.data = w.data.cuda()
    return


def kill_neurons(w, threshold):
    return w.masked_fill_(mask=torch.lt(w, threshold), value=0)


class TimeIt(object):
    def __init__(self):
        self.t0 = time.time()

    def end(self):
        t1 = time.time()
        print(t1 - self.t0)
