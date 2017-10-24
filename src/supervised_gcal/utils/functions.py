import time

import torch
from src.supervised_gcal.utils.math import normalize
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights


def linear_decay(w, start, epoch, final_epoch):
    radius = start + epoch * (1.0 - start) / final_epoch
    normalize(apply_circular_mask_to_weights(w.data.t_(), radius=radius))
    w.data.t_()
    return


def kill_neurons(w, threshold):
    return w.masked_fill_(mask=torch.lt(w, threshold), value=0)


class TimeIt(object):
    def __init__(self):
        self.t0 = time.time()

    def end(self):
        t1 = time.time()
        print(t1 - self.t0)


def afferent_normalize(radius, strength, afferent_input, activation):
    reshaped_input = afferent_input.data.repeat(afferent_input.data.shape[1], 1)
    masked_input = apply_circular_mask_to_weights(reshaped_input, radius)
    sums = masked_input.sum(1).unsqueeze(1).t()
    den = 1 + strength * sums
    activation = activation / den
    return activation


def piecewise_sigmoid(min_theta, max_theta, input):
    mask_zeros = torch.le(input, min_theta)
    mask_ones = torch.ge(input, max_theta)
    input.sub_(min_theta).div_(max_theta - min_theta)
    input.masked_fill_(mask=mask_zeros, value=0)
    input.masked_fill_(mask=mask_ones, value=1)
    return input