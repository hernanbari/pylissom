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


def afferent_normalize(radius, afferent_normalization_strength, stimulus, weighted_sum):
    reshaped_input = stimulus.data.repeat(stimulus.data.shape[1], 1)
    masked_input = apply_circular_mask_to_weights(reshaped_input, radius)
    sums = masked_input.sum(1).unsqueeze(1).t()
    den = 1 + afferent_normalization_strength * sums
    weighted_sum = weighted_sum / den
    return weighted_sum


def piecewise_sigmoid(min_theta, max_theta, activation):
    mask_zeros = torch.le(activation, min_theta)
    mask_ones = torch.ge(activation, max_theta)
    activation.sub_(min_theta).div_(max_theta - min_theta)
    activation.masked_fill_(mask=mask_zeros, value=0)
    activation.masked_fill_(mask=mask_ones, value=1)
    return activation