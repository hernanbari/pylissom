import torch

from src.math import normalize
from src.nn.utils.weights import apply_circular_mask_to_weights


def linear_decay(w, start, epoch, final_epoch):
    radius = start + epoch * (1.0 - start) / final_epoch
    normalize(apply_circular_mask_to_weights(w.data.t_(), radius=radius))
    w.data.t_()
    return


def kill_neurons(w, threshold):
    return w.masked_fill_(mask=torch.lt(w, threshold), value=0)


# TODO: test
def afferent_normalize(radius, strength, afferent_input, activation):
    reshaped_input = afferent_input.data.repeat(afferent_input.data.size()[1], 1)
    masked_input = apply_circular_mask_to_weights(reshaped_input, radius)
    sums = masked_input.sum(1).unsqueeze(1).t()
    den = 1 + strength * sums
    activation = activation / den
    return activation


def piecewise_sigmoid(min_theta, max_theta, input):
    mask_zeros = torch.le(input, min_theta)
    mask_ones = torch.ge(input, max_theta)
    output = (input - min_theta).div(max_theta - min_theta)
    output.masked_fill_(mask=mask_zeros, value=0)
    output.masked_fill_(mask=mask_ones, value=1)
    return output


def check_compatible_mul(module_one, module_two):
    if module_one.out_features != module_two.in_features:
        raise ValueError(
            "Matmul: {}.out_features doesn't match {}.in_features".format(str(module_one), str(module_two)))


def check_compatible_add(module_one, module_two):
    if module_one.out_features != module_two.out_features:
        raise ValueError(
            "Add: {}.out_features doesn't match {}.out_features".format(str(module_one), str(module_two)))
