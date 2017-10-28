from collections import OrderedDict

import torch
from src.supervised_gcal.layers.modules.complex import NormalizeDecorator, CircularMaskDecorator, AfferentNorm
from src.supervised_gcal.layers.modules.simple import GaussianCloudLinear, DifferenceOfGaussiansLinear, \
    PiecewiseSigmoid
from src.supervised_gcal.utils.functions import check_compatible_mul, check_compatible_add
from src.supervised_gcal.utils.math import normalize
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights


class Cortex(GaussianCloudLinear):
    def __init__(self, in_features, out_features, radius, sigma=None):
        super(Cortex, self).__init__(in_features, out_features, sigma=sigma)
        self.weight.data = normalize(
            apply_circular_mask_to_weights(self.weight.data,
                                           radius=radius))


class AfferentNormCortex(torch.nn.Sequential):
    def __init__(self, in_features, out_features, radius, aff_norm_strength, sigma=None,
                 cortex_cls=Cortex, aff_norm_cls=AfferentNorm):
        self.sigma = sigma
        self.aff_norm_strength = aff_norm_strength
        self.radius = radius
        self.out_features = out_features
        self.in_features = in_features
        layers = OrderedDict(
            {'cortex': cortex_cls(in_features=in_features, out_features=out_features, radius=radius, sigma=sigma),
             'aff_norm': aff_norm_cls(aff_norm_strength=aff_norm_strength, radius=radius)})
        super(AfferentNormCortex, self).__init__(layers)


class DiffOfGaussians(DifferenceOfGaussiansLinear):
    def __init__(self, in_features, out_features, on, radius, sigma_surround,
                 sigma_center=None):
        super(DiffOfGaussians, self).__init__(in_features=in_features, out_features=out_features, on=on,
                                              sigma_surround=sigma_surround, sigma_center=sigma_center)
        self.radius = radius
        self.out_features = out_features
        self.in_features = in_features
        self.weight = NormalizeDecorator(
            CircularMaskDecorator(self.weight.data,
                                  r=radius))


class LGN(torch.nn.Sequential):
    def __init__(self, in_features, out_features, on, radius, sigma_surround,
                 sigma_center=None, min_theta=None, max_theta=None,
                 diff_of_gauss_cls=DiffOfGaussians, pw_sigmoid_cls=PiecewiseSigmoid):
        self.max_theta = max_theta
        self.min_theta = min_theta
        layers = OrderedDict({
            'diff_of_gaussians': diff_of_gauss_cls(in_features=in_features, out_features=out_features, on=on,
                                                   radius=radius, sigma_center=sigma_center,
                                                   sigma_surround=sigma_surround),
            'piecewise_sigmoid': pw_sigmoid_cls(min_theta=min_theta, max_theta=max_theta)})
        super(LGN, self).__init__(layers)


class ReducedLissom(torch.nn.Module):
    def __init__(self, afferent_module, excitatory_module, inhibitory_module,
                 min_theta=1.0, max_theta=1.0, settling_steps=10):
        super(ReducedLissom, self).__init__()
        check_compatible_mul(afferent_module, inhibitory_module)
        check_compatible_mul(afferent_module, excitatory_module)
        self.inhibitory_module = inhibitory_module
        self.excitatory_module = excitatory_module
        self.afferent_module = afferent_module
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.settling_steps = settling_steps
        self.piecewise_sigmoid = PiecewiseSigmoid(min_theta=min_theta, max_theta=max_theta)

    def forward(self, cortex_input):
        afferent_activation = self.piecewise_sigmoid(self.afferent_module(self.input))
        current_activation = afferent_activation.clone()
        for _ in range(self.settling_steps):
            excitatory_activation = self.excitatory_module(current_activation)
            inhibitory_activation = self.inhibitory_module(current_activation)

            sum_activations = afferent_activation + excitatory_activation - inhibitory_activation

            current_activation = self.piecewise_sigmoid(sum_activations)
        return current_activation


class Lissom(torch.nn.Module):
    def __init__(self, on, off, v1):
        super(Lissom, self).__init__()
        check_compatible_add(on, off)
        check_compatible_mul(on, v1)
        self.v1 = v1
        self.off = off
        self.on = on

    def forward(self, input):
        on_output = self.on(input)
        off_output = self.off(input)
        lgn_activation = on_output + off_output
        activation = self.v1(lgn_activation)
        return activation
