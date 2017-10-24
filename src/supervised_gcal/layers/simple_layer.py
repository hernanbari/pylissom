from collections import OrderedDict

import torch
from src.supervised_gcal.layers.abstract_layer import AbstractLayer
from src.supervised_gcal.utils.functions import afferent_normalize, piecewise_sigmoid
from src.supervised_gcal.utils.weights import get_gaussian_weights_variable, apply_circular_mask_to_weights


class SimpleLayer(AbstractLayer):
    def __init__(self, input_shape, self_shape, min_theta=0.0, max_theta=1.0, strength=1.0, radius=1.0,
                 afferent_normalization=False, sparse=False, name=''):
        self.sparse = sparse
        self.radius = radius
        self.strength = strength
        self.name = name
        self.weights = None
        self.afferent_normalization = afferent_normalization
        self.max_theta = max_theta
        self.min_theta = min_theta
        super(SimpleLayer, self).__init__(input_shape, self_shape)

    def forward(self, stimulus):
        self.input = stimulus
        weighted_sum = self.strength * self.matmul(stimulus.data, self.weights.data)
        if self.afferent_normalization:
            weighted_sum = afferent_normalize(self.radius, self.afferent_normalization_strength, stimulus, weighted_sum)
        activation = piecewise_sigmoid(self.min_theta, self.max_theta, weighted_sum)
        self.activation = torch.autograd.Variable(activation)
        return self.activation

    def matmul(self, vector, matrix):
        if not self.sparse:
            return torch.matmul(vector, matrix)
        else:
            # Pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense,
            # That's why it's reversed
            return torch.matmul(matrix.t(), vector.t()).t()


# class GaussianLinear(torch.nn.Linear):
#     def __init__(self, in_features, out_features, sigma=1.0):
#         super(GaussianLinear).__init__(in_features, out_features)
#         self.sigma = sigma
#         self.weight = torch.nn.Parameter(get_gaussian_weights_variable(input_shape=in_features,
#                                                                        output_shape=out_features,
#                                                                        sigma=sigma))
#
#
# class GaussianCloudLinear(torch.nn.Linear):
#     def __init__(self, in_features, out_features, sigma=1.0):
#         super(GaussianCloudLinear, self).__init__(in_features, out_features)
#         self.sigma = sigma
#         weight = get_gaussian_weights_variable(input_shape=in_features,
#                                                output_shape=out_features,
#                                                sigma=sigma)
#         uniform_noise = torch.FloatTensor(weight.size()).uniform_(0, 1)
#         self.weight = torch.nn.Parameter(weight * uniform_noise)
#
#
# class DifferenceOfGaussiansLinear(torch.nn.Linear):
#     def __init__(self, in_features, out_features, on, sigma_surround, sigma_center=1.0):
#         super(DifferenceOfGaussiansLinear).__init__(in_features, out_features)
#         self.on = on
#         self.sigma_surround = sigma_surround
#         self.sigma_center = sigma_center
#         sigma_center_weights_matrix = get_gaussian_weights_variable(in_features, out_features,
#                                                                     sigma_center)
#         sigma_surround_weights_matrix = get_gaussian_weights_variable(in_features, out_features,
#                                                                       sigma_surround)
#         if on:
#             diff = (sigma_center_weights_matrix.weight.data - sigma_surround_weights_matrix.weight.data).t()
#         else:
#             diff = (sigma_surround_weights_matrix.weight.data - sigma_center_weights_matrix.weight.data).t()
#         self.weight = torch.nn.Parameter(diff)
#
#
# class PiecewiseSigmoid(torch.nn.Module):
#     def __init__(self, min_theta=0.0, max_theta=1.0):
#         super(PiecewiseSigmoid, self).__init__()
#         self.max_theta = max_theta
#         self.min_theta = min_theta
#
#     def forward(self, input):
#         return piecewise_sigmoid(min_theta=self.min_theta, max_theta=self.max_theta, input=input)
#
#
# class
#
#
# class AfferentNorm(torch.nn.Module):
#     def __init__(self, strength):
#         super(AfferentNorm, self).__init__()
#         self.strength = strength
#
#     def forward(self, afferent_input, activation):
#         return afferent_normalize(strength=self.strength,
#                                   afferent_input=afferent_input,
#                                   activation=activation)
#
#
# class CircularMask(torch.nn.Module):
#     def __init__(self, radius):
#         super(CircularMask, self).__init__()
#         self.radius = radius
#
#     def forward(self, input):
#         # TODO: change to not in_place
#         apply_circular_mask_to_weights(matrix=input, radius=self.radius)
#
#
# class Cortex(torch.nn.Sequential):
#     def __init__(self, in_features, out_features, radius, aff_norm_strength=None, min_theta=None, sigma=None, max_theta=None):
#         self.max_theta = max_theta
#         self.sigma = sigma
#         self.min_theta = min_theta
#         self.aff_norm_strength = aff_norm_strength
#         self.radius = radius
#         self.out_features = out_features
#         self.in_features = in_features
#         layers = OrderedDict()
#         layers['gaussian_cloud'] = GaussianCloudLinear(in_features=in_features, out_features=out_features,
#                                                         sigma=sigma)
#         layers['circular_mask'] = CircularMask(radius=radius)
#         if aff_norm_strength is not None:
#             layers['aff_norm'] = AfferentNorm(strength=aff_norm_strength)
#         layers['piecewise_sigmoid'] = PiecewiseSigmoid(min_theta=min_theta, max_theta=max_theta)
#         super(Cortex, self).__init__(layers)
