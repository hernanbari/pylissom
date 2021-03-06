import torch
from torch.nn import Linear, Module

from pylissom.nn.functional.functions import piecewise_sigmoid
from pylissom.nn.functional.weights import get_gaussian_weights, apply_circular_mask_to_weights


class GaussianLinear(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    where A is a Gaussian matrix

    Parameters:
        - **sigma** -
    """
    # ASSUMES SQUARE MAPS
    def __init__(self, in_features, out_features, sigma=1.0):
        super(GaussianLinear, self).__init__(in_features, out_features, bias=False)
        self.sigma = sigma
        self.weight = torch.nn.Parameter(get_gaussian_weights(in_features=in_features,
                                                              out_features=out_features,
                                                              sigma=sigma))

    def __repr__(self):
        super_repr = super(GaussianLinear, self).__repr__()[:-1]
        return super_repr \
               + ', ' + 'sigma=' + str(self.sigma) \
               + ')'


class GaussianCloudLinear(GaussianLinear):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    where A is a Gaussian matrix multiplied with Gaussian Noise

    Parameters:
        - **sigma** -
    """
    # ASSUMES SQUARE MAPS
    def __init__(self, in_features, out_features, sigma=1.0):
        super(GaussianCloudLinear, self).__init__(in_features, out_features, sigma=sigma)
        uniform_noise = torch.FloatTensor(self.weight.data.size()).uniform_(0, 1)
        self.weight.data.mul_(uniform_noise)


class PiecewiseSigmoid(Module):
    r"""Applies a piecewise approximation of the sigmoid function :math:`f(x) = 1 / ( 1 + exp(-x))`

    The formula is as follows:
    TODO
    Parameters:
        - **min_theta** -
        - **max_theta** -

    """
    def __init__(self, min_theta=0.0, max_theta=1.0):
        super(PiecewiseSigmoid, self).__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta

    def forward(self, input):
        # TODO: add inplace option
        return piecewise_sigmoid(min_theta=self.min_theta, max_theta=self.max_theta, inp=input)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'min_theta=' + str(self.min_theta) \
               + ', ' + 'max_theta=' + str(self.max_theta) \
               + ')'


# class AfferentNorm(Module):
#     def __init__(self, strength, radius):
#         super(AfferentNorm, self).__init__()
#         self.radius = radius
#         self.strength = strength
#
#     def forward(self, afferent_input, activation):
#         # TODO: add inplace option
#         return afferent_normalize(strength=self.strength,
#                                   afferent_input=afferent_input,
#                                   activation=activation,
#                                   radius=self.radius)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + 'radius=' + str(self.radius) \
#                + ', ' + 'strength=' + str(self.strength) \
#                + ')'

#
# # TODO: test, remove if not used
# class CircularMask(Module):
#     def __init__(self, radius):
#         super(CircularMask, self).__init__()
#         self.radius = radius
#
#     def forward(self, input):
#         # TODO: add inplace option
#         apply_circular_mask_to_weights(matrix=input, radius=self.radius)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + 'radius=' + str(self.radius) \
#                + ')'


class UnnormalizedDifferenceOfGaussiansLinear(Linear):
    r"""NOT USED, only for example in notebooks"""
    # Not used in lissom, only for example
    # ASSUMES SQUARE MAPS
    def __init__(self, in_features, out_features, on, sigma_surround, sigma_center=1.0):
        super(UnnormalizedDifferenceOfGaussiansLinear, self).__init__(in_features, out_features, bias=False)
        self.on = on
        self.sigma_surround = sigma_surround
        self.sigma_center = sigma_center
        sigma_center_weights_matrix = get_gaussian_weights(in_features, out_features,
                                                           sigma_center)
        sigma_surround_weights_matrix = get_gaussian_weights(in_features, out_features,
                                                             sigma_surround)
        if on:
            diff = sigma_center_weights_matrix - sigma_surround_weights_matrix
        else:
            diff = sigma_surround_weights_matrix - sigma_center_weights_matrix
        self.weight = torch.nn.Parameter(diff)
