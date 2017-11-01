import torch
from src.supervised_gcal.utils.functions import piecewise_sigmoid, afferent_normalize
from src.supervised_gcal.utils.weights import get_gaussian_weights, apply_circular_mask_to_weights


class GaussianLinear(torch.nn.Linear):
    # ASSUMES SQUARE MAPS
    def __init__(self, in_features, out_features, sigma=1.0):
        super(GaussianLinear, self).__init__(in_features, out_features, bias=False)
        self.sigma = sigma
        self.weight = torch.nn.Parameter(get_gaussian_weights(in_features=in_features,
                                                              out_features=out_features,
                                                              sigma=sigma))


class GaussianCloudLinear(torch.nn.Linear):
    # ASSUMES SQUARE MAPS
    def __init__(self, in_features, out_features, sigma=1.0):
        super(GaussianCloudLinear, self).__init__(in_features, out_features, bias=False)
        self.sigma = sigma
        weight = get_gaussian_weights(in_features=in_features,
                                      out_features=out_features,
                                      sigma=sigma)
        uniform_noise = torch.FloatTensor(weight.size()).uniform_(0, 1)
        self.weight = torch.nn.Parameter(weight * uniform_noise)


class DifferenceOfGaussiansLinear(torch.nn.Linear):
    # ASSUMES SQUARE MAPS
    def __init__(self, in_features, out_features, on, sigma_surround, sigma_center=1.0):
        super(DifferenceOfGaussiansLinear, self).__init__(in_features, out_features, bias=False)
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


class PiecewiseSigmoid(torch.nn.Module):
    def __init__(self, min_theta=0.0, max_theta=1.0):
        super(PiecewiseSigmoid, self).__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta

    def forward(self, input):
        return piecewise_sigmoid(min_theta=self.min_theta, max_theta=self.max_theta, input=input)


class AfferentNorm(torch.nn.Module):
    def __init__(self, strength, radius):
        super(AfferentNorm, self).__init__()
        self.radius = radius
        self.strength = strength

    def forward(self, afferent_input, activation):
        return afferent_normalize(strength=self.strength,
                                  afferent_input=afferent_input,
                                  activation=activation,
                                  radius=self.radius)


class CircularMask(torch.nn.Module):
    def __init__(self, radius):
        super(CircularMask, self).__init__()
        self.radius = radius

    def forward(self, input):
        # TODO: change to not in_place
        apply_circular_mask_to_weights(matrix=input, radius=self.radius)
