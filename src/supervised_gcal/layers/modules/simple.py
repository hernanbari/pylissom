import torch
from src.supervised_gcal.utils.functions import piecewise_sigmoid
from src.supervised_gcal.utils.weights import get_gaussian_weights_variable


class GaussianLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, sigma=1.0):
        super(GaussianLinear, self).__init__(in_features, out_features)
        self.sigma = sigma
        self.weight = torch.nn.Parameter(get_gaussian_weights_variable(input_shape=in_features,
                                                                       output_shape=out_features,
                                                                       sigma=sigma))


class GaussianCloudLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, sigma=1.0):
        super(GaussianCloudLinear, self).__init__(in_features, out_features)
        self.sigma = sigma
        weight = get_gaussian_weights_variable(input_shape=in_features,
                                               output_shape=out_features,
                                               sigma=sigma)
        uniform_noise = torch.FloatTensor(weight.size()).uniform_(0, 1)
        self.weight = torch.nn.Parameter(weight * uniform_noise)


class DifferenceOfGaussiansLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, on, sigma_surround, sigma_center=1.0):
        super(DifferenceOfGaussiansLinear, self).__init__(in_features, out_features)
        self.on = on
        self.sigma_surround = sigma_surround
        self.sigma_center = sigma_center
        sigma_center_weights_matrix = get_gaussian_weights_variable(in_features, out_features,
                                                                    sigma_center)
        sigma_surround_weights_matrix = get_gaussian_weights_variable(in_features, out_features,
                                                                      sigma_surround)
        if on:
            diff = (sigma_center_weights_matrix.weight.data - sigma_surround_weights_matrix.weight.data).t()
        else:
            diff = (sigma_surround_weights_matrix.weight.data - sigma_center_weights_matrix.weight.data).t()
        self.weight = torch.nn.Parameter(diff)


class PiecewiseSigmoid(torch.nn.Module):
    def __init__(self, min_theta=0.0, max_theta=1.0):
        super(PiecewiseSigmoid, self).__init__()
        self.max_theta = max_theta
        self.min_theta = min_theta

    def forward(self, input):
        return piecewise_sigmoid(min_theta=self.min_theta, max_theta=self.max_theta, input=input)