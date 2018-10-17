import pytest
import torch

from pylissom.nn.modules import Cortex
from pylissom.utils.stimuli import gaussian_generator


@pytest.fixture(params=[4, 9])
def in_features(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def connective_radius(request):
    return request.param


@pytest.fixture()
def cortex(in_features, connective_radius, sigma_x):
    return Cortex(in_features, in_features, connective_radius, sigma_x)


@pytest.fixture()
def x_center(in_features):
    return in_features // 2


@pytest.fixture()
def y_center(in_features):
    return in_features // 2


@pytest.fixture(params=[0.5, 1, 5])
def sigma_x(request):
    return request.param


@pytest.fixture(params=[0.5, 1, 5])
def sigma_y(request):
    return request.param


@pytest.fixture(params=[0, 45, 90, 135])
def gaussian_orientation(request):
    return request.param


@pytest.fixture()
def gaussian_numpy(in_features, x_center, y_center, sigma_x, sigma_y, gaussian_orientation):
    gauss_size = int(in_features ** 0.5)
    sigma_x = 3
    sigma_y = sigma_x
    return gaussian_generator(gauss_size, x_center, y_center, sigma_x, sigma_y, gaussian_orientation)


@pytest.fixture()
def gaussian_tensor(gaussian_numpy, in_features):
    return torch.from_numpy(gaussian_numpy).view(in_features)


@pytest.fixture()
def gaussian_variable(gaussian_tensor):
    return torch.autograd.Variable(gaussian_tensor).unsqueeze(0)