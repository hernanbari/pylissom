import copy

import pytest
import torch

from pylissom.nn.modules import Cortex, PiecewiseSigmoid, ReducedLissom
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


@pytest.fixture(params=[0.5, 1])
def sigma_y(request):
    return request.param


@pytest.fixture(params=[0, 90, 135])
def gaussian_orientation(request):
    return request.param


@pytest.fixture()
def gaussian_numpy(in_features, x_center, y_center, sigma_x, sigma_y, gaussian_orientation):
    gauss_size = int(in_features ** 0.5)
    return gaussian_generator(gauss_size, x_center, y_center, sigma_x, sigma_y, gaussian_orientation)


@pytest.fixture()
def gaussian_tensor(gaussian_numpy, in_features):
    return torch.from_numpy(gaussian_numpy).view(in_features)


@pytest.fixture()
def gaussian_variable(gaussian_tensor):
    return torch.autograd.Variable(gaussian_tensor).unsqueeze(0)


# @pytest.fixture()
# def piecewise_sigmoid(min_theta, max_theta):
#     return PiecewiseSigmoid(min_theta=min_theta, max_theta=max_theta)


@pytest.fixture(params=[0, 1, 3])
def settling_steps(request):
    return request.param


@pytest.fixture(params=[0, 1, 1.5])
def min_theta(request):
    return request.param


@pytest.fixture(params=[0, 1])
def max_theta(request):
    return request.param


@pytest.fixture(params=[0, 1.5])
def afferent_strength(request):
    return request.param


@pytest.fixture(params=[0, 3.5])
def excitatory_strength(request):
    return request.param


@pytest.fixture(params=[0, 0.5])
def inhibitory_strength(request):
    return request.param


@pytest.fixture()
def rlissom(settling_steps, afferent_strength, excitatory_strength, inhibitory_strength):
    cortex = Cortex(9, 9, 5, 0.9)
    cortex2 = Cortex(9, 9, 3, 1.1)
    cortex3 = Cortex(9, 9, 2, 0.5)
    return ReducedLissom(afferent_module=cortex, excitatory_module=cortex2, inhibitory_module=cortex3,
                         settling_steps=settling_steps, min_theta=0.2, max_theta=1.1,
                         afferent_strength=afferent_strength, excitatory_strength=excitatory_strength,
                         inhibitory_strength=inhibitory_strength)


def copy_module(m):
    return copy.deepcopy(m)
