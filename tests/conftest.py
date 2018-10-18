import copy

import pytest
import torch
import random

from pylissom.datasets import ThreeDotFaces
from torch.utils.data import DataLoader
from pylissom.optim import CortexHebbian, ReducedLissomHebbian, SequentialOptimizer
from pylissom.nn.modules import Cortex, PiecewiseSigmoid, ReducedLissom, LGN, Lissom
from pylissom.utils.stimuli import gaussian_generator
from pylissom.utils.training import Pipeline


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
    cortex = get_dummy_cortex()
    cortex2 = get_dummy_cortex()
    cortex3 = get_dummy_cortex()
    return ReducedLissom(afferent_module=cortex, excitatory_module=cortex2, inhibitory_module=cortex3,
                         settling_steps=settling_steps, min_theta=0.2, max_theta=1.1,
                         afferent_strength=afferent_strength, excitatory_strength=excitatory_strength,
                         inhibitory_strength=inhibitory_strength)


def get_dummy_cortex():
    radius = int(random.uniform(1, 5))
    sigma = random.uniform(0.5, 1.1)
    return Cortex(9, 9, radius, sigma)


def get_dummy_cortex_hebbian(cortex):
    learning_rate = random.uniform(0, 3.5)
    return CortexHebbian(cortex, learning_rate)


def get_dummy_lgn(on=False):
    radius = int(random.uniform(1, 5))
    sigma_center = random.uniform(0.5, 1.1)
    sigma_surround = random.uniform(1, 1.7)
    min_theta = random.uniform(0, 1.5)
    max_theta = random.uniform(0, 1.5)
    strength = random.uniform(0, 3.5)
    return LGN(9, 9, on, radius, sigma_surround, sigma_center, min_theta, max_theta, strength)


def get_dummy_rlissom():
    settling_steps = int(random.uniform(0, 3))
    afferent_strength = random.uniform(0, 3.5)
    excitatory_strength = random.uniform(0, 3.5)
    inhibitory_strength = random.uniform(0, 3.5)
    min_theta = random.uniform(0, 1.5)
    max_theta = random.uniform(0, 1.5)
    cortex = get_dummy_cortex()
    cortex2 = get_dummy_cortex()
    cortex3 = get_dummy_cortex()
    return ReducedLissom(afferent_module=cortex, excitatory_module=cortex2, inhibitory_module=cortex3,
                         settling_steps=settling_steps, min_theta=min_theta, max_theta=max_theta,
                         afferent_strength=afferent_strength, excitatory_strength=excitatory_strength,
                         inhibitory_strength=inhibitory_strength)


def get_dummy_rlissom_hebbian(rlissom):
    return SequentialOptimizer(get_dummy_cortex_hebbian(rlissom.afferent_module),
                               get_dummy_cortex_hebbian(rlissom.inhibitory_module),
                               get_dummy_cortex_hebbian(rlissom.excitatory_module))


def get_dummy_lissom():
    rlissom = get_dummy_rlissom()
    on = get_dummy_lgn(on=True)
    off = get_dummy_lgn(on=False)
    return Lissom(on, off, rlissom)


def get_dummy_lissom_hebbian(lissom):
    return get_dummy_rlissom_hebbian(lissom.v1)


def copy_module(m):
    return copy.deepcopy(m)


@pytest.fixture(params=[0.1, 0.2])
def learning_rate(request):
    return request.param
