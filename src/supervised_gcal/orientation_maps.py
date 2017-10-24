from collections import Counter
from functools import lru_cache
import matplotlib.pyplot as plt

from skimage.transform import rotate
from tqdm import tqdm

import torch
import numpy as np

from src.utils.images import generate_horizontal_bar, translate
from src.utils.pipeline import Pipeline


class OrientationMap(object):
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    @staticmethod
    def maximum_activations(model, inputs):
        activations = []
        for inp in tqdm(inputs):
            inp = Pipeline.process_input(inp)
            inp = inp.cuda() if model.is_cuda else inp
            act = model(inp)
            activations.append(act)
        maximums, _ = torch.max(torch.stack(activations), 0)
        return maximums

    @staticmethod
    def calculate_keys_activations(model, inputs):
        return {k: OrientationMap.maximum_activations(model, array) for k, array in inputs.items()}

    @lru_cache()
    def get_orientation_map(self):
        activations = self.calculate_keys_activations(self.model, self.inputs)
        mat = torch.stack(list(activations.values()))
        values, preferences = torch.max(mat, 0).squeeze()
        keys = list(activations.keys())
        orientation_map = [keys[idx] for idx in preferences]
        return np.reshape(np.asarray(orientation_map), self.model.self_shape)

    @staticmethod
    def orientation_hist(orientation_map):
        orientation_hist = Counter(orientation_map.flatten().tolist())
        return orientation_hist

    @lru_cache()
    def get_orienatation_hist(self):
        return self.orientation_hist(self.get_orientation_map())


def plot_orientation_map(orientation_map):
    return plt.imshow(orientation_map, cmap='gist_rainbow')


def plot_orientation_hist(orientation_hist):
    values = [float(v) for v in orientation_hist.values()]
    labels = [str(k) + '°' for k in orientation_hist.keys()]
    plot = plt.pie(values, labels=labels,
                   autopct='%.2f')
    return plot


def metrics_orientation_hist(orientation_hist):
    values = [float(v) for v in orientation_hist.values()]
    normalized = values / np.linalg.norm(values, ord=1)
    mean = np.mean(normalized)
    std = np.std(normalized)
    return mean, std


def get_oriented_lines(size, orientations=180):
    vertical_bar = generate_horizontal_bar(size)
    move_vertical = translate(vertical_bar, over_x=False)
    inputs = {}
    for degree in np.linspace(0, 180, num=orientations):
        rotated = []
        for im in move_vertical:
            # mode : {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}
            rot = rotate(im, degree, mode='reflect')
            rotated.append(rot.astype(im.dtype))
        inputs[int(degree)] = rotated
    return numpy_dict_to_tensors(inputs)


def numpy_dict_to_tensors(d):
    return {k: map(torch.from_numpy, v) for k, v in d.items()}
