from collections import Counter
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import rotate
from tqdm import tqdm, tqdm_notebook

from src.utils.stimuli import translate, generate_horizontal_bar
from src.utils.training.pipeline import Pipeline


class OrientationMap(object):
    # TODO: optimize using vectorization to calculate activations
    def __init__(self, model, inputs, use_tqdm_notebook=True):
        self.use_tqdm_notebook = use_tqdm_notebook
        self.model = model
        self.inputs = inputs
        self.tqdm = tqdm_notebook if self.use_tqdm_notebook else tqdm

    def maximum_activations(self, model, inputs):
        activations = []
        for inp in self.tqdm(inputs):
            inp = Pipeline.process_input(inp)
            act = model(inp)
            activations.append(act)
        maximums, _ = torch.max(torch.stack(activations), 0)
        return maximums

    def calculate_keys_activations(self, model, inputs):
        return {k: self.maximum_activations(model, array) for k, array in inputs.items()}

    @lru_cache()
    def get_orientation_map(self):
        activations = self.calculate_keys_activations(self.model, self.inputs)
        mat = torch.stack(list(activations.values()))
        _, preferences = torch.max(mat, 0)
        keys = list(activations.keys())
        orientation_map = [keys[idx.data[0]] for idx in preferences.squeeze()]
        # Assumes Square Maps
        rows = int(np.sqrt(self.model.out_features))
        return np.reshape(np.asarray(orientation_map), (rows, rows))

    @staticmethod
    def orientation_hist(orientation_map):
        orientation_hist = Counter(orientation_map.flatten().tolist())
        return orientation_hist

    @lru_cache()
    def get_orientation_hist(self):
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
