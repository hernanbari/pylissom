from skimage.transform import rotate
from tqdm import tqdm

import torch
import numpy as np

from src.utils.images import plot_matrix, generate_horizontal_bar, translate
from src.utils.pipeline import Pipeline


def generate_all_inputs(size, orientations=180):
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
    return inputs


def maximum_activations(model, inputs):
    activations = []
    var = False
    for inp in tqdm(inputs):
        inp = torch.autograd.Variable(inp)
        inp = Pipeline.process_input(inp)
        inp = inp.cuda()
        act = model(inp)
        if not var:
            try:
                shape = model.self_shape
            except Exception:
                shape = model.v1.self_shape
            plot_matrix(np.reshape(act.data.cpu().numpy(), shape))
            var = True
        activations.append(act.data)
    maximums, _ = torch.max(torch.stack(activations), 0)
    return maximums


def calculate_keys_activations(model, inputs):
    activations = {}
    for k, images in inputs.items():
        activations[k] = maximum_activations(model, images)
    return activations


def get_orientation_map(activations, shape):
    mat = torch.stack(list(activations.values()))
    values, preferences = torch.max(mat, 0)
    keys = list(activations.keys())
    orientation_map = []
    for idx in preferences[0]:
        orientation_map.append(keys[idx])
    return np.reshape(np.asarray(orientation_map), shape)
