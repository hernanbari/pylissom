import numpy as np
import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils import custom_sigmoid, get_gaussian, circular_mask, normalize


class LissomLGNLayer(Layer):
    def __init__(self, input_shape, self_shape, sigma_center=1.0, sigma_sorround=3.0, min_theta=0.0, max_theta=1.0,
                 lgn_factor=1.0, on=False):
        self.lgn_factor = lgn_factor
        self.on = on
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.sigma_sorround = sigma_sorround
        self.sigma_center = sigma_center
        super().__init__(input_shape, self_shape)

    def _get_weight_variable(self, shape, sigma, radius=20):
        array = normalize(circular_mask(get_gaussian(shape, sigma), radius=radius))
        nnz_idx = np.nonzero(array)
        ans = torch.nn.Parameter(torch.sparse.FloatTensor(torch.from_numpy(np.asarray(nnz_idx)), torch.from_numpy(array[nnz_idx]), torch.Size(array.shape)))
        return ans

    def _setup_variables(self):
        sigma_center_weights_matrix = self._get_weight_variable(self.afferent_weights_shape, self.sigma_center)
        sigma_sorround_weights_matrix = self._get_weight_variable(self.afferent_weights_shape, self.sigma_sorround)
        if self.on:
            self.weights = torch.nn.Parameter((sigma_center_weights_matrix - sigma_sorround_weights_matrix).data)
        else:
            self.weights = torch.nn.Parameter((sigma_sorround_weights_matrix - sigma_center_weights_matrix).data)

    def forward(self, input):
        # TODO: pytorch implements sparse matmul only sparse x dense -> sparse and sparse x dense -> dense
        processed_input = self.process_input(input)
        import ipdb; ipdb.set_trace()
        torch.spmm()
        var = torch.matmul(processed_input, self.weights.data)
        var = custom_sigmoid(self.min_theta, self.max_theta,
                             self.lgn_factor * var)
        return var
