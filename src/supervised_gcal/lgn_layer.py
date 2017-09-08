import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils import custom_sigmoid, get_gaussian, circular_mask, normalize


class LissomLGNLayer(Layer):
    def __init__(self, input_shape, self_shape, sigma_center=0.5, sigma_sorround=2.0, min_theta=0.0, max_theta=1.0, lgn_factor=1.0, on=False):
        self.lgn_factor = lgn_factor
        self.on = on
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.sigma_sorround = sigma_sorround
        self.sigma_center = sigma_center
        super().__init__(input_shape, self_shape)

    def _get_weight_variable(self, shape, sigma, radius=10):
        # TODO: learn what Parameter means
        # sigma = (radius / 5 if radius / 5 > 1 else 1) if radius is not None else 2
        return torch.nn.Parameter(torch.Tensor(normalize(circular_mask(get_gaussian(shape, sigma),
                                                                       radius=radius))))

    def _setup_variables(self):
        sigma_center_weights_matrix = self._get_weight_variable(self.afferent_weights_shape, self.sigma_center)
        sigma_sorround_weights_matrix = self._get_weight_variable(self.afferent_weights_shape, self.sigma_sorround)
        if self.on:
            self.weights = torch.nn.Parameter((sigma_center_weights_matrix - sigma_sorround_weights_matrix).data)
        else:
            self.weights = torch.nn.Parameter((sigma_sorround_weights_matrix - sigma_center_weights_matrix).data)

    def forward(self, input):
        processed_input = self.process_input(input)
        var = custom_sigmoid(self.min_theta, self.max_theta, self.lgn_factor * torch.matmul(processed_input, self.weights.data))
        return var
