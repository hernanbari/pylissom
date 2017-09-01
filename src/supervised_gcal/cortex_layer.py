############
# V1/ Cortex
# Conexiones
# 0. Se conectan con cierto centro random a su LGN on/off optimo con cierto radio definido
# 1. El radio definido tiene q ser grande y q se superpongan
# 2. Pesos iniciales random normalizados
# 3. Conexiones laterales positivas excitatorias un poco menores a las aferentes
# 4. Conexiones laterales positivas inhibitorias mas grandes q la afferente
# Activacion
# 5. relu_con_threshold de
# 6. on_dot_product(input,pesos) + off_dot_product(input,pesos)
# 7. + excitatoria_dot_product(neurona,pesos)_t-1 - inhibitoria_dot_product(neurona,pesos)_t-1
# Learning
# 8. Hebbian rule con normalizacion (checkear pargina 77 que significa actividad presinaptica en el codigo)
#    CREO Q ES TIPO, INPUT VS OUTPUT, en gcal queda mas claro
# 9. connection death de lateral connections despues de cierto t_d debajo de un threshold

#########
# V1/Cortex
# Conexiones
# 0. Creo q no cambian
# Activaciones
# 1. Igual q lissom, agrega constantes
# 2. Se repite 16 veces por input, en la cual los pesos afferentes se mantienen pero el resto cambia por alguna razon
#   q no entendi (Creo q igual q lissom)
# Adaptation
# 3. Cuando terminan los settling steps, cada neurona se updatea el treshold haciendo un promedio exponencial smoothed
#   sobre sus patrones de activacion
# Learning
# 4. Misma q lissom

import numpy as np

import torch
from src.supervised_gcal.layer import Layer
from src.supervised_gcal.utils import get_zeros, get_uniform, mask_distance_gt_radius, normalize


def circular_mask(mat, radius):
    if radius is None:
        return mat
    dims = mat.shape[0]
    half_dims = int(np.sqrt(dims))
    tmp_shape = (half_dims, half_dims, half_dims, half_dims)

    # When the distance between the points of the two matrices is greater than radius, set to 0
    mask = np.fromfunction(function=lambda x, y, mu_x, mu_y: mask_distance_gt_radius(x, y, mu_x, mu_y, radius),
                           shape=tmp_shape, dtype=int)
    mask = np.reshape(mask, mat.shape)
    masked_mat = np.ma.masked_where(condition=mask, a=mat)
    return masked_mat.filled(0)


class LissomCortexLayer(Layer):
    def __init__(self, input_shape, self_shape, theta=0.0, afferent_radius=None, excitatory_radius=2,
                 inhibitory_radius=None):
        self.inhibitory_radius = inhibitory_radius
        self.excitatory_radius = excitatory_radius
        self.afferent_radius = afferent_radius
        self.theta = theta
        super().__init__(input_shape, self_shape)

    def _get_weight_variable(self, radius):
        # TODO: learn what Parameter means
        return torch.nn.Parameter(torch.Tensor(normalize(circular_mask(get_uniform(self.weights_shape),
                                                                       radius=radius))))

    def _setup_variables(self):
        self.on_weights = self._get_weight_variable(self.afferent_radius)

        self.off_weights = self._get_weight_variable(self.afferent_radius)

        self.inhibitory_weights = self._get_weight_variable(self.inhibitory_radius)

        self.excitatory_weights = self._get_weight_variable(self.excitatory_radius)

        self.retina_weights = self._get_weight_variable(self.afferent_radius)

        # Variable que guarda activaciones previas
        self.previous_activations = torch.Tensor(get_zeros(self.previous_activations_shape))

    def _afferent_activation(self, input, weights):
        return torch.clamp(torch.matmul(input, weights.data), min=self.theta, max=1)

    def _lateral_activation(self, previous_activations, weights):
        return torch.clamp(torch.matmul(previous_activations, weights.data), min=self.theta, max=1)

    def forward(self, input, simple_lissom=True):
        input = input.data.view((1, 784))
        if simple_lissom:
            retina = input
            self.retina = retina
            self.retina_activation = self._afferent_activation(retina, self.retina_weights)
            self.afferent_activation = self.retina_activation
        else:
            on, off = input
            on_activation = self._afferent_activation(on, self.on_weights)
            # off_activation = self._afferent_activation(off, self.off_weights)
            # self.afferent_activation = tf.add(on_activation, off_activation)
            # self.on = on
            # self.off = off
        self.excitatory_activation = self._lateral_activation(self.afferent_activation,
                                                              self.excitatory_weights)
        self.inhibitory_activation = self._lateral_activation(self.afferent_activation,
                                                              self.inhibitory_weights)

        new_activations = torch.clamp(
            self.afferent_activation + 0.2 * self.excitatory_activation - self.inhibitory_activation * 0.4,
            min=self.theta, max=1)

        self.previous_activations = new_activations

        return new_activations


def inference_cortex(input, lgn_shape, v1_shape, simple_lissom):
    v1_layer = LissomCortexLayer(lgn_shape, v1_shape)
    if simple_lissom:
        v1 = v1_layer.activation(input)
    else:
        on, off = input
        v1 = v1_layer.activation((on, off))
    return v1, v1_layer
