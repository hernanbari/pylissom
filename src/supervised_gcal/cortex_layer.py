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


def custom_sigmoid(input, tetha, name):
    with tf.name_scope(name):
        less_mask = tf.less(input, tf.constant(tetha, dtype=tf.float32, shape=input.shape, name='thetas'),
                            name='theta_mask')
        mask_update = tf.where(less_mask, get_zeros(shape=input.shape), input, name='mask_update')
        return tf.minimum(mask_update, get_ones(shape=mask_update.shape), name='output')


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
        return torch.autograd.Variable(torch.Tensor(normalize(circular_mask(get_uniform(self.weights_shape),
                                                               radius=radius))))

    def _setup_variables(self):
        self.on_weights = self._get_weight_variable(self.afferent_radius)

        self.off_weights = self._get_weight_variable(self.afferent_radius)

        self.inhibitory_weights = self._get_weight_variable(self.inhibitory_radius)

        self.excitatory_weights = self._get_weight_variable(self.excitatory_radius)

        self.retina_weights = self._get_weight_variable(self.afferent_radius)

        # Variable que guarda activaciones previas
        self.previous_activations = self._get_weight_variable(self.afferent_radius)

    def _afferent_activation(self, input, weights, name):
        return custom_sigmoid(torch.matmul(input, weights), self.theta, name=name)

    def _lateral_activation(self, previous_activations, weights, name):
        return custom_sigmoid(tf.matmul(previous_activations, weights, name=name + '/matmul'), self.theta, name=name)

    def _activation(self, input, simple_lissom=True):
        if simple_lissom:
            retina = input
            self.retina = retina
            self.retina_activation = self._afferent_activation(retina, self.retina_weights, name='retina_activation')
            self.afferent_activation = tf.identity(self.retina_activation, name='afferent_activation')
        else:
            on, off = input
            on_activation = self._afferent_activation(on, self.on_weights, name='on_activation')
            off_activation = self._afferent_activation(off, self.off_weights, name='off_activation')
            self.afferent_activation = tf.add(on_activation, off_activation, name='afferent_activation')
            self.on = on
            self.off = off

        with tf.control_dependencies([self.afferent_activation]):
            self.excitatory_activation = self._lateral_activation(self.afferent_activation,
                                                                  self.excitatory_weights,
                                                                  name='excitatory_activation')
            self.inhibitory_activation = self._lateral_activation(self.afferent_activation,
                                                                  self.inhibitory_weights,
                                                                  name='inhibitory_activation')

            new_activations = custom_sigmoid(
                tf.subtract(tf.add(self.afferent_activation, tf.multiply(self.excitatory_activation,
                                                                         tf.constant(0.2, dtype=tf.float32,
                                                                                     shape=self.excitatory_activation.shape)),
                                   name='sum_aff_exc'),
                            tf.multiply(self.inhibitory_activation,
                                        tf.constant(0.4, dtype=tf.float32, shape=self.inhibitory_activation.shape)),
                            name='sub_inhib'),
                self.theta,
                name='activation')

        with tf.control_dependencies([new_activations]):
            self.previous_activations_assign = tf.assign(self.previous_activations, new_activations,
                                                         name='assign_previous_activations')

            self.inhibitory_activation = tf.Print(self.inhibitory_activation, [self.inhibitory_activation], first_n=0,
                                                  summarize=self.inhibitory_activation.shape.num_elements(),
                                                  name='inhibitory_activation_print')

        output = tf.tuple([new_activations, self.previous_activations_assign], name='lissom_final_ops')
        # For weights update in training step
        self.activity = output[0]
        return output[0]


def inference_cortex(input, lgn_shape, v1_shape, scope, simple_lissom):
    v1_layer = LissomCortexLayer(lgn_shape, v1_shape, name=scope + 'v1')
    if simple_lissom:
        v1 = v1_layer.activation(input)
    else:
        on, off = input
        v1 = v1_layer.activation((on, off))
    return v1, v1_layer
