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
import tensorflow as tf

from tf_src.src import Layer
from tf_src.src import get_zeros, get_ones


def normalize(input, name='normalize'):
    return tf.divide(input, tf.norm(input, ord=1, axis=0), name=name)


def get_uniform(shape):
    random = tf.random_uniform(shape, minval=0.0, maxval=0.5, dtype=tf.float32, name='random_uniform')
    return random


def custom_sigmoid(input, tetha, name):
    with tf.name_scope(name):
        less_mask = tf.less(input, tf.constant(tetha, dtype=tf.float32, shape=input.shape, name='thetas'),
                            name='theta_mask')
        mask_update = tf.where(less_mask, get_zeros(shape=input.shape), input, name='mask_update')
        return tf.minimum(mask_update, get_ones(shape=mask_update.shape), name='output')


def util_fun(x, y, mu_x, mu_y, radius):
    return np.sqrt(((x - mu_x) ** 2 + (y - mu_y) ** 2)) > radius


def circular_mask(mat, radius, name='circular_mask'):
    if radius is None:
        return tf.identity(mat, name=name)
    with tf.name_scope(name):
        dims = mat.shape[0]
        half_dims = int(np.sqrt(dims.value))
        tmp_shape = (half_dims, half_dims, half_dims, half_dims)
        # When the distance between the points of the two matrices is greater than radius, set to 0
        mask = np.fromfunction(function=lambda x, y, mu_x, mu_y: util_fun(x, y, mu_x, mu_y, radius),
                               shape=tmp_shape, dtype=int)
        mask_reshaped = tf.reshape(mask, mat.shape, name='reshape')
        mask_update = tf.where(mask_reshaped, get_zeros(shape=mat.shape), mat, name=name)
        return mask_update


class LissomCortexLayer(Layer):
    def __init__(self, input_shape, self_shape, name, theta=0.0, afferent_radius=None, excitatory_radius=2,
                 inhibitory_radius=None):
        self.inhibitory_radius = inhibitory_radius
        self.excitatory_radius = excitatory_radius
        self.afferent_radius = afferent_radius
        self.theta = theta
        super().__init__(input_shape, self_shape, name)
        self._setup()

    def _setup(self):
        with tf.name_scope(self.name):
            self.on_weights = tf.Variable(normalize(circular_mask(get_uniform(self.weights_shape),
                                                                  radius=self.afferent_radius)), name='on_weights')

            self.off_weights = tf.Variable(normalize(circular_mask(get_uniform(self.weights_shape),
                                                                   radius=self.afferent_radius)), name='off_weights')

            self.inhibitory_weights = tf.Variable(normalize(circular_mask(get_uniform(self.weights_shape),
                                                                          radius=self.inhibitory_radius)),
                                                  name='inhibitory_weights')

            self.excitatory_weights = tf.Variable(normalize(circular_mask(get_uniform(self.weights_shape),
                                                                          radius=self.excitatory_radius)),
                                                  name='excitatory_weights')

            self.retina_weights = tf.Variable(normalize(circular_mask(get_uniform(self.weights_shape),
                                                                      radius=self.afferent_radius)),
                                              name='retina_weights')

            # Variable que guarda activaciones previas
            self.previous_activations = tf.Variable(
                tf.zeros(self.previous_activations_shape, dtype=tf.float32), trainable=False,
                name='previous_activations')

    def _afferent_activation(self, input, weights, name):
        return custom_sigmoid(tf.matmul(input, weights, name=name + '/matmul'), self.theta, name=name)

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
