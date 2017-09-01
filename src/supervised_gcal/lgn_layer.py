###########
# Lissom
###########
# LGN On/Off
# Conexiones
# 0. Se conectan a su retina optima con cierto radio definido
# 1. todas las neuronas ocultas tienen el mismo peso respecto a la retina
# 2. los pesos iniciales de cada neurona respecto al input estan fijos y se definen por la diferencia de gaussianas
# 3. on y off son iguales solo que cambia el orden de la resta de gaussianas (centro vs surround)
# Activacion
# 4. relu_con_threshold(dot product(input, pesos)) por neurona
# Learning
# 5. Ninguno


##########
# GCAL
##########
# Se agrega "homeostatic adaptation" en V1 y "divisive inhibition - constrast-gain control" en LGN
##########
# LGN
# Conexiones
# 0. Los pesos de las laterales son fijas gaussianas
# Activacion
# 1. Antes de aplicar la relu, se divide la funcion de lissom por la suma de las conexiones laterales
#    Es un compomente de normalizacion
# Learning
# 2. Ninguno

import numpy as np
import tensorflow as tf

from tf_src.src import Layer


def gaussian(x, y, mu_x, mu_y, sigma):
    num = np.power(x - mu_x, 2) + np.power(y - mu_y, 2)
    den = 2 * np.power(sigma, 2)
    return np.float32(np.exp(-np.divide(num, den)))


class LissomLGNLayer(Layer):
    def __init__(self, input_shape, self_shape, sigma1, sigma2, name):
        super().__init__(input_shape, self_shape, name)
        self.sigma2 = sigma2
        self.sigma1 = sigma1
        self._setup()

    def _setup(self):
        with tf.name_scope(self.name):
            sigma1_weights_matrix = self._gaussian_weights(self.sigma1)
            sigma2_weights_matrix = self._gaussian_weights(self.sigma2)
            diff_of_gaussians = sigma1_weights_matrix - sigma2_weights_matrix
            self.weights = tf.constant(diff_of_gaussians, dtype=tf.float32, name='weights')
        return

    def _gaussian_weights(self, sigma):
        weights_shape_4d = tuple(self.input_shape.concatenate(self.self_shape).as_list())
        weights_matrix = np.fromfunction(function=lambda x, y, mu_x, mu_y: gaussian(x, y, mu_x, mu_y, sigma),
                                         shape=weights_shape_4d, dtype=np.float32)
        # TODO: Check this reshape, not tested, behaviour not thought trough
        reshaped_weights_matrix = np.reshape(weights_matrix, self.weights_shape)
        return reshaped_weights_matrix

    def _activation(self, images):
        act = tf.nn.relu(tf.matmul(images, self.weights), name='activation')
        return act


def inference_lgn(images, image_shape, lgn_shape, sigma_center, sigma_sourround, scope):
    on_layer = LissomLGNLayer(image_shape, lgn_shape, sigma1=sigma_center, sigma2=sigma_sourround, name=scope + 'on')
    off_layer = LissomLGNLayer(image_shape, lgn_shape, sigma2=sigma_center, sigma1=sigma_sourround, name=scope + 'off')
    on = on_layer.activation(images)
    off = off_layer.activation(images)
    return on, off
