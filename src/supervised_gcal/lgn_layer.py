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

import tensorflow as tf

from src.gcal_model.layer import Layer


def gaussian(x, y, mu_x, mu_y, sigma):
    raise NotImplementedError


class LissomLGNLayer(Layer):
    def __init__(self, input_shape, self_shape, sigma1, sigma2, name):
        super(LissomLGNLayer).__init__(input_shape, self_shape, name)
        self.sigma2 = sigma2
        self.sigma1 = sigma1
        self._setup()

    def _setup(self):
        sigma1_weights_matrix = self._gaussian_weights(self.sigma1)
        sigma2_weights_matrix = self._gaussian_weights(self.sigma2)
        diff_of_gaussians = sigma1_weights_matrix - sigma2_weights_matrix
        self.weights = tf.constant(diff_of_gaussians, name='weights')
        return

    def _gaussian_weights(self, sigma):
        return tf.np.fromfunction(function=lambda x, y, mu_x, mu_y: gaussian(x, y, mu_x, mu_y, sigma),
                                  shape=self.input_shape + self.self_shape)

    def _activation(self, image):
        return tf.nn.relu(tf.matmul(image, self.weights), name='activation')
