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

import tensorflow as tf

from src.supervised_gcal.layer import Layer


class LissomCortexLayer(Layer):
    def __init__(self, input_shape, self_shape, name):
        super().__init__(input_shape, self_shape, name)
        self._setup()

    def _setup(self):
        with tf.name_scope(self.name):
            self.on_weights = tf.Variable(
                tf.truncated_normal(self.weights_shape, dtype=tf.float32, mean=0.5), name='on_weights')
            self.off_weights = tf.Variable(
                tf.truncated_normal(self.weights_shape, dtype=tf.float32, mean=0.5), name='off_weights')
            self.inhibitory_weights = tf.Variable(
                tf.truncated_normal(self.weights_shape, dtype=tf.float32, mean=0.5), name='inhibitory_weights')
            self.excitatory_weights = tf.Variable(
                tf.truncated_normal(self.weights_shape, dtype=tf.float32, mean=0.5), name='excitatory_weights')

            self.retina_weights = tf.Variable(
                tf.truncated_normal(self.weights_shape, dtype=tf.float32, mean=0.5), name='retina_weights')

            # Variable que guarda activaciones previas
            self.previous_activations = tf.Variable(
                tf.zeros(self.previous_activations_shape, dtype=tf.float32), trainable=False,
                name='previous_activations')

    @staticmethod
    def _afferent_activation(input, weights, name):
        return tf.matmul(input, weights, name=name)

    @staticmethod
    def _lateral_activation(previous_activations, weights, name):
        return tf.matmul(previous_activations, weights, name=name)

    def _activation(self, input, simple_lissom=True):
        if simple_lissom:
            retina = input
            retina_activation = self._afferent_activation(retina, self.retina_weights, name='retina_activation')
            # afferent_activation = tf.nn.relu(retina_activation, name='activation')
            afferent_activation = tf.minimum(retina_activation, tf.ones(retina_activation.shape),
                                             name='afferent_activation')
            self.retina = retina
        else:
            on, off = input
            on_activation = self._afferent_activation(on, self.on_weights, name='on_activation')
            off_activation = self._afferent_activation(off, self.off_weights, name='off_activation')
            afferent_activation = tf.add(on_activation, off_activation, name='afferent_activation')
            self.on = on
            self.off = off

        excitatory_activation = self._lateral_activation(self.previous_activations, self.excitatory_weights,
                                                         name='excitatory_activation')
        inhibitory_activation = self._lateral_activation(self.previous_activations, self.inhibitory_weights,
                                                         name='inhibitory_activation')
        new_activations = tf.nn.relu(afferent_activation + excitatory_activation - inhibitory_activation,
                                     name='activation')
        with tf.control_dependencies([new_activations]):
            self.previous_activations_assign = self.previous_activations.assign(new_activations)
        output = tf.tuple([new_activations, self.previous_activations_assign])
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
