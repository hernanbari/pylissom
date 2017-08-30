import tensorflow as tf

from src.supervised_gcal.cortex_layer import LissomCortexLayer
from src.supervised_gcal.utils import get_zeros


def hebbian_learning(weights, input, output, learning_rate):
    # Weight adaptation of a single neuron
    # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
    with tf.name_scope(weights.name[:-2]):
        zero_mask = tf.equal(weights, get_zeros(shape=weights.shape), name='zeros_mask')

        with tf.name_scope('hebbian_rule'):
            delta = tf.multiply(learning_rate, tf.matmul(tf.transpose(input), output, name='matmul'),
                                name='delta')
            hebbian = tf.add(weights, delta, 'sum_delta')
        zero_update = tf.where(zero_mask, get_zeros(shape=hebbian.shape), hebbian, name='weights_with_old_zeros')
        normalization = tf.divide(zero_update, tf.norm(zero_update, axis=0, name='hebbian_norm'), name='normalization')
        update_op = tf.assign(weights, normalization, name='update_weights')
        return update_op


class LissomHebbianOptimizer(object):
    def update_weights(self, lissom_layer, simple_lissom):
        assert isinstance(lissom_layer, LissomCortexLayer)
        with tf.name_scope(self.name):
            params = []
            if simple_lissom:
                update_retina = hebbian_learning(lissom_layer.retina_weights, lissom_layer.retina,
                                                 lissom_layer.activity,
                                                 self.learning_rate)
                params.append(update_retina)
            else:
                update_on = hebbian_learning(lissom_layer.on_weights, lissom_layer.on, lissom_layer.activity,
                                             self.learning_rate)

                update_off = hebbian_learning(lissom_layer.off_weights, lissom_layer.off, lissom_layer.activity,
                                              self.learning_rate)
                params.append(update_on)
                params.append(update_off)

            update_excitatory = hebbian_learning(lissom_layer.excitatory_weights, lissom_layer.excitatory_activation,
                                                 lissom_layer.activity, self.learning_rate)

            update_inhibitory = hebbian_learning(lissom_layer.inhibitory_weights, lissom_layer.inhibitory_activation,
                                                 lissom_layer.activity, self.learning_rate)
            params.append(update_inhibitory)
            params.append(update_excitatory)

            return tf.tuple(params, name='lissom_learning')

    def __init__(self, learning_rate, name):
        self.learning_rate = learning_rate
        self.name = name
