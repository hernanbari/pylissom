import tensorflow as tf

from src.supervised_gcal.cortex_layer import LissomCortexLayer


def hebbian_learning(weights, input, output, learning_rate=0.1):
    # Weight adaptation of a single neuron
    # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
    with tf.name_scope('hebbian_rule'):
        hebbian = tf.add(weights, learning_rate * tf.multiply(input, output))
    col_sum = tf.reduce_sum(hebbian, axis=1, name='col_sum')
    normalization = tf.divide(hebbian, col_sum, name='normalization')
    update_op = tf.assign(weights, normalization, name='update_weights')
    return update_op


class LissomHebbianOptimizer(object):
    def update_weights(self, lissom_layer):
        assert isinstance(lissom_layer, LissomCortexLayer)
        with tf.name_scope(self.name):
            update_on = hebbian_learning(lissom_layer.on_weights, lissom_layer.on, lissom_layer.activity,
                                         self.learning_rate)

            update_off = hebbian_learning(lissom_layer.off_weights, lissom_layer.off, lissom_layer.activity,
                                          self.learning_rate)

            update_excitatory = hebbian_learning(lissom_layer.excitatory_weights, lissom_layer.previous_activations,
                                                 lissom_layer.activity, self.learning_rate)

            update_inhibitory = hebbian_learning(lissom_layer.inhibitory_weights, lissom_layer.previous_activations,
                                                 lissom_layer.activity, self.learning_rate)
            return tf.tuple([update_on, update_off, update_excitatory, update_inhibitory], name='lissom_learning')

    def __init__(self, learning_rate, name):
        self.learning_rate = learning_rate
        self.name = name
