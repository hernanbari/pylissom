import tensorflow as tf


class Layer(object):
    def __init__(self, input_shape, self_shape, name):
        self.self_shape = self_shape
        self.input_shape = input_shape
        self.name = name
        self.weights_shape = [self.input_shape.num_elements(), self.self_shape.num_elements()]

    def setup(self):
        with tf.name_scope(self.name):
            self._setup()

    def activation(self, input):
        with tf.name_scope(self.name):
            return self._activation(input)

    def _setup(self):
        raise NotImplementedError

    def _activation(self, input):
        raise NotImplementedError


