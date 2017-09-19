import tensorflow as tf


def get_zeros(shape):
    return tf.zeros(shape=shape, dtype=tf.float32, name='zeros')


def get_ones(shape):
    return tf.ones(shape=shape, dtype=tf.float32, name='ones')