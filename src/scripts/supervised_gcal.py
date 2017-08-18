import tensorflow as tf
from python.layers.core import dense


def inference(images):
    v2 = None
    # LGN On
    # LGN Off
    # V1
    # V2
    # Multi layer perceptron
    with tf.name_scope('multi_layer_perceptron'):
        hidden1 = tf.contrib.layers.fully_connected(inputs=v2, num_outputs=25, scope='hidden1')
        logits = tf.contrib.layers.fully_connected(hidden1, num_outputs=10, scope='softmax_linear')

    return logits

