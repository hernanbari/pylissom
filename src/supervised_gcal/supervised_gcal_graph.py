# Ideas implementativas:
# 0. Usar tf.contrib.layers.fully_connected / tf.layers.dense para las LGN, pasandole funciones custom de activaciones
#   e inicializacion, con trainable false para las LGN
# 1. Creo q no van a servir para V1, solo para LGN
# 2. Si no, implementar a mano to_do, tal vez sea mas facil, pq la parte de las conexiones laterales no va a funcar creo
# 3. Buscar funcion gaussiana ya implementada q diga cual es el valor de y dado gaussiana con centro x, checkear topo.
#   Esto sirve para pesos de LGN
# 4. Encontrar funcion q te diga como se conenectan las layers entre si, checkear topo
# Detalles:
# - Hay constantes de normalizacion por todos lados del libro/paper q no los anote
# - Entender mejor settling steps
# - Ver el codigo de topographica a ver como implementan el homeostatic adaptation y gain control
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

from src.supervised_gcal.cortex_layer import LissomCortexLayer
from src.supervised_gcal.hebbian_optimizer import HebbianOptimizer
from src.supervised_gcal.lgn_layer import LissomLGNLayer


def inference_lgn(images, image_shape, lgn_shape, sigma_center, sigma_sourround, scope):
    on_layer = LissomLGNLayer(image_shape, lgn_shape, sigma1=sigma_center, sigma2=sigma_sourround, name=scope+'on')
    off_layer = LissomLGNLayer(image_shape, lgn_shape, sigma2=sigma_center, sigma1=sigma_sourround, name=scope+'off')
    on = on_layer.activation(images)
    off = off_layer.activation(images)
    return on, off


def inference_cortex(on, off, lgn_shape, v1_shape, scope):
    v1_layer = LissomCortexLayer(lgn_shape, v1_shape, name=scope+'v1')
    v1 = v1_layer.activation((on, off))
    return v1


def inference_classification(v1):
    # Multi layer perceptron
    # TODO: learn why tf.name_scope doesn't work as expected
    # with tf.name_scope('multi_layer_perceptron/') as scope:
    scope = 'classification/multi_layer_perceptron/'
    hidden1 = tf.contrib.layers.fully_connected(inputs=v1, num_outputs=25, scope=scope+'hidden1/')
    logits = tf.contrib.layers.fully_connected(hidden1, num_outputs=10, activation_fn=tf.identity, scope=scope+'logits/')

    return logits


def inference_lissom(images, image_shape):
    scope = 'lissom/'
    lgn_shape = image_shape
    on, off = inference_lgn(images, image_shape, lgn_shape, 1, 1, scope)
    v1_shape = image_shape
    v1 = inference_cortex(on, off, lgn_shape, v1_shape, scope)
    return v1


def inference(images, image_shape):
    # TODO: Reduce lgn_shape, it's too big and doesn't fit on memory, implement connection field radius
    v1 = inference_lissom(images, image_shape)
    # logits = inference_classification(v1)
    # Maybe a tf.tuple??
    return v1, v1  # , logits


def training_cortex(v1):
    # return None
    with tf.name_scope('lissom/') as scope:
        import ipdb; ipdb.set_trace()
        optimizer = HebbianOptimizer()
        train_op = optimizer.minimize(v1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))
    return train_op


def training_classification(loss, learning_rate):
    # Create the gradient descent optimizer with the given learning rate.
    with tf.name_scope('classification/') as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))
    return train_op


def training(v1, loss, learning_rate):
    train_op_v1 = training_cortex(v1)
    # train_op_classification = training_classification(loss, learning_rate)
    # Maybe a tf.tuple??
    return train_op_v1, v1 #, train_op_classification


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
    with tf.name_scope('classification/evaluation/') as scope:
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='loss')


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
