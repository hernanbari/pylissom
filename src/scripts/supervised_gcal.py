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


def inference_lgn(image, lgn_shape, sigma_center, sigma_sourround):
    on_layer = LissomLGNLayer(image.shape, lgn_shape, sigma1=sigma_center, sigma2=sigma_sourround, name='on')
    off_layer = LissomLGNLayer(image.shape, lgn_shape, sigma2=sigma_center, sigma1=sigma_sourround, name='off')
    on = on_layer.activation(image)
    off = off_layer.activation(image)
    return on, off


def inference_cortex(on, off, v1_shape):
    v1_layer = LissomCortexLayer(on.shape, v1_shape, name='v1')
    v1 = v1_layer.activation((on, off))
    return v1


def inference(image):
    on, off = inference_lgn(image, image.shape, 1, 1)
    v1 = inference_cortex(on, off, image.shape)
    # Multi layer perceptron
    with tf.name_scope('multi_layer_perceptron'):
        hidden1 = tf.contrib.layers.fully_connected(inputs=v1, num_outputs=25,
                                                    scope='hidden1')
        logits = tf.contrib.layers.fully_connected(hidden1, num_outputs=10, activation_fn=tf.identity,
                                                   scope='softmax_linear')
    return logits


def training(loss):
    optimizer = HebbianOptimizer()
    train_op = optimizer.minimize(loss)
    return train_op


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


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
