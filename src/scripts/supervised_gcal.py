import tensorflow as tf
from python.layers.core import dense

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

def on_weights():
    weights = 1
    return weights


def off_weights():
    weights = 1
    return weights


def inference_lgn(image, weights):
    lgn = 1
    return lgn


def inference_cortex(on, off):
    v1 = 1
    return v1


def inference(image):
    on = inference_lgn(image, on_weights())
    off = inference_lgn(image, off_weights())
    v1 = inference_cortex(on, off)
    # Multi layer perceptron
    with tf.name_scope('multi_layer_perceptron'):
        hidden1 = tf.contrib.layers.fully_connected(inputs=v1, num_outputs=25,
                                                    scope='hidden1')
        logits = tf.contrib.layers.fully_connected(hidden1, num_outputs=10, activation_fn=tf.identity,
                                                   scope='softmax_linear')

    return logits
