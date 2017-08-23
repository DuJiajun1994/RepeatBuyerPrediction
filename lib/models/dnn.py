from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib


def dnn(inputs, is_training=True, dropout_keep_prob=0.8):
    regularizer = layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('dnn'):
        net = inputs
        for _ in range(3):
            net = layers_lib.fully_connected(net,
                                             num_outputs=1024,
                                             activation_fn=tf.nn.tanh,
                                             weights_regularizer=regularizer,
                                             biases_regularizer=regularizer)
            net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training)
        net = layers_lib.fully_connected(net,
                                         num_outputs=2,
                                         activation_fn=None,
                                         weights_regularizer=regularizer,
                                         biases_regularizer=regularizer)
    return net
