from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib


def logistic_regression(inputs):
    regularizer = layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('logistic_regression'):
        net = layers_lib.fully_connected(inputs,
                                         num_outputs=2,
                                         activation_fn=None,
                                         weights_regularizer=regularizer,
                                         biases_regularizer=regularizer)
    return net
