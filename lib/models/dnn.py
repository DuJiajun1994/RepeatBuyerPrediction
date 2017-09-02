from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers


def dnn(inputs, is_training=True, dropout_keep_prob=0.8):
    regularizer = layers.l2_regularizer(scale=0.01)
    normalizer = layers.batch_norm
    with tf.variable_scope('dnn'):
        net = inputs
        for _ in range(3):
            net = layers.fully_connected(net,
                                         num_outputs=1024,
                                         activation_fn=tf.nn.tanh,
                                         normalizer_fn=normalizer,
                                         weights_regularizer=regularizer,
                                         biases_regularizer=regularizer)
            net = layers.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training)
        net = layers.fully_connected(net,
                                     num_outputs=2,
                                     activation_fn=None,
                                     normalizer_fn=normalizer,
                                     weights_regularizer=regularizer,
                                     biases_regularizer=regularizer)
    return net
