from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def logistic_regression(inputs):
    input_shape = inputs.get_shape().as_list()
    data_length = input_shape[1]
    with tf.variable_scope('logistic_regression'):
        w = tf.Variable(tf.zeros([data_length, 2]))
        b = tf.Variable(tf.zeros([2]))
        net = tf.matmul(inputs, w) + b
    return net
