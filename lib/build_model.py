from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.dnn import dnn


def build_model(model_name, inputs, is_training=True, dropout_keep_prob=0.5):
    if model_name == 'dnn':
        net = dnn(inputs)
    else:
        raise Exception('model {} is not existed'.format(model_name))
    return net
