from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.dnn import dnn
from models.logistic_regression import logistic_regression


def build_model(model_name, inputs, is_training, dropout_keep_prob):
    model = None
    if model_name == 'dnn':
        model = dnn(inputs, is_training, dropout_keep_prob)
    elif model_name == 'logistic_regression':
        model = logistic_regression(inputs)
    assert model is not None, \
        'data provider {} is not existed'.format(model_name)
    return model
