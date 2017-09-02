from paths import add_lib_path
add_lib_path()

import tensorflow as tf
import argparse
import os
import sys
import numpy as np
from get_data_provider import get_data_provider
from config_provider import get_config
from paths import Paths


def test_model(data_name, cfg_name, trained_model_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    data_provider = get_data_provider(data_name)

    meta_path = os.path.join(Paths.output_path, '{}.meta'.format(trained_model_name))
    assert os.path.exists(meta_path), \
        '{} is not existed'.format(meta_path)
    saver = tf.train.import_meta_graph(meta_path)

    test_labels = np.ndarray(cfg.test_size, dtype=np.int)
    test_scores = np.ndarray(cfg.test_size, dtype=np.float)

    with tf.Session() as sess:
        # Load trained model
        checkpoint_path = os.path.join(Paths.output_path, trained_model_name)
        print('checkpoint path: {}'.format(checkpoint_path))
        saver.restore(sess, checkpoint_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        predicts = graph.get_tensor_by_name("predicts:0")

        iter_num = int((data_provider.test_size - 1) / cfg.batch_size + 1)
        for i in range(iter_num):
            batch_size = min(cfg.batch_size, data_provider.test_size - cfg.batch_size * i)
            batch_data, batch_labels = data_provider.next_batch(batch_size, 'test')
            batch_predicts = sess.run(predicts, feed_dict={x: batch_data,
                                                           y: batch_labels,
                                                           is_training: False,
                                                           dropout_keep_prob: cfg.dropout_keep_prob})
            for j in range(batch_size):
                test_labels[i*cfg.batch_size+j] = batch_labels[j]
                test_scores[i*cfg.batch_size+j] = batch_predicts[j][1]
    print('labels:')
    print(test_labels)
    print('scores:')
    print(test_scores)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test Repeat Buyer Prediction Model')
    parser.add_argument('--data', dest='data_name',
                        help='data to use',
                        default='', type=str)
    parser.add_argument('--cfg', dest='cfg_name',
                        help='train&test config to use',
                        default='', type=str)
    parser.add_argument('--model', dest='trained_model_name',
                        help='trained model to use',
                        default='', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    test_model(args.data_name, args.cfg_name, args.trained_model_name)
