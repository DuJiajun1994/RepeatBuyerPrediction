import sys
import os
lib_path = os.getcwd()
print(lib_path)
sys.path.insert(0, lib_path)
print('System path:')
print(sys.path)

import tensorflow as tf
import argparse
import sys
from data_provider import DataProvider
from config_provider import get_config
from paths import Paths


def test_model(data_name, cfg_name, trained_model_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    input_data = DataProvider(data_name)

    meta_path = os.path.join(Paths.output_path, '{}.meta'.format(trained_model_name))
    assert os.path.exists(meta_path), \
        '{} is not existed'.format(meta_path)
    saver = tf.train.import_meta_graph(meta_path)

    with tf.Session() as sess:
        # Load trained model
        checkpoint_path = os.path.join(Paths.output_path, trained_model_name)
        print('checkpoint path: {}'.format(checkpoint_path))
        saver.restore(sess, checkpoint_path)

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
