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
from build_model import build_model
from data_provider import DataProvider
from config_provider import get_config
from paths import Paths


def train_model(model_name, data_name, cfg_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    input_data = DataProvider(data_name)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Repeat Buyer Prediction Model')
    parser.add_argument('--model', dest='model_name',
                        help='model to use',
                        default='dnn', type=str)
    parser.add_argument('--data', dest='data_name',
                        help='data to use',
                        default='', type=str)
    parser.add_argument('--cfg', dest='cfg_name',
                        help='train, val and test config to use',
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
    train_model(args.model_name, args.data_name, args.cfg_name)
