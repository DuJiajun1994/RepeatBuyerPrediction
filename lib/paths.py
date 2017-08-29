import os
import sys
from easydict import EasyDict


Paths = EasyDict()
Paths.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Paths.data_path = os.path.join(Paths.root_path, 'data')
Paths.output_path = os.path.join(Paths.root_path, 'output')
Paths.cfg_path = os.path.join(Paths.root_path, 'cfgs')
Paths.lib_path = os.path.join(Paths.root_path, 'lib')


def add_lib_path():
    '''
    Add lib path to PYTHONPATH
    :return:
    '''
    sys.path.insert(0, Paths.lib_path)
    print('add lib path: {}'.format(Paths.lib_path))
