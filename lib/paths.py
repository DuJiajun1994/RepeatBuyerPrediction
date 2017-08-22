import os
from easydict import EasyDict

Paths = EasyDict()
Paths.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Paths.data_path = os.path.join(Paths.root_path, 'data')
Paths.output_path = os.path.join(Paths.root_path, 'output')
Paths.cfg_path = os.path.join(Paths.root_path, 'cfgs')

