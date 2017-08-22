import numpy as np
import os
from paths import Paths

class DataProvider:
    def __init__(self, data_name):
        pass

    def next_batch(self, batch_size, phase):
        '''
        Get data and label of next batch.
        :param batch_size: batch size
        :param phase: train, val or test
        :return:
            data: input data
            label: whether is repeated buyer, 1 for true, 0 for false
        '''
        pass

