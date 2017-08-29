from data_providers.data1 import Data1


def get_data_provider(data_name):
    data_provider = None
    if data_name == 'data1':
        data_provider = Data1()
    assert data_provider is not None, \
        'data provider {} is not existed'.format(data_name)
    return data_provider


class DataProvider(object):
    def __init__(self):
        self.input_length = None
        self.val_size = None

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


