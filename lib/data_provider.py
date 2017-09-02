class DataProvider(object):
    def __init__(self):
        self._input_length = None
        self._val_size = None

    def next_batch(self, batch_size, phase):
        '''
        Get data and label of next batch.
        :param batch_size: batch size
        :param phase: train, val or test
        :return:
            data: input data
            label: whether is repeated buyer, 1 for true, 0 for false
        '''
        raise NotImplementedError

    @property
    def input_length(self):
        assert self._input_length is not None
        return self._input_length

    @property
    def val_size(self):
        assert self._val_size is not None
        return self._val_size

    @property
    def test_size(self):
        assert self._test_size is not None
        return self._test_size
