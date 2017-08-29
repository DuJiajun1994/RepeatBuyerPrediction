from data_providers.data1 import Data1


def get_data_provider(data_name):
    data_provider = None
    if data_name == 'data1':
        data_provider = Data1()
    assert data_provider is not None, \
        'data provider {} is not existed'.format(data_name)
    return data_provider