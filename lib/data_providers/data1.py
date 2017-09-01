import numpy as np
import os
import random
import pandas as pd
from paths import Paths
from data_provider import DataProvider


class Data1(DataProvider):
    def __init__(self):
        DataProvider.__init__(self)
        self._train_index = 0
        self._val_index = 0
        self._test_index = 0

        train_pair_path = os.path.join(Paths.data_path, 'train_pair.csv')
        self._train_df = pd.read_csv(train_pair_path,
                                     dtype={
                                         'user_id': np.int,
                                         'merchant_id': np.int,
                                         'label': np.float,
                                         'visit_cat': np.str,
                                         'visit_seller': np.str,
                                         'visit_brand': np.str,
                                         'buy_cat': np.str,
                                         'buy_seller': np.str,
                                         'buy_brand': np.str
                                     })
        train_val_list = list(self._train_df.index)
        random.shuffle(train_val_list)
        self._train_size = int(len(train_val_list) * 0.7)
        self._val_size = len(train_val_list) - self._train_size
        self._train_list = train_val_list[:self._train_size]
        self._val_list = train_val_list[self._train_size:]

        user_data_path = os.path.join(Paths.data_path, 'user_date_ver3.txt')
        merchant_data_path = os.path.join(Paths.data_path, 'mer_code_ver2.txt')
        self._user_data = np.loadtxt(user_data_path, dtype=np.int)
        self._merchant_data = np.loadtxt(merchant_data_path, dtype=np.int)
        self._input_length = 40466

    @staticmethod
    def _parse_feature(id_str, vector_length):
        feature = np.zeros(vector_length, dtype=np.int)
        if pd.isnull(id_str):
            return feature
        id_arr = id_str.split(',')
        for id_cnt in id_arr:
            id_cnt_arr = id_cnt.split('*')
            feature_id = int(id_cnt_arr[0])
            cnt = int(id_cnt_arr[1])
            assert 0 < feature_id <= vector_length
            feature[feature_id-1] += cnt
        return feature

    def _get_pair_data(self, df, batch_ids):
        feature_names = ['visit_cat', 'visit_seller', 'visit_brand', 'buy_cat', 'buy_seller', 'buy_brand']
        feature_length = {
            'visit_cat': 1671,
            'visit_seller': 4995,
            'visit_brand': 8477,
            'buy_cat': 1671,
            'buy_seller': 4995,
            'buy_brand': 8477
        }
        pair_data = np.vstack([
            np.hstack([
                self._parse_feature(df[feature_name][index_id], feature_length[feature_name])
                for feature_name in feature_names
            ]) for index_id in batch_ids
        ])
        return pair_data

    def _get_batch_data(self, df, batch_ids, phase):
        batch_pair_data = self._get_pair_data(df, batch_ids)
        batch_user_data = np.vstack([self._user_data[df['user_id'][i] - 1] for i in batch_ids])
        batch_merchant_data = np.vstack([self._merchant_data[df['merchant_id'][i] - 1] for i in batch_ids])
        batch_data = np.hstack([batch_pair_data, batch_user_data, batch_merchant_data])

        if phase != 'test':
            batch_label = np.hstack([
                int(df['label'][index_id]) for index_id in batch_ids
            ])
        else:
            batch_label = None
        return batch_data, batch_label

    @staticmethod
    def _get_batch_ids(id_list, id_index, batch_size):
        if id_index + batch_size <= len(id_list):
            batch_ids = id_list[id_index:(id_index+batch_size)]
        else:
            batch_ids = id_list[id_index:] + id_list[:(batch_size + id_index - len(id_list))]
        next_id_index = (id_index+batch_size) % len(id_list)
        return batch_ids, next_id_index

    def next_batch(self, batch_size, phase):
        assert phase in ('train', 'val', 'test')
        batch_data = None
        batch_label = None
        if phase == 'train':
            batch_ids, self._train_index = self._get_batch_ids(self._train_list, self._train_index, batch_size)
            batch_data, batch_label = self._get_batch_data(self._train_df, batch_ids, phase)
        elif phase == 'val':
            batch_ids, self._val_index = self._get_batch_ids(self._val_list, self._val_index, batch_size)
            batch_data, batch_label = self._get_batch_data(self._train_df, batch_ids, phase)
        # elif phase == 'test':
        #     batch_ids, self._test_index = self._get_batch_ids(self._test_list, self._test_index, batch_size)
        #     batch_data, _ = self._get_batch_data(self._test_df, batch_ids, phase)
        return batch_data, batch_label

if __name__ == '__main__':
    data_provider = Data1()
    data, label = data_provider.next_batch(100, 'train')
    save_path = os.path.join(Paths.output_path, 'batch_data.txt')
    np.savetxt(save_path, data, fmt='%d')
    print(label)
