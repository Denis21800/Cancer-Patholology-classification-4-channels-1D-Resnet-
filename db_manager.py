from config import PipelineConfig
from pymongo import MongoClient
import numpy as np
from copy import deepcopy


class DBManager(object):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset = {}
        self.dataset = {}

    def connect(self):
        pass

    def get_data(self, labels=None):
        pass

    def upload_data(self, data):
        pass


class MongoDBManager(DBManager):
    def __init__(self, config: PipelineConfig):
        super(MongoDBManager, self).__init__(config)
        self.db_name = self.config.mongo_db_name
        self.host = self.config.mongo_host
        self.port = self.config.mongo_port
        self.col_name = self.config.mongo_col_name
        self.client = None
        self.db = None
        self.data_col = None

    def connect(self):
        self.client = MongoClient(self.host, self.port)
        self.db = self.client[self.db_name]
        self.data_col = self.db[self.col_name]

    def upload_data(self, data):
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')
            intensity_arr = data_rec.get('intensity')
            p_mass_arr = data_rec.get('pm')
            label = data_rec.get('label')
            is_test = data_rec.get('is_test')
            file = data_rec.get('file')
            if type(intensity_arr) == np.ndarray:
                intensity_arr = intensity_arr.tolist()
            if type(p_mass_arr) == np.ndarray:
                p_mass_arr = p_mass_arr.tolist()

            self.data_col.insert_one({'file': file,
                                      'label': label,
                                      'is_test': is_test,
                                      'intensity_data': intensity_arr,
                                      'p_mass_data': p_mass_arr})

    def get_data(self, labels=None):
        if labels:
            cursor = self.data_col.find({'label': {'$in': labels}})
        else:
            cursor = self.data_col.find()

        for index, rec in enumerate(cursor):
            file_ = rec.get('file')
            intensity_arr = rec.get('intensity_data')
            label_ = rec.get('label')
            is_test = rec.get('is_test')
            pm_arr = rec.get('p_mass_data')
            data_rec = {'label': label_,
                        'intensity': intensity_arr,
                        'pm': pm_arr,
                        'file': file_,
                        'is_test': is_test,
                        'metadata': None}
            self.dataset.update({index: {'data': data_rec}})


class MixedData(object):
    def __init__(self, config):
        config_to_mix = deepcopy(config)
        self.db_manager_1 = MongoDBManager(config)
        config_to_mix.mongo_db_name = 'cancer_data_2'
        self.db_manager_2 = MongoDBManager(config_to_mix)
        self.dataset = {}

    def connect(self):
        self.db_manager_1.connect()
        self.db_manager_2.connect()

    def get_data(self):
        self.db_manager_1.get_data()
        self.db_manager_2.get_data()
        dataset_1 = {}
        dataset_2 = {}
        for key in self.db_manager_1.dataset:
            data = self.db_manager_1.dataset.get(key)
            data_rec = data.get('data')
            label = data_rec.get('label')
            file = data_rec.get('file')
            if label in (2, 5):
                continue
            if label == 4:
                label = 2
            data_rec.update({'label': label})
            dataset_1.update({file: data_rec})

        for key in self.db_manager_2.dataset:
            data = self.db_manager_1.dataset.get(key)
            data_rec = data.get('data')
            file = data_rec.get('file')
            label = data_rec.get('label')
            if label == 4:
                continue
            dataset_2.update({file: data_rec})

        index = 0

        for key in dataset_1:
            if key in dataset_2:
                item_1 = dataset_1.get(key)
                intensity_1 = item_1.get('intensity')
                pm_arr_1 = item_1.get('pm')
                is_test_1 = item_1.get('is_test')
                label_1 = item_1.get('label')

                item_2 = dataset_2.get(key)
                intensity_2 = item_2.get('intensity')
                pm_arr_2 = item_2.get('pm')
                is_test_2 = item_2.get('is_test')
                label_2 = item_2.get('label')

                assert label_1 == label_2
                file = f'{key}_mixed'
                data_rec = {'label': label_1,
                            'intensity_1': intensity_1,
                            'pm_1': pm_arr_1,
                            'intensity_2': intensity_2,
                            'pm_2': pm_arr_2,
                            'file': file,
                            'is_test': is_test_1,
                            'metadata': None}
                self.dataset.update({index: {'data': data_rec}})
                index += 1
