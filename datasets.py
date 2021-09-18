import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ModelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data.get(index)
        rec = item.get('data')
        pm_arr_1 = rec.get('pm_1')
        intensity_arr_1 = rec.get('intensity_1')
        x_data_1 = np.vstack((pm_arr_1, intensity_arr_1))
        pm_arr_2 = rec.get('pm_2')
        intensity_arr_2 = rec.get('intensity_2')
        x_data_2 = np.vstack((pm_arr_2, intensity_arr_2))
        x_data = np.vstack((x_data_1, x_data_2))
        y_data = rec.get('label')
        file_ = rec.get('file')
        o_index = rec.get('o_index')
        metadata = None
        o_index = o_index if o_index is not None else []
        return x_data, y_data, file_, o_index

    def __len__(self):
        return len(self.data)


class ModelData(object):
    def __init__(self, data):
        assert data
        self.data = data
        self.train_loader = None
        self.test_loader = None
        self.all_data_loader = None
        self.val_loader = None

    def create_model_data(self):
        train_data = {}
        test_data = {}
        val_data = {}
        test_index = 0
        train_index = 0
        val_index = 0
        for key in self.data:
            item = self.data.get(key)
            rec = item.get('data')
            is_test = rec.get('is_test')
            if is_test == 1:
                test_data.update({test_index: item})
                test_index += 1
            elif is_test == 0:
                train_data.update({train_index: item})
                train_index += 1
            elif is_test == 2:
                val_data.update({val_index: item})
                val_index += 1
        test_dataset = ModelDataset(test_data)
        train_dataset = ModelDataset(train_data)
        val_dataset = ModelDataset(val_data)
        all_data = ModelDataset(self.data)
        if train_dataset:
            self.train_loader = DataLoader(dataset=train_dataset, shuffle=True)
        if test_dataset:
            self.test_loader = DataLoader(dataset=test_dataset, shuffle=True)
        self.all_data_loader = DataLoader(dataset=all_data, shuffle=True)
        if val_dataset:
            self.val_loader = DataLoader(dataset=val_dataset, shuffle=True)
