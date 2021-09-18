import os
from pathlib import Path

from config import PipelineConfig
from pymongo import MongoClient
import numpy as np
from copy import deepcopy
import cv2
from PIL import Image
import librosa.display
import matplotlib.pyplot as plt


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


class ConvertData(object):
    IMG_SIZE = 512

    def __init__(self, cfg_path, out_dir):
        config = PipelineConfig(cfg_path)
        self.db_manager = MongoDBManager(config)
        self.out_dir = out_dir

    def connect(self):
        self.db_manager.connect()
        self.db_manager.connect()

    def convert_data(self):
        self.db_manager.get_data()
        for key in self.db_manager.dataset:
            data = self.db_manager.dataset.get(key)
            data_rec = data.get('data')
            label = data_rec.get('label')
            file = data_rec.get('file')
            if 'AUG' in file:
                continue
            pm = data_rec.get('pm')
            intensity = data_rec.get('intensity')
            image_seq = self.get_img_seq(pm, intensity)
          #  self.save_image_seq(image_seq, file, label)

    def save_image_seq(self, image_seq, file, label):
        out_dir_l1 = self.out_dir / str(label)
        if not os.path.exists(out_dir_l1):
            os.makedirs(out_dir_l1)
        out_dir_l2 = out_dir_l1 / file
        if not os.path.exists(out_dir_l2):
            os.makedirs(out_dir_l2)

        for idx, item in enumerate(image_seq):
            out_file = out_dir_l2 / f'{idx}.png'
            cv2.imwrite(str(out_file), item)

    def get_img_seq(self, mz, intensity):
        index = 0
        chunks = []
        while index <= len(mz):
            mz_chunk = mz[index: index + self.IMG_SIZE]
            intensity_chunk = intensity[index: index + self.IMG_SIZE]
            index += self.IMG_SIZE
            chunk_image = self.get_img_data(mz_chunk, intensity_chunk)
            chunks.append(chunk_image)

        return chunks

    def get_img_data(self, mz_chunk, intensity_chunk):
        img_buffer = np.zeros((self.IMG_SIZE, self.IMG_SIZE))
        mz_chunk = (np.array(mz_chunk) * self.IMG_SIZE).astype(np.uint8)
        intensity_chunk = (np.array(intensity_chunk))

        for i, mz in enumerate(mz_chunk):
            img_buffer[mz, i] = intensity_chunk[i]

        img = Image.fromarray(img_buffer)
        imcv = cv2.cvtColor((np.array(img) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        imcv = cv2.applyColorMap(imcv, cv2.COLORMAP_JET)
        # cv2.imshow('sample', imcv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return imcv


config_path = './config/config_orig.json'
OUT_DIR = Path('/home/dp/Data/CancerImg/')
if __name__ == '__main__':
    convert = ConvertData(config_path, OUT_DIR)
    convert.connect()
    convert.convert_data()
