from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope
import pickle
import numpy as np


class PreprocessingData(object):
    def __init__(self):
        self.outlier_detector = None
        self.scale = MinMaxScaler()
        self.all_data = None

    @staticmethod
    def calculate_ppm(value, err=10):
        bound = err * value / 1000000
        l_bound = round(value - bound, 6)
        r_bound = round(value + bound, 6)
        return l_bound + r_bound / 2

    def get_all_data(self, data):
        intensity_data = []
        p_mass_data = []
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')

            intensity_ = data_rec.get('intensity')
            pm = data_rec.get('pm')
            intensity_data += intensity_
            p_mass_data += pm
        p_mass_arr = np.array(p_mass_data)
        intensity_arr = np.array(intensity_data)
        self.all_data = np.vstack((p_mass_arr, intensity_arr)).T

    def create_scale_object(self, data, log):
        if self.all_data is None:
            self.get_all_data(data)
        if log:
            all_p_pass = self.all_data[:, 0]
            all_intensity = self.all_data[:, 1]
            log_p_pass = np.log(all_p_pass[all_p_pass > 0])
            log_intensity = np.log(all_intensity[all_intensity > 0])
            all_p_pass[all_p_pass > 0] = log_p_pass
            all_intensity[all_intensity > 0] = log_intensity
            log_arr = np.vstack((all_p_pass, all_intensity)).T
            self.scale.fit(log_arr)
        else:
            self.scale.fit(self.all_data)

    def create_outlier(self, data, contamination):
        assert data
        if self.all_data is None:
            self.get_all_data(data)
        self.outlier_detector = EllipticEnvelope(contamination=contamination)
        self.outlier_detector.fit(self.all_data)

    def elliptic_envelope(self, data):
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')
            intensity_arr = data_rec.get('intensity')
            p_mass_arr = data_rec.get('pm')
            rti_arr = data_rec.get('rti')
            if type(intensity_arr) == list:
                intensity_arr = np.array(intensity_arr)
            if type(p_mass_arr) == list:
                p_mass_arr = np.array(p_mass_arr)
            if type(rti_arr) == list:
                rti_arr = np.array(rti_arr)
            record_data = np.vstack((p_mass_arr, intensity_arr)).T
            outlier = self.outlier_detector.predict(record_data)
            removed_indices = np.where(outlier == -1)
            p_mass_arr = np.delete(p_mass_arr, removed_indices)
            intensity_arr = np.delete(intensity_arr, removed_indices)
            rti_arr = np.delete(rti_arr, removed_indices)
            data_rec.update({
                'intensity': intensity_arr,
                'pm': p_mass_arr,
                'rti': rti_arr
            })

    def mm_scale(self, data, log=True):
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')
            intensity_arr = data_rec.get('intensity')
            p_mass_arr = data_rec.get('pm')
            if type(intensity_arr) == list:
                intensity_arr = np.array(intensity_arr)
            if type(p_mass_arr) == list:
                p_mass_arr = np.array(p_mass_arr)

            if log:
                log_p_mass_arr = np.log(p_mass_arr[p_mass_arr > 0])
                log_intensity_arr = np.log(intensity_arr[intensity_arr > 0])
                p_mass_arr[p_mass_arr > 0] = log_p_mass_arr
                intensity_arr[intensity_arr > 0] = log_intensity_arr
            record_data = np.vstack((p_mass_arr, intensity_arr)).T
            record_data = self.scale.transform(record_data)
            data_rec.update({
                'intensity': record_data[:, 1],
                'pm': record_data[:, 0]
            })

    @staticmethod
    def align_data_to_rti(data, bound, step):
        assert data
        assert bound > 0
        assert step > 0
        align_size = int(bound / step)
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')
            intensity_arr = data_rec.get('intensity')
            p_mass_arr = data_rec.get('pm')
            rti_arr = data_rec.get('rti')
            if type(intensity_arr) == list:
                intensity_arr = np.array(intensity_arr)
            if type(p_mass_arr) == list:
                p_mass_arr = np.array(p_mass_arr)
            if type(rti_arr) == list:
                rti_arr = np.array(rti_arr)

            intensity_aligned = np.zeros(align_size)
            p_mass_aligned = np.zeros(align_size)
            original_index = np.zeros(align_size)

            for i in range(len(rti_arr)):
                index = rti_arr[i] / step
                index = int(np.round(index, 0))
                intensity_aligned[index] = intensity_arr[i]
                p_mass_aligned[index] = p_mass_arr[i]
                original_index[index] = i

            data_rec.update({
                'intensity': intensity_aligned,
                'pm': p_mass_aligned,
                'o_index': original_index
            })

    def save_outlier_detector(self, model_path):
        assert self.outlier_detector
        assert model_path
        with open(model_path, 'wb') as file_:
            pickle.dump(self.outlier_detector, file_)

    def save_mm_scale(self, model_path):
        assert self.scale
        assert model_path
        with open(model_path, 'wb') as file_:
            pickle.dump(self.scale, file_)

    def load_outlier_detector(self, model_path):
        assert model_path
        with open(model_path, 'rb') as file_:
            self.outlier_detector = pickle.load(file_)

    def load_mm_scale(self, model_path):
        assert self.scale
        assert model_path
        with open(model_path, 'rb') as file_:
            self.scale = pickle.load(file_)
