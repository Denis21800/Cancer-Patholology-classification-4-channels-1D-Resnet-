import pandas as pd
import os


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance


class ModelOutData(object):
    predict_data = 'predict_.csv'
    original_data = 'original_.csv'
    clean_data = 'clean_.csv'
    cam_data = 'cam_.csv'

    def __init__(self, out_dir, filename):
        assert out_dir
        assert filename
        self.file = filename
        self.out_path = self.__create_dir(out_dir, f'{self.file}_out')

    @staticmethod
    def __create_dir(out_dir, directory):
        assert directory
        out_path = os.path.join(out_dir, directory)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        return out_path

    def save_prediction_result(self, data):
        assert data
        df = pd.DataFrame.from_records([data], index='File')
        outfile_path = os.path.join(self.out_path, self.predict_data)
        df.to_csv(outfile_path)

    def save_input_data(self, data, mode='original'):
        assert data
        f_ = self.original_data if mode=='original' else self.clean_data
        df = pd.DataFrame().from_dict(data)
        df = df[['pm', 'intensity']]
        outfile_path = os.path.join(self.out_path, f_)
        df.to_csv(outfile_path, index=False)

    def save_cam(self, cam, data_index):
        assert cam is not None
        df = pd.DataFrame(cam)
        outfile_path = os.path.join(self.out_path, self.cam_data)
        df.to_csv(outfile_path, index=False)

    @property
    def get_out_path(self):
        return self.out_path
