import numpy as np
from pyteomics import mgf
from preprocessor import PreprocessingData
import os


class MGFParser(object):
    def __init__(self):
        self.data = {}
        self.preprocessor = PreprocessingData()
        self.index = 0

    def load_mgf(self, file, label, is_test=0):
        reader = mgf.read(file)
        data_pm = []
        data_intensity = []
        data_rti = []
        print(f'Parsing file: {file}...')
        for spectrum in reader:
            params = spectrum.get('params')
            p_mass_intensity_params = params.get('pepmass')
            rti = params.get('rtinseconds')
            p_mass = p_mass_intensity_params[0]
            intensity = p_mass_intensity_params[1]
            p_mass = np.round(self.preprocessor.calculate_ppm(p_mass), 2)
            try:
                intensity = np.round(intensity, 2)
            except TypeError:
                intensity = 0
            data_pm.append(p_mass)
            data_intensity.append(intensity)
            data_rti.append(rti)
        file_name_key = os.path.basename(file)
        data_rec = {'pm': data_pm,
                    'intensity': data_intensity,
                    'rti': data_rti,
                    'label': label,
                    'is_test': is_test,
                    'file': file_name_key
                    }
        self.data.update({self.index: {'data': data_rec}})
        self.index += 1

    def load_directory(self, directory, labels, val_len=-1):
        assert os.path.exists(directory)

        for d_ in os.walk(directory):
            dir_to_parse = d_[0]
            if not dir_to_parse:
                continue
            val_count = 0
            for file in os.listdir(dir_to_parse):
                filename = os.fsdecode(file)
                if filename.endswith('.mgf'):
                    label = -1
                    for key in labels:
                        item = labels[key]
                        if item in dir_to_parse:
                            label = key
                    is_test = 1 if val_count < val_len or val_len == -1 else 0
                    current_file = os.path.join(dir_to_parse, file)
                    self.load_mgf(current_file, label=label, is_test=is_test)
                    val_count += 1
