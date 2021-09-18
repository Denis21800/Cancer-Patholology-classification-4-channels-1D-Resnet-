from db_manager import MongoDBManager
from copy import deepcopy
import numpy as np


class Augment(object):
    D_LEN = 0.1
    AUG_P = 0.15
    EXPAND_DATA_RATIO = 5
    VAL_SIZE = 0.5

    def __init__(self, config):
        assert config
        self.db = MongoDBManager(config)
        self.db.connect()
        self.original_data = None
        self.extracted = {}
        self.aug_data = {}
        self.label_data = {}

    def get_all_records(self):
        self.db.get_data()
        assert self.db.dataset
        return self.db.dataset

    def process(self):
        self.original_data = self.get_all_records()
        self.extract_for_mix()
        self.augment()
        print(f'Total augmented: {len(self.aug_data)}')
        self.split_set()
        print('Upload to db...')
        self.db.upload_data(self.aug_data)
        print('FINISHED!')

    def extract_for_mix(self):
        assert self.original_data

        for index in self.original_data:
            data = self.original_data.get(index)
            record = data.get('data')
            intensity = record.get('intensity')
            p_mass = record.get('pm')
            extracted = []
            for i in range(len(p_mass)):
                if p_mass[i] > 0 and intensity[i] > 0:
                    extracted.append((p_mass[i], intensity[i]))

            extract_len = int(self.D_LEN / 2 * len(extracted))
            head = extracted[0:extract_len]
            tail = extracted[-extract_len:]
            all_extracted = head + tail
            self.extracted.update({index: all_extracted})

    def augment(self):
        assert self.original_data
        assert self.extracted
        aug_index = 0

        for it in range(self.EXPAND_DATA_RATIO):
            print(f'Iteration: {it}')
            for index in self.original_data:
                data = self.original_data.get(index)
                record = data.get('data')
                intensity = record.get('intensity')
                p_mass = record.get('pm')
                file = record.get('file')
                label = record.get('label')
                is_test = record.get('is_test')
                aug_intensity = deepcopy(intensity)
                aug_p_mass = deepcopy(p_mass)
                print()
                print('-' * 32)
                print(f'Process file: {file}')
                aug_p_mass, aug_intensity = self.mix_up(aug_p_mass, aug_intensity)
                aug_file_name = f'{file}_AUG_{aug_index}'
                data_rec = {'pm': aug_p_mass,
                            'intensity': aug_intensity,
                            'label': label,
                            'is_test': is_test,
                            'file': aug_file_name
                            }

                self.aug_data.update({aug_index: {'data': data_rec}})
                if is_test == 1:
                    if label not in self.label_data:
                        self.label_data.update({label: [aug_index]})
                    else:
                        l_index = self.label_data.get(label)
                        l_index.append(aug_index)
                        self.label_data.update({label: l_index})

                aug_index += 1

    def split_set(self):
        assert self.label_data
        print()
        print('Splitting data...')
        for label in self.label_data:
            label_indices = self.label_data.get(label)
            test_size = len(label_indices)
            assert test_size > 0
            marked = self.__test_val_mark(label_indices)
            print(f'Label: {label} {marked} for val')
            print(f'Label: {label} {test_size - marked + 20} for test')

    def __test_val_mark(self, label_indices):
        marked = 0
        set_size = len(label_indices)
        to_mark = int(self.VAL_SIZE * set_size)
        mark = 2
        while marked < to_mark:
            rnd_indexes = np.random.choice(label_indices, 1)[0]
            data = self.aug_data.get(rnd_indexes)
            record = data.get('data')
            is_test = record.get('is_test')
            if is_test == 2 or is_test == 0:
                continue
            record.update({'is_test': mark})
            self.aug_data.update({rnd_indexes: {'data': record}})
            marked += 1
        return marked

    @staticmethod
    def __get_non_zero_count(p_mass, intensity):
        m_len = 0
        for i in range(len(p_mass)):
            if p_mass[i] > 0 and intensity[i] > 0:
                m_len += 1
        return m_len

    def mix_up(self, p_mass, intensity):
        m_len = self.__get_non_zero_count(p_mass, intensity)
        assert m_len > 0
        print(f'Non zero count: {m_len}')
        c_mixed = int(self.AUG_P * m_len)
        print(f'To mix: {c_mixed}')
        total_count = 0
        while total_count < c_mixed:
            rnd_pos = np.random.randint(0, len(p_mass))
            if rnd_pos >= len(p_mass) - c_mixed // 10:
                continue
            while p_mass[rnd_pos] != 0 and intensity[rnd_pos] != 0:
                rnd_pos += 1
                if rnd_pos >= len(p_mass) - 1:
                    break
            if p_mass[rnd_pos] == 0 and intensity[rnd_pos] == 0:
                random_key = np.random.randint(0, len(self.extracted))
                rnd_data = self.extracted.get(random_key)
                rnd_index = np.random.randint(0, len(rnd_data))
                mix_value = rnd_data[rnd_index]
                mix_p_mass, mix_intensity = mix_value
                p_mass[rnd_pos] = mix_p_mass
                intensity[rnd_pos] = mix_intensity
                total_count += 1

        m_len_final = self.__get_non_zero_count(p_mass, intensity)
        print(f'Non zero count (after): {m_len_final}')
        return p_mass, intensity


class Config(object):
    def __init__(self):
        self.mongo_db_name = "cancer_data_2"
        self.mongo_host = "localhost"
        self.mongo_port = 27017
        self.mongo_col_name = "data_col"


if __name__ == '__main__':
    cfg = Config()
    aug = Augment(cfg)
    aug.process()
