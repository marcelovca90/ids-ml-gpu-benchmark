import os

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_data_types, log_value_counts
from modules.preprocessing.utils import (_label_encode, _one_hot_encode,
                                         _replace_values)


class IoT_23(BasePreprocessingPipeline):

    def __init__(self) -> None:
        super().__init__()
        self.folder = os.path.join('datasets', 'iot_23')
        self.name = 'IoT_23'
        self.target = 'label'

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        def filter_fn(x): return ('bro' in x)
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = filter(filter_fn, all_folders)
        base_filename = 'conn.log.labeled'
        data_frames = []
        for folder in sub_folders:
            full_filename = os.path.join(folder, base_filename)
            log_print(f'Started processing folder \'{folder}\'.')
            df = pd.read_table(filepath_or_buffer=full_filename,
                               skiprows=8, nrows=None, low_memory=False)
            df.columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h',
                          'id.resp_p', 'proto', 'service', 'duration',
                          'orig_bytes', 'resp_bytes', 'conn_state',
                          'local_orig', 'local_resp', 'missed_bytes',
                          'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
                          'resp_ip_bytes', self.target]
            df = df.drop(columns=['ts', 'uid', 'service', 'local_orig',
                         'local_resp', 'history', 'id.orig_h', 'id.resp_h'])
            df = df.drop(df.tail(1).index)
            df = df.drop_duplicates()
            df = df.dropna(axis='index')
            data_frames.append(df)
            log_print(f'Finished processing folder \'{folder}\'.')
        self.data = pd.concat(data_frames, copy=False)

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        _replace_values(self.data, 'duration',   '-',                                                           np.float64(0.0))                            # noqa
        _replace_values(self.data, 'orig_bytes', '-',                                                           np.uint64(0))                               # noqa
        _replace_values(self.data, 'resp_bytes', '-',                                                           np.uint64(0))                               # noqa
        _replace_values(self.data, self.target,      '-   Malicious   Attack',                                  'Attack')                                   # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   Attack',                            'Attack')                                   # noqa
        _replace_values(self.data, self.target,      '(empty)   Benign   -',                                    'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '-   benign   -',                                          'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '-   Benign   -',                                          'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '(empty)   Benign   -',                                    'Benign')                                   # noqa
        _replace_values(self.data, self.target,      'CARhxZ3hLNVO3xYFok   Benign   -',                         'Benign')                                   # noqa
        _replace_values(self.data, self.target,      'COLnd035cNITygYHp3   Benign   -',                         'Benign')                                   # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C',                                     'C&C')                                      # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   C&C',                               'C&C')                                      # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-FileDownload',                        'C&C-FileDownload')                         # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat',                           'C&C-HeartBeat')                            # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   C&C-HeartBeat',                     'C&C-HeartBeat')                            # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat-Attack',                    'C&C-HeartBeat-Attack')                     # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat-FileDownload',              'C&C-HeartBeat-FileDownload')               # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-HeartBeat-PartOfAHorizontalPortScan', 'C&C-HeartBeat-PartOfAHorizontalPortScan')  # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-Mirai',                               'C&C-Mirai')                                # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-PartOfAHorizontalPortScan',           'C&C-PartOfAHorizontalPortScan')            # noqa
        _replace_values(self.data, self.target,      '-   Malicious   C&C-Torii',                               'C&C-Torii')                                # noqa
        _replace_values(self.data, self.target,      '-   Malicious   DDoS',                                    'DDoS')                                     # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   DDoS',                              'DDoS')                                     # noqa
        _replace_values(self.data, self.target,      '-   Malicious   FileDownload',                            'FileDownload')                             # noqa
        _replace_values(self.data, self.target,      '-   Malicious   Okiru',                                   'Okiru')                                    # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   Okiru',                             'Okiru')                                    # noqa
        _replace_values(self.data, self.target,      '-   Malicious   Okiru-Attack',                            'Okiru-Attack')                             # noqa
        _replace_values(self.data, self.target,      '-   Malicious   PartOfAHorizontalPortScan',               'PartOfAHorizontalPortScan')                # noqa
        _replace_values(self.data, self.target,      '(empty)   Malicious   PartOfAHorizontalPortScan',         'PartOfAHorizontalPortScan')                # noqa
        _replace_values(self.data, self.target,      '-   Malicious   PartOfAHorizontalPortScan-Attack',        'PartOfAHorizontalPortScan-Attack')         # noqa
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)
        self.data.drop_duplicates()
        self.data.dropna(axis='index')

    @function_call_logger
    def set_dtypes(self) -> None:
        log_print('Data types before inference and manual definition:')
        log_data_types(self.data)
        self.data = self.data.infer_objects().astype({
            'id.orig_p': np.uint64,
            'id.resp_p': np.uint64,
            'duration': np.float64,
            'orig_bytes': np.uint64,
            'resp_bytes': np.uint64,
            'missed_bytes': np.uint64,
            'orig_pkts': np.uint64,
            'orig_ip_bytes': np.uint64,
            'resp_pkts': np.uint64,
            'resp_ip_bytes': np.uint64,
            self.target: 'category'
        })
        log_print('Data types after inference and manual definition:')
        log_data_types(self.data)

    @function_call_logger
    def encode(self) -> None:
        self.data = _one_hot_encode(self.data, 'proto')
        self.data, _ = _label_encode(self.data, 'conn_state')
        self.data, self.mappings = _label_encode(self.data, self.target)
