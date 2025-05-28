import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.preprocessor import BasePreprocessingPipeline
from modules.preprocessing.stats import log_data_types, log_value_counts
from modules.preprocessing.utils import (_label_encode, _one_hot_encode,
                                         _replace_values)

sys.path.append(Path(__file__).absolute().parent.parent)

class IoT_23(BasePreprocessingPipeline):

    def __init__(self, binarize=False) -> None:
        super().__init__(binarize=binarize)
        self.folder = os.path.join('datasets', 'iot_23')
        self.name = 'IoT_23'
        self.target = 'label'

    @function_call_logger
    def prepare(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = [x for x in all_folders if 'bro' in x]
        base_filename = 'conn.log.labeled'
        for folder in sub_folders:
            filename_csv = os.path.join(folder, base_filename)
            filename_parquet = filename_csv.replace('.labeled', '.parquet')
            col_names = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h',
                         'id.resp_p', 'proto', 'service', 'duration',
                         'orig_bytes', 'resp_bytes', 'conn_state',
                         'local_orig', 'local_resp', 'missed_bytes', 'history',
                         'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
                         'resp_ip_bytes', 'label']
            col_drops = ['ts', 'uid', 'service', 'local_orig', 'local_resp',
                         'history', 'id.orig_h', 'id.resp_h']
            log_print(f'Converting file \'{filename_csv}\' to parquet.')
            df = pd.read_csv(filename_csv, sep='\t',  comment='#',
                             dtype=str, na_values='', names=col_names)
            df = df.drop(columns=col_drops)
            df.to_parquet(filename_parquet)
            log_print(f'Converted file \'{filename_csv}\' to parquet.')

    @function_call_logger
    def load(self) -> None:
        work_folder = os.path.join(os.getcwd(), self.folder, 'source')
        all_folders = [x[0] for x in os.walk(work_folder)]
        sub_folders = [x for x in all_folders if 'bro' in x]
        base_filename = 'conn.log.parquet'
        data_frames = []
        for folder in sub_folders:
            full_filename = os.path.join(folder, base_filename)
            log_print(f'Started loading parquet files in \'{folder}\'.')
            df = pd.read_parquet(full_filename)
            # df = df.drop_duplicates()
            # df = df.dropna()
            data_frames.append(df)
            log_print(f'Finished loading parquet files in \'{folder}\'.')
        self.data = pd.concat(data_frames, copy=False)

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        _replace_values(self.data, 'duration',       '-',                                                       0.0)                                        # noqa
        _replace_values(self.data, 'orig_bytes',     '-',                                                       0)                                          # noqa
        _replace_values(self.data, 'resp_bytes',     '-',                                                       0)                                          # noqa
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
        if self.binarize:
            self.data[self.target] = np.where(
                (self.data[self.target] == 'Benign'), 'Benign', 'Malign'
            )
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)
